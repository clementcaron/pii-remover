from __future__ import annotations

import collections
from gliner import GLiNER
from src.engine.base import AnonymizerBase
from src.proxy.state import state_manager
from src.core.logger import logger

class NuNERAnonymizer(AnonymizerBase):
    """
    PII detection and anonymization using NuNER/GLiNER zero-shot model via standard gliner library (CPU).
    """
    def __init__(self, model_path: str = "numind/NuNER_Zero"):
        # Labels we want NuNER to look for.
        self.labels = [
            "person", 
            "email", 
            "phone number", 
            "credit card", 
            "cryptocurrency address", 
            "ip address", 
            "location", 
            "iban", 
            "social security number", 
            "date",
            "organization",
            "company"
        ]
        
        # Mapping from NuNER labels to our standard internal tokens
        self.label_mapping = {
            "person": "PERSON",
            "email": "EMAIL_ADDRESS",
            "phone number": "PHONE_NUMBER",
            "credit card": "CREDIT_CARD",
            "cryptocurrency address": "CRYPTO",
            "ip address": "IP_ADDRESS",
            "location": "LOCATION",
            "iban": "IBAN_CODE",
            "social security number": "US_SSN",
            "date": "DATE_TIME",
            "organization": "ORG",
            "company": "ORG"
        }

        from src.core.config import settings
        import torch

        device = "cpu"
        if settings.USE_GPU:
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("CUDA is available. Enabling GPU acceleration for NuNER.")
            else:
                logger.warning("USE_GPU=true but CUDA is not available. Falling back to CPU.")
        
        logger.info(f"Loading NuNER model from {model_path} via gliner ({device})...")
        try:
            self.model = GLiNER.from_pretrained(model_path)
            self.model.to(device)
            logger.info("NuNER model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load NuNER model: {e}")
            raise

        self.last_timing = {}

    def _merge_entities(self, entities: list[dict], text: str) -> list[dict]:
        """
        Consolidates adjacent or overlapping entities of the same label into a single entity.
        Higher threshold sometimes causes fragmentation; this merges them back.
        """
        if not entities:
            return entities
        
        # Sort by start position
        sorted_ents = sorted(entities, key=lambda x: x['start'])
        merged = []
        
        curr = dict(sorted_ents[0])
        for next_ent in sorted_ents[1:]:
            # Check if same label and adjacent/overlapping
            # We allow for a 1-character gap to handle potential tokenization issues
            if next_ent['label'] == curr['label'] and next_ent['start'] <= curr['end'] + 1:
                curr['end'] = max(curr['end'], next_ent['end'])
                curr['text'] = text[curr['start']:curr['end']]
                curr['score'] = (curr.get('score', 0) + next_ent.get('score', 0)) / 2
            else:
                merged.append(curr)
                curr = dict(next_ent)
        merged.append(curr)
        return merged


    def anonymize_batch(self, texts: list[str], request_id: str) -> list[str]:
        if not texts:
            return texts

        import time
        start_inference = time.perf_counter()
        
        # Filter out empty or whitespace-only strings
        valid_indices = [i for i, t in enumerate(texts) if t and t.strip()]
        valid_texts = [texts[i] for i in valid_indices]
        
        if not valid_texts:
            return texts

        # 1. NuNER Inference
        try:
            # GLiNER.inference is the replacement for batch_predict_entities
            # We use threshold 0.5 for a better balance between precision and recall
            batch_entities = self.model.inference(valid_texts, self.labels, flat_ner=True, threshold=0.5)
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            # Fallback to individual calls if batch fails
            batch_entities = [self.model.predict_entities(t, self.labels, threshold=0.5) for t in valid_texts]

        inference_duration = time.perf_counter() - start_inference
        
        start_processing = time.perf_counter()
        results = list(texts)
        
        # Global counters for the whole batch to ensure unique tokens per request
        counters = collections.defaultdict(int)
        
        for i, ner_entities in zip(valid_indices, batch_entities):
            text = texts[i]
            
            # Post-process: Merge fragmented entities
            # NuNER labels are mapped to internal labels
            mapped_ner = []
            for ent in ner_entities:
                label = ent['label']
                internal_label = self.label_mapping.get(label, label.upper())
                mapped_ner.append({**ent, 'label': internal_label})
            
            # Merge fragments
            entities = self._merge_entities(mapped_ner, text)
            
            # Sort entities by start position descending to avoid index shift
            entities = sorted(entities, key=lambda x: x['start'], reverse=True)
            
            anonymized_text = text
            for ent in entities:
                label = ent['label']
                start = ent['start']
                end = ent['end']
                original_pii = ent['text']
                
                counters[label] += 1
                token = f"[{label}_{counters[label]}]"
                
                logger.debug(f"[{request_id}] Detected PII ({label}): '{original_pii}' -> {token}")
                state_manager.set_mapping(request_id, token, original_pii)
                anonymized_text = anonymized_text[:start] + token + anonymized_text[end:]
            
            results[i] = anonymized_text

        processing_duration = time.perf_counter() - start_processing
        
        self.last_timing = {
            "inference_ms": (inference_duration * 1000) / len(valid_texts),
            "processing_ms": (processing_duration * 1000) / len(valid_texts),
            "total_inference_ms": inference_duration * 1000,
            "total_processing_ms": processing_duration * 1000,
            "batch_size": len(valid_texts)
        }
            
        return results

    def anonymize(self, text: str, request_id: str) -> str:
        return self.anonymize_batch([text], request_id)[0]

    def deanonymize(self, text: str, request_id: str) -> str:
        if not text:
            return text

        mapping = state_manager.get_mapping(request_id)
        if not mapping:
            return text

        # Simple replacement for non-streaming de-anonymization
        # Sort by length descending to avoid partial replacements if tokens overlap
        for token, original in sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True):
            text = text.replace(token, original)
        
        return text
