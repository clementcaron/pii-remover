from __future__ import annotations

import re
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, RecognizerRegistry, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from src.engine.base import AnonymizerBase
from src.proxy.state import state_manager

import phonenumbers
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, RecognizerRegistry, Pattern, EntityRecognizer, RecognizerResult

class InternationalPhoneRecognizer(EntityRecognizer):
    """
    Custom recognizer that uses 'phonenumbers' to detect phone numbers from all countries accurately.
    """
    def __init__(self):
        super().__init__(supported_entities=["PHONE_NUMBER"])

    def analyze(self, text, entities, nlp_artifacts=None):
        results = []
        # Find matches using Google's libphonenumber logic (handles international formats)
        for match in phonenumbers.PhoneNumberMatcher(text, None):
            results.append(
                RecognizerResult(
                    entity_type="PHONE_NUMBER",
                    start=match.start,
                    end=match.end,
                    score=0.8
                )
            )
        return results

class RegexNLPAnonymizer(AnonymizerBase):
    """
    PII detection and anonymization using NLP, Regex, and International Phone Parsing.
    """
    def __init__(self):
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers()

        # 1. Add International Phone Recognizer
        registry.add_recognizer(InternationalPhoneRecognizer())

        # 2. Add a "Fuzzy Email" recognizer for messy formats (e.g., john[at]email.com)
        fuzzy_email_pattern = Pattern(
            name="fuzzy_email", 
            regex=r"\b[A-Za-z0-9._%+-]+(?:\s*\[at\]\s*|\s*\(at\)\s*|@)[A-Za-z0-9.-]+(?:\s*\[dot\]\s*|\s*\(dot\)\s*|\.)[A-Z|a-z]{2,}\b", 
            score=0.6
        )
        fuzzy_email_recognizer = PatternRecognizer(supported_entity="EMAIL_ADDRESS", patterns=[fuzzy_email_pattern])
        registry.add_recognizer(fuzzy_email_recognizer)

        # 3. Add a Broad Phone Regex (for numbers without delimiters like 5550123)
        # We give this a lower score so it only 'wins' if context or other rules agree
        broad_phone_pattern = Pattern(
            name="broad_phone",
            regex=r"\b\d{7,15}\b",
            score=0.3
        )
        broad_phone_recognizer = PatternRecognizer(supported_entity="PHONE_NUMBER", patterns=[broad_phone_pattern])
        registry.add_recognizer(broad_phone_recognizer)

        self.analyzer = AnalyzerEngine(registry=registry)
        self.anonymizer = AnonymizerEngine()
        # Extended "Catch All" list of entities
        self.entities = [
            "PERSON", 
            "EMAIL_ADDRESS", 
            "PHONE_NUMBER", 
            "CREDIT_CARD", 
            "CRYPTO", 
            "IP_ADDRESS", 
            "LOCATION", 
            "IBAN_CODE", 
            "US_SSN", 
            "DATE_TIME"
        ]

    def anonymize(self, text: str, request_id: str) -> str:
        if not text:
            return text

        # Analyze text for PII
        results = self.analyzer.analyze(text=text, entities=self.entities, language='en')

        # Custom counter to create unique tokens like [PERSON_1]
        counters = collections.defaultdict(int)
        
        def operator_callback(pii_text: str, entity_type: str) -> str:
            counters[entity_type] += 1
            token = f"[{entity_type}_{counters[entity_type]}]"
            # Store mapping for de-anonymization
            state_manager.set_mapping(request_id, token, pii_text)
            return token

        # Anonymize using the custom operator
        operators = {
            entity: OperatorConfig("custom", {"lambda": lambda x, et=entity: operator_callback(x, et)})
            for entity in self.entities
        }

        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=operators
        )

        return anonymized_result.text

    def deanonymize(self, text: str, request_id: str) -> str:
        if not text:
            return text

        mapping = state_manager.get_mapping(request_id)
        if not mapping:
            return text

        # Simple replacement for non-streaming de-anonymization
        # Sort by length descending to avoid partial replacements if tokens overlap (unlikely here)
        for token, original in sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True):
            text = text.replace(token, original)
        
        return text

import collections
