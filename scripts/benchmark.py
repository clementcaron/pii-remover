from __future__ import annotations

import time
import os
import uuid
import psutil
import statistics

from src.engine.nuner import NuNERAnonymizer

GOLD_STANDARD_DATASET = [
    {
        "text": "My name is John Doe, I live in New York and my email is john.doe@example.com.",
        "entities": ["John Doe", "New York", "john.doe@example.com"]
    },
    {
        "text": "Call me at +1-555-0123. My credit card is 1234-5678-9012-3456.",
        "entities": ["+1-555-0123", "1234-5678-9012-3456"]
    },
    {
        "text": "The IBAN for the transfer is DE89370400440532013000. Date: 2023-10-25.",
        "entities": ["DE89370400440532013000", "2023-10-25"]
    },
    {
        "text": "Bitcoin address: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa. IP: 192.168.1.1",
        "entities": ["1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", "192.168.1.1"]
    },
    {
        "text": "Social security number: 000-00-0000. My office is located in Paris, France.",
        "entities": ["000-00-0000", "Paris, France"]
    },
    {
        "text": "Social security number: 000-00-0000. My office is located in Paris, France.",
        "entities": ["000-00-0000", "Paris, France"]
    }
]

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # MB

def calculate_recall(engine, dataset):
    correct_detections = 0
    total_entities = 0
    
    for item in dataset:
        req_id = str(uuid.uuid4())
        anonymized = engine.anonymize(item["text"], req_id)
        
        # Check if original entities are still present in anonymized text
        for entity in item["entities"]:
            total_entities += 1
            if entity not in anonymized:
                correct_detections += 1
            else:
                print(f"FAILED RECALL: Engine '{engine.__class__.__name__}' missed '{entity}' in text: '{item['text'][:50]}...'")
                print(f"               Anonymized text: '{anonymized[:100]}...'")
                
    return (correct_detections / total_entities) * 100 if total_entities > 0 else 0

def run_benchmark(engine, name, iterations=5):
    print(f"\nBenchmarking {name}...")
    total_latencies = []
    inference_times = []
    processing_times = []
    
    # Measure Recall first
    recall = calculate_recall(engine, GOLD_STANDARD_DATASET)
    
    for _ in range(iterations):
        for item in GOLD_STANDARD_DATASET:
            text = item["text"]
            req_id = str(uuid.uuid4())
            start_full = time.perf_counter()
            engine.anonymize(text, req_id)
            duration = time.perf_counter() - start_full
            total_latencies.append(duration)
            
            if hasattr(engine, 'last_timing'):
                inference_times.append(engine.last_timing.get("inference_ms", 0))
                processing_times.append(engine.last_timing.get("processing_ms", 0))
    
    mem_after = get_memory_usage()
    
    avg_total = statistics.mean(total_latencies) * 1000
    p95_total = statistics.quantiles(total_latencies, n=20)[18] * 1000
    
    avg_inference = statistics.mean(inference_times) if inference_times else 0
    avg_processing = statistics.mean(processing_times) if processing_times else 0
    
    return {
        "Engine": name,
        "Avg Total (ms)": f"{avg_total:.2f}",
        "Inference (ms)": f"{avg_inference:.2f}",
        "Processing (ms)": f"{avg_processing:.2f}",
        "P95 Total (ms)": f"{p95_total:.2f}",
        "Mem (MB)": f"{mem_after:.2f}",
        "Recall (%)": f"{recall:.1f}%",
    }

if __name__ == "__main__":
    # Ensure tabulate is installed or use simple print
    try:
        from tabulate import tabulate
    except ImportError:
        def tabulate(data, headers, tablefmt):
            header_line = " | ".join(headers)
            print(header_line)
            print("-" * len(header_line))
            for row in data:
                print(" | ".join(map(str, row)))
            return ""

    results = []
    
    # Regex/Spacy engine was removed in V2

    # Benchmark NuNER (New)
    try:
        from src.engine.nuner import NuNERAnonymizer
        new_engine = NuNERAnonymizer()
        results.append(run_benchmark(new_engine, "NuNER (Sequential)"))
        
        # Batch NuNER
        print("\nBenchmarking NuNER (Batch)...")
        start_batch = time.perf_counter()
        texts = [item["text"] for item in GOLD_STANDARD_DATASET] * 2
        req_id = str(uuid.uuid4())
        new_engine.anonymize_batch(texts, req_id)
        duration = (time.perf_counter() - start_batch) * 1000
        
        results.append({
            "Engine": "NuNER (Batch x10)",
            "Avg Total (ms)": f"{duration:.2f} (total)",
            "Inference (ms)": f"{new_engine.last_timing.get('total_inference_ms', 0):.2f}",
            "Processing (ms)": f"{new_engine.last_timing.get('total_processing_ms', 0):.2f}",
            "P95 Total (ms)": "N/A",
            "Mem (MB)": f"{get_memory_usage():.2f}",
            "Recall (%)": "N/A", # Batch recall is the same as sequential
        })
    except Exception as e:
        print(f"Error benchmarking NuNER: {e}")

    print("\n--- Detailed Benchmark Results ---")
    headers = ["Engine", "Avg Total (ms)", "Inference (ms)", "Processing (ms)", "Recall (%)", "Mem (MB)"]
    data = [[r.get(h, "N/A") for h in headers] for r in results]
    print(tabulate(data, headers=headers, tablefmt="github"))
