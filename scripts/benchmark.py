from __future__ import annotations

import time
import os
import uuid
import psutil
import statistics

from src.engine.regex_nlp import RegexNLPAnonymizer
from src.engine.nuner import NuNERAnonymizer

SAMPLE_TEXTS = [
    "My name is John Doe, I live in New York and my email is john.doe@example.com.",
    "Call me at +1-555-0123. My credit card is 1234-5678-9012-3456.",
    "The IBAN for the transfer is DE89370400440532013000. Date: 2023-10-25.",
    "Bitcoin address: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa. IP: 192.168.1.1",
    "Social security number: 000-00-0000. My office is located in Paris, France."
]

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # MB

def run_benchmark(engine, name, iterations=10):
    print(f"\nBenchmarking {name}...")
    total_latencies = []
    inference_times = []
    processing_times = []
    
    start_total_bench = time.perf_counter()
    for _ in range(iterations):
        for text in SAMPLE_TEXTS:
            req_id = str(uuid.uuid4())
            start_full = time.perf_counter()
            engine.anonymize(text, req_id)
            duration = time.perf_counter() - start_full
            total_latencies.append(duration)
            
            if hasattr(engine, 'last_timing'):
                inference_times.append(engine.last_timing.get("inference_ms", 0))
                processing_times.append(engine.last_timing.get("processing_ms", 0))
    
    end_total_bench = time.perf_counter()
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
    
    # Benchmark Regex/Spacy (Old)
    try:
        old_engine = RegexNLPAnonymizer()
        results.append(run_benchmark(old_engine, "RegexNLP (Spacy)"))
    except Exception as e:
        print(f"Error benchmarking RegexNLP: {e}")

    # Benchmark NuNER (New)
    try:
        new_engine = NuNERAnonymizer()
        results.append(run_benchmark(new_engine, "NuNER (Sequential)"))
        
        # Parallel NuNER
        print("\nBenchmarking NuNER (Parallel Threads)...")
        from concurrent.futures import ThreadPoolExecutor
        start_parallel = time.perf_counter()
        req_id = str(uuid.uuid4())
        texts = SAMPLE_TEXTS * 2
        with ThreadPoolExecutor(max_workers=len(texts)) as executor:
            list(executor.map(lambda t: new_engine.anonymize(t, req_id), texts))
        duration = (time.perf_counter() - start_parallel) * 1000
        results.append({
            "Engine": "NuNER (Parallel x10)",
            "Avg Total (ms)": f"{duration:.2f} (total)",
            "Inference (ms)": "N/A",
            "Processing (ms)": "N/A",
            "P95 Total (ms)": "N/A",
            "Mem (MB)": f"{get_memory_usage():.2f}",
        })
        
        # Batch NuNER
        print("\nBenchmarking NuNER (Batch)...")
        start_batch = time.perf_counter()
        # Simulate a large batch
        new_engine.anonymize_batch(texts, req_id)
        duration = (time.perf_counter() - start_batch) * 1000
        
        results.append({
            "Engine": "NuNER (Batch x10)",
            "Avg Total (ms)": f"{duration:.2f} (total)",
            "Inference (ms)": f"{new_engine.last_timing.get('total_inference_ms', 0):.2f}",
            "Processing (ms)": f"{new_engine.last_timing.get('total_processing_ms', 0):.2f}",
            "P95 Total (ms)": "N/A",
            "Mem (MB)": f"{get_memory_usage():.2f}",
        })
    except Exception as e:
        print(f"Error benchmarking NuNER: {e}")

    print("\n--- Detailed Benchmark Results ---")
    headers = ["Engine", "Avg Total (ms)", "Inference (ms)", "Processing (ms)", "P95 Total (ms)", "Mem (MB)"]
    data = [[r[h] for h in headers] for r in results]
    print(tabulate(data, headers=headers, tablefmt="github"))
