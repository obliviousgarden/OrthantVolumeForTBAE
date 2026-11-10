"""Independent diagnostic memory profiling. NEVER used inside timed_warm.

Reports the peak *incremental* RSS observed by a 2-ms observer thread and
tracemalloc's peak traced allocation. The former is sampling-dependent,
particularly for microsecond functions; the latter is not equivalent to RSS
and can omit memory managed directly by native C/Fortran libraries.
"""
from __future__ import annotations

import math
import threading
import tracemalloc
from collections.abc import Callable

import psutil

SAMPLE_SECONDS = 0.002


def memory_once(fn: Callable, /, *args, **kwargs) -> tuple[object, dict]:
    """Run a single un-timed evaluation, return (value, memory fields in bytes).

    Profiling operates AFTER the primary timing, or during the final Kmax
    accuracy scan when the Aono method cannot meet the common error target.
    """
    process = psutil.Process()
    idle_rss = process.memory_info().rss
    peak_rss = [idle_rss]
    stop = threading.Event()

    def sample_rss():
        while not stop.is_set():
            rss = process.memory_info().rss
            if rss > peak_rss[0]:
                peak_rss[0] = rss
            stop.wait(SAMPLE_SECONDS)

    tracemalloc.start()
    observer = threading.Thread(target=sample_rss, daemon=True)
    observer.start()
    try:
        result = fn(*args, **kwargs)
        _, peak_traced = tracemalloc.get_traced_memory()
    finally:
        stop.set()
        observer.join()
        try:
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
        finally:
            tracemalloc.stop()
    return result, {
        'peak_rss_bytes': int(peak_rss[0]),
        'baseline_rss_bytes': int(idle_rss),
        'peak_rss_delta_bytes': int(max(0, peak_rss[0] - idle_rss)),
        'peak_traced_bytes': int(peak_traced),
        'rss_sampling_interval_ms': 1000*SAMPLE_SECONDS,
    }


def _memory_fields(prefix: str, result: dict) -> dict:
    """Stable field names, friendly units. Both diagnostics use bytes/MB."""
    d = {
        f'{prefix}_memory_peak_rss_bytes': result['peak_rss_bytes'],
        f'{prefix}_memory_baseline_rss_bytes': result['baseline_rss_bytes'],
        f'{prefix}_memory_peak_rss_delta_bytes': result['peak_rss_delta_bytes'],
        f'{prefix}_memory_peak_traced_bytes': result['peak_traced_bytes'],
    }
    d[f'{prefix}_memory_peak_rss_delta_MiB'] = result['peak_rss_delta_bytes']/1048576
    d[f'{prefix}_memory_peak_traced_MiB'] = result['peak_traced_bytes']/1048576
    return d


def memory_fields(prefix:str, result: dict | None) -> dict:
    if result is None:
        return {key: math.nan for key in (
            f'{prefix}_memory_peak_rss_bytes',
            f'{prefix}_memory_baseline_rss_bytes',
            f'{prefix}_memory_peak_rss_delta_bytes',
            f'{prefix}_memory_peak_traced_bytes',
            f'{prefix}_memory_peak_rss_delta_MiB',
            f'{prefix}_memory_peak_traced_MiB',
        )}
    return _memory_fields(prefix, result)
