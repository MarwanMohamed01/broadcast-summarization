"""
Tiny helper to measure current-process RSS memory in MB.

Uses psutil if available, falls back to zero. Avoids hard dependency
on psutil so the pipeline still runs if it is not installed.
"""
import os


def current_rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0
