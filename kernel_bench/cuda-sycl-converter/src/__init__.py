"""
CUDA-SYCL Kernel Accuracy Test Suite

A comprehensive test suite for validating CUDA to SYCL kernel conversions,
specifically targeting the LCZero chess engine neural network operations.
"""

__version__ = "0.5.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__license__ = "GPL-3.0"

# Version info
VERSION_INFO = {
    "version": "0.5.0",
    "release_date": "2024-03-16",
    "python_requires": ">=3.8",
}

# Import main components (lazy loading to avoid circular imports)
def get_tester():
    """Get KernelAccuracyTester class"""
    from .core.tester import KernelAccuracyTester
    return KernelAccuracyTester

def get_harnesses():
    """Get all kernel harnesses"""
    from .harnesses.all_harnesses import ALL_HARNESSES
    from .harnesses.batch4_harnesses import PHASE5_BATCH4_HARNESSES
    
    all_harnesses = {}
    all_harnesses.update(ALL_HARNESSES)
    all_harnesses.update(PHASE5_BATCH4_HARNESSES)
    return all_harnesses

__all__ = [
    "get_tester",
    "get_harnesses",
    "VERSION_INFO",
]
