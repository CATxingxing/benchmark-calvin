"""Collect NumPy's package-local binaries without requiring conda metadata.

The build environment reuses a host Python that already provides NumPy; this
hook collects NumPy's package-local shared libraries so the frozen node is
self-consistent regardless of the host layout (ported from the official
``libero_benchmark`` recipe).
"""

from PyInstaller.utils.hooks import collect_dynamic_libs


binaries = collect_dynamic_libs("numpy")
