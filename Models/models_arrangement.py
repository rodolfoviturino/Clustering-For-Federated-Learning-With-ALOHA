"""Public JAX model API for HFL/ALOHA experiments.

The project now uses JAX as the simulation backend.  This facade preserves the
historical import path ``Models.models_arrangement`` while keeping the actual
implementation in ``Models.jax_models_arrangement``.
"""

from Models.jax_models_arrangement import (
    D2D_ENERGY_EFFICIENCY_PROFILES,
    JaxTraceResult,
    error_calculator,
    error_calculator_trace_jax,
    prepare_clusters_for_jax,
    prepare_device_metrics_for_jax,
)

__all__ = [
    "D2D_ENERGY_EFFICIENCY_PROFILES",
    "JaxTraceResult",
    "error_calculator",
    "error_calculator_trace_jax",
    "prepare_clusters_for_jax",
    "prepare_device_metrics_for_jax",
]
