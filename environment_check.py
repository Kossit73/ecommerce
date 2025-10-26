"""Utility helpers to verify backend analytics dependencies."""

from __future__ import annotations

import importlib.util
import logging
from functools import lru_cache
from typing import Dict, List

LOGGER = logging.getLogger(__name__)

# These packages power the heavy modelling routines and must be available.
REQUIRED_MODULES: Dict[str, str] = {
    "scipy": "SciPy core package",
    "scipy.stats": "SciPy statistical distributions",
    "scipy.optimize": "SciPy optimisation routines",
    "statsmodels": "statsmodels time-series toolkit",
    "sklearn": "scikit-learn machine learning toolkit",
    "networkx": "NetworkX graph analytics",
}

# Helpful but non-blocking extras – the app can limp along without them.
OPTIONAL_MODULES: Dict[str, str] = {
    "numpy_financial": "NumPy Financial helpers (falls back to NumPy if missing)",
    "xlsxwriter": "XlsxWriter Excel writer",
    "openpyxl": "openpyxl Excel writer",
}


@lru_cache(maxsize=1)
def diagnose_environment() -> Dict[str, object]:
    """Return dependency availability details.

    The result is cached so repeated UI renders do not re-scan modules.
    """

    missing_required: List[str] = []
    missing_optional: List[str] = []
    module_status: Dict[str, bool] = {}

    for module_name, friendly in REQUIRED_MODULES.items():
        available = importlib.util.find_spec(module_name) is not None
        module_status[module_name] = available
        if not available:
            missing_required.append(friendly)

    for module_name, friendly in OPTIONAL_MODULES.items():
        available = importlib.util.find_spec(module_name) is not None
        module_status[module_name] = available
        if not available:
            missing_optional.append(friendly)

    return {
        "ready": not missing_required,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "module_status": module_status,
    }


def require_environment(*, strict: bool = True) -> None:
    """Ensure required dependencies are installed.

    Args:
        strict: When ``True`` (default) a missing dependency raises ``RuntimeError``;
            otherwise the failure is logged as a warning so the caller can decide how
            to proceed.
    """

    status = diagnose_environment()
    if status["ready"]:
        return

    missing = status["missing_required"]
    message = (
        "Missing required analytics dependencies: "
        + ", ".join(missing)
        + ". Install them with `pip install -r backend-requirements.txt`."
    )

    if strict:
        raise RuntimeError(message)
    LOGGER.warning(message)
