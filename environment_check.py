"""Utility helpers to verify backend analytics dependencies."""

from __future__ import annotations

import importlib.util
import logging
from functools import lru_cache
from typing import Dict, List

LOGGER = logging.getLogger(__name__)

# Core dependencies that power the modelling pipeline.
REQUIRED_MODULES: Dict[str, str] = {
    "numpy": "NumPy numerical computing library",
    "pandas": "Pandas data toolkit",
    "plotly": "Plotly charting library",
}

# Helpful but non-blocking extras – the app can limp along without them.
OPTIONAL_MODULES: Dict[str, str] = {
    "matplotlib": "Matplotlib visualisation toolkit",
    "numpy_financial": "NumPy Financial helpers",
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
        available = _module_available(module_name)
        module_status[module_name] = available
        if not available:
            missing_required.append(friendly)

    for module_name, friendly in OPTIONAL_MODULES.items():
        available = _module_available(module_name)
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


def _module_available(module_name: str) -> bool:
    """Return ``True`` if ``module_name`` can be imported without errors."""

    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        # Nested module lookups (e.g. ``scipy.optimize``) raise ``ModuleNotFoundError``
        # when the parent distribution is missing. Treat this the same as ``find_spec``
        # returning ``None`` so the caller can surface a friendly warning instead of
        # crashing the Streamlit script.
        return False
    except ValueError:
        # ``find_spec`` can raise ``ValueError`` for namespace packages that do not
        # expose any loaders. This again simply indicates the module is unavailable.
        return False
    except Exception:  # pragma: no cover - extremely defensive guardrail
        LOGGER.debug("Unexpected error while probing module %s", module_name, exc_info=True)
        return False
