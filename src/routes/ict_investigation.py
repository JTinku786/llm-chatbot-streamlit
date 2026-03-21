"""ICT investigation routing + SMC pipeline integration utilities."""

from __future__ import annotations

import importlib.util
import inspect
import os
import re
from datetime import datetime
from pathlib import Path


INTENT_PATTERN = re.compile(r"\bict\s+investigation\b", flags=re.IGNORECASE)
ENTITY_PATTERN = re.compile(r"\b(?:for|on)\s+([A-Za-z.\-]{1,15})\b", flags=re.IGNORECASE)


def extract_ict_entity(prompt: str) -> str:
    """Extract ticker entity from prompts like 'ICT investigation for INFY'."""
    if not prompt:
        return ""
    if not INTENT_PATTERN.search(prompt):
        return ""

    match = ENTITY_PATTERN.search(prompt)
    if not match:
        return ""
    return match.group(1).upper()


def _candidate_pipeline_paths() -> list[Path]:
    """Return candidate paths where smc_complete_pipeline.py may exist."""
    env_path = os.getenv("SMC_PIPELINE_PATH", "").strip()
    candidates = []
    if env_path:
        candidates.append(Path(env_path))

    repo_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            repo_root / "smc_complete_pipeline.py",
            repo_root / "src" / "routes" / "smc_complete_pipeline.py",
        ]
    )
    return candidates


def _load_pipeline_module():
    """Load the SMC pipeline module from known locations."""
    for path in _candidate_pipeline_paths():
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location("smc_complete_pipeline", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, str(path)
    return None, ""


def _call_with_supported_signature(func, ticker: str):
    """Call target function while supporting zero/one-arg signatures."""
    signature = inspect.signature(func)
    params = [
        p for p in signature.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if not params:
        return func()
    return func(ticker)


def run_ict_investigation(ticker: str) -> dict:
    """Run ICT investigation by dispatching into smc_complete_pipeline.py."""
    normalized_ticker = (ticker or "").upper().strip()
    result = {
        "ok": False,
        "route": "ict_investigation_pipeline",
        "ticker": normalized_ticker,
        "generated_at_utc": datetime.utcnow().isoformat(),
        "pipeline_path": "",
    }

    if not normalized_ticker:
        result["error"] = "No ticker symbol provided."
        return result

    module, pipeline_path = _load_pipeline_module()
    result["pipeline_path"] = pipeline_path
    if module is None:
        result["error"] = (
            "smc_complete_pipeline.py not found. "
            "Set SMC_PIPELINE_PATH or place file in project root / src/routes."
        )
        return result

    try:
        for fn_name in ("run_pipeline", "run", "process_ticker", "analyze_ticker", "main"):
            fn = getattr(module, fn_name, None)
            if callable(fn):
                payload = _call_with_supported_signature(fn, normalized_ticker)
                result["ok"] = True
                result["handler"] = fn_name
                result["data"] = payload
                return result

        result["error"] = (
            "No supported callable found in smc_complete_pipeline.py. "
            "Expected one of: run_pipeline, run, process_ticker, analyze_ticker, main."
        )
    except Exception as exc:
        result["error"] = str(exc)

    return result
