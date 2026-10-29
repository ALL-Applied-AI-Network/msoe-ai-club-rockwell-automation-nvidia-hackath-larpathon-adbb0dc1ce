"""Hash-bound identity contract for a reviewed native-RGB detector adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import inspect
import json
from pathlib import Path
import re
from types import FunctionType
from typing import Callable


SUPPORTED_OUTPUT_SCHEMA = "anonymous_cube_detector_v1"
ORGANIZER_INPUT_CONVENTION = "organizer_native_rgb"
SUPPORTED_INTERNAL_CONVENTION = "opencv_bgr"
SUPPORTED_CONVERSION = "RGB_TO_BGR"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def module_file_sha256(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise ValueError("detector adapter module file is missing")
    return hashlib.sha256(source.read_bytes()).hexdigest()


@dataclass(frozen=True)
class DetectorAdapterContract:
    adapter_name: str
    adapter_semantic_version: str
    callable_module_name: str
    callable_qualified_name: str
    callable_module_file_sha256: str
    adapter_configuration_sha256: str
    input_color_convention: str
    internal_detector_color_convention: str
    color_conversion_convention: str
    output_schema_version: str
    compatible_calibration_ids: tuple[str, ...]
    compatible_workspace_versions: tuple[str, ...]
    verification_evidence_sha256: str
    canonical_contract_sha256: str
    verified: bool = False

    def payload(self) -> dict:
        value = asdict(self)
        value.pop("canonical_contract_sha256")
        return value

    def computed_contract_sha256(self) -> str:
        return canonical_sha256(self.payload())

    def serialize(self) -> dict:
        return asdict(self)


def build_detector_adapter_contract(**fields: object) -> DetectorAdapterContract:
    """Build a contract with its canonical hash; verification still occurs at use."""
    provisional = DetectorAdapterContract(canonical_contract_sha256="0" * 64, **fields)
    return replace(
        provisional,
        canonical_contract_sha256=provisional.computed_contract_sha256(),
    )


def inspect_detector_callable(detector: Callable) -> dict:
    if not isinstance(detector, FunctionType) or not inspect.isfunction(detector):
        raise ValueError("detector adapter must be a stable Python function")
    if detector.__name__ == "<lambda>" or "<lambda>" in detector.__qualname__:
        raise ValueError("lambda detector adapters are unsupported")
    if "<locals>" in detector.__qualname__ or detector.__closure__:
        raise ValueError("dynamic or closure detector adapters are unsupported")
    module = inspect.getmodule(detector)
    if module is None or not detector.__module__ or module.__name__ != detector.__module__:
        raise ValueError("detector adapter module identity is unavailable")
    module_file = inspect.getsourcefile(detector) or getattr(module, "__file__", None)
    if not module_file:
        raise ValueError("detector adapter lacks a stable inspectable module file")
    resolved = Path(module_file).resolve(strict=True)
    if resolved.suffix.lower() not in {".py", ".pyw"}:
        raise ValueError("detector adapter must originate from a stable Python source file")
    return {
        "module_name": detector.__module__,
        "qualified_name": detector.__qualname__,
        "module_file": str(resolved),
        "module_file_sha256": module_file_sha256(resolved),
    }


def verify_detector_adapter(
    detector: Callable,
    contract: DetectorAdapterContract | None,
    *,
    adapter_configuration: dict | None,
    calibration_compatibility_id: str,
    workspace_version: str,
) -> dict:
    if contract is None:
        raise ValueError("verified detector adapter contract is required")
    if not isinstance(contract, DetectorAdapterContract):
        raise ValueError("detector adapter contract has an unsupported type")
    if contract.verified is not True:
        raise ValueError("detector adapter contract is unverified")
    if not contract.adapter_name or not contract.adapter_semantic_version:
        raise ValueError("detector adapter name/version is missing")
    hashes = {
        "module file": contract.callable_module_file_sha256,
        "configuration": contract.adapter_configuration_sha256,
        "verification evidence": contract.verification_evidence_sha256,
        "canonical contract": contract.canonical_contract_sha256,
    }
    for label, value in hashes.items():
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"detector adapter {label} SHA-256 is missing or malformed")
    if contract.computed_contract_sha256() != contract.canonical_contract_sha256:
        raise ValueError("detector adapter canonical contract hash mismatch")
    if adapter_configuration is None:
        raise ValueError("detector adapter configuration is required")
    configuration_hash = canonical_sha256(adapter_configuration)
    if configuration_hash != contract.adapter_configuration_sha256:
        raise ValueError("detector adapter configuration hash mismatch")
    if contract.input_color_convention != ORGANIZER_INPUT_CONVENTION:
        raise ValueError("detector adapter input must be organizer-native RGB")
    if contract.internal_detector_color_convention != SUPPORTED_INTERNAL_CONVENTION:
        raise ValueError("detector adapter internal color convention is unsupported")
    if contract.color_conversion_convention != SUPPORTED_CONVERSION:
        raise ValueError("detector adapter RGB-to-BGR conversion declaration is missing or unsupported")
    if contract.output_schema_version != SUPPORTED_OUTPUT_SCHEMA:
        raise ValueError("detector adapter output schema version is unsupported")
    if not contract.compatible_calibration_ids or calibration_compatibility_id not in contract.compatible_calibration_ids:
        raise ValueError("detector adapter calibration compatibility mismatch")
    if not contract.compatible_workspace_versions or workspace_version not in contract.compatible_workspace_versions:
        raise ValueError("detector adapter workspace compatibility mismatch")
    runtime = inspect_detector_callable(detector)
    if runtime["module_name"] != contract.callable_module_name:
        raise ValueError("detector adapter callable module mismatch")
    if runtime["qualified_name"] != contract.callable_qualified_name:
        raise ValueError("detector adapter callable qualified-name mismatch")
    if runtime["module_file_sha256"] != contract.callable_module_file_sha256:
        raise ValueError("detector adapter module-file hash mismatch")
    return {
        **runtime,
        "adapter_name": contract.adapter_name,
        "adapter_semantic_version": contract.adapter_semantic_version,
        "adapter_configuration_sha256": configuration_hash,
        "color_conversion_convention": contract.color_conversion_convention,
        "output_schema_version": contract.output_schema_version,
        "verification_evidence_sha256": contract.verification_evidence_sha256,
        "contract_sha256": contract.canonical_contract_sha256,
    }
