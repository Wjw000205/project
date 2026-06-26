from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("src", "configs", "scripts", "tests")
FORBIDDEN_TOKENS = (
    "knn_hybrid",
    "knn",
    "KNN",
    "KNNShapeConfig",
    "ShapeKNNHybrid",
    "utils.knn_shape",
    "predict_bank_outputs",
    "save_shape_knn_bank",
    "shape_knn_bank",
    "NearestNeighbors",
    "calibration",
    "Calibration",
    "calibrator",
    "Calibrator",
    "gate_calibrator",
)


def _iter_active_text_files():
    for rel_dir in SCAN_DIRS:
        root = ROOT / rel_dir
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path == Path(__file__).resolve():
                continue
            if path.suffix.lower() not in {".py", ".yaml", ".yml"}:
                continue
            yield path


def test_removed_legacy_modules_do_not_reappear() -> None:
    offenders: list[str] = []
    for path in _iter_active_text_files():
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_TOKENS:
            if token in text:
                offenders.append(f"{path.relative_to(ROOT)} contains {token}")
    assert offenders == []
