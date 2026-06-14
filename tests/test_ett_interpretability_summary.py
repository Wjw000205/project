import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_ett_interpretability import ROOT, _report_intro


def test_report_intro_names_actual_best_results_file() -> None:
    best_results = ROOT / "outputs" / "ett_interpretability_report" / "best_results_ett_h96_h192.csv"

    intro = _report_intro(best_results)

    assert "`outputs/ett_interpretability_report/best_results_ett_h96_h192.csv`" in intro
    assert "ett_horizon_specific_moe_tune/best_results.csv" not in intro
