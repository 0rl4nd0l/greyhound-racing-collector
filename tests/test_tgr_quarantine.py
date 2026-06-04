import importlib.util
from pathlib import Path

from temporal_feature_builder import TemporalFeatureBuilder
from temporal_feature_builder_optimized import OptimizedTemporalFeatureBuilder


class DummyTemporalBuilder:
    def __init__(self):
        self.seen = None

    def set_tgr_enabled(self, enabled):
        self.seen = enabled


def _real_ml_system_class():
    module_path = Path(__file__).resolve().parents[1] / "ml_system_v4.py"
    spec = importlib.util.spec_from_file_location("real_ml_system_v4_for_tgr_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.MLSystemV4


def test_temporal_builder_refuses_tgr_enable_without_research_override(monkeypatch):
    monkeypatch.delenv("GREYHOUND_ALLOW_TGR", raising=False)
    builder = TemporalFeatureBuilder(":memory:")

    builder.set_tgr_enabled(True)

    assert builder._tgr_runtime_enabled is False


def test_optimized_temporal_builder_refuses_tgr_enable_without_research_override(monkeypatch):
    monkeypatch.delenv("GREYHOUND_ALLOW_TGR", raising=False)
    builder = OptimizedTemporalFeatureBuilder(":memory:")

    builder.set_tgr_enabled(True)

    assert builder._tgr_runtime_enabled is False


def test_ml_system_tgr_toggle_is_quarantined_without_research_override(monkeypatch):
    monkeypatch.delenv("GREYHOUND_ALLOW_TGR", raising=False)
    cls = _real_ml_system_class()
    ml = cls.__new__(cls)
    ml.temporal_builder = DummyTemporalBuilder()

    cls.set_tgr_enabled(ml, True)

    assert ml._tgr_enabled is False
    assert ml.temporal_builder.seen is False


def test_ml_system_tgr_research_override_can_enable(monkeypatch):
    monkeypatch.setenv("GREYHOUND_ALLOW_TGR", "1")
    cls = _real_ml_system_class()
    ml = cls.__new__(cls)
    ml.temporal_builder = DummyTemporalBuilder()

    cls.set_tgr_enabled(ml, True)

    assert ml._tgr_enabled is True
    assert ml.temporal_builder.seen is True


def test_ml_system_blocks_legacy_tgr_columns_without_research_override(monkeypatch):
    monkeypatch.delenv("GREYHOUND_ALLOW_TGR", raising=False)
    cls = _real_ml_system_class()
    ml = cls.__new__(cls)
    ml._tgr_enabled = False

    result = cls._maybe_block_tgr_disabled_prediction_inputs(
        ml,
        race_id="RACE_TGR_GUARD",
        expected_cols=["historical_win_rate", "tgr_total_races"],
        missing_cols=["tgr_total_races"],
    )

    assert result is not None
    assert result["success"] is False
    assert result["error"] == "TGR-disabled guardrail blocked legacy TGR compatibility path"
    assert result["fallback_reason"] == "Loaded artifact requires TGR features while TGR is disabled"
    assert result["tgr_guardrail"]["status"] == "blocked"
    assert result["tgr_guardrail"]["tgr_enabled"] is False
    assert result["tgr_guardrail"]["research_override_required"] is True
    assert result["tgr_guardrail"]["research_override_present"] is False
    assert result["tgr_guardrail"]["expected_tgr_columns"] == ["tgr_total_races"]
    assert result["tgr_guardrail"]["missing_tgr_columns"] == ["tgr_total_races"]


def test_ml_system_allows_legacy_tgr_columns_with_research_override(monkeypatch):
    monkeypatch.setenv("GREYHOUND_ALLOW_TGR", "1")
    cls = _real_ml_system_class()
    ml = cls.__new__(cls)
    ml._tgr_enabled = True

    result = cls._maybe_block_tgr_disabled_prediction_inputs(
        ml,
        race_id="RACE_TGR_ALLOWED",
        expected_cols=["historical_win_rate", "tgr_total_races"],
        missing_cols=["tgr_total_races"],
    )

    assert result is None
