"""Unit tests for the comparative report across paradigms.

Regression coverage
-------------------
- The report ranked "best" runs by test metrics, so the headline numbers were
  chosen on the same cohort they were reported on.
"""

from pathlib import Path

import pytest

from dementia_boost.core.layout import ResultsLayout
from dementia_boost.telemetry.report import (
    build_comparative_report,
    calculate_percentage_delta,
    resolve_configuration,
)

_METRICS = ("accuracy", "precision", "recall", "f1_score", "auc")


def _payload(
    cohort: str,
    runs: dict[str, dict[str, float]],
    split_id: str = "split_a",
    config: str = "config_a",
) -> dict:
    """Builds a results payload as `save_to_json` writes it.

    Args:
        cohort: Cohort marker of the payload.
        runs: Mapping of run ID to its metrics; missing metrics default to 0.5
            and `log_loss` to 0.7.
        split_id: Split the configuration was trained on.
        config: Configuration ID the results belong to.

    Returns:
        A payload with `cohort`, `configuration`, `individual_runs`, and
        mean/std statistics.
    """
    individual = [
        {
            "run_id": run_id,
            "log_loss": 0.7,
            **{name: 0.5 for name in _METRICS},
            **metrics,
        }
        for run_id, metrics in runs.items()
    ]
    stats = {}
    for name in _METRICS:
        values = [run[name] for run in individual]
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        stats[name] = {"mean": mean, "std": std}
    return {
        "cohort": cohort,
        "configuration": {
            "config_id": config,
            "label": "test configuration",
            "spec": {"split_id": split_id},
        },
        "aggregated_statistics": stats,
        "individual_runs": individual,
    }


def _paradigm(
    val_runs: dict,
    test_runs: dict | None = None,
    split_id: str = "split_a",
    config: str = "config_a",
) -> tuple[dict, dict]:
    """Returns the (val, test) payload pair of one paradigm."""
    return (
        _payload("val", val_runs, split_id, config),
        _payload("test", test_runs or val_runs, split_id, config),
    )


def _report(**paradigms: tuple[dict, dict]) -> dict:
    """Builds a report from `name=(val, test)` pairs."""
    return build_comparative_report(
        test_results={name: test for name, (_, test) in paradigms.items()},
        val_results={name: val for name, (val, _) in paradigms.items()},
    )


class TestSelectedRuns:
    """Validates that the shown run is chosen on validation, not on test."""

    def test_selected_run_follows_validation_not_test(self) -> None:
        """The run with the best validation AUC is shown, whatever its test score."""
        val = {"a": {"auc": 0.60}, "b": {"auc": 0.90}}
        test = {
            "a": {"auc": 0.95, "accuracy": 0.99},
            "b": {"auc": 0.55, "accuracy": 0.4},
        }

        report = _report(baseline=_paradigm(val, test))

        assert report["metadata"]["selected_runs"] == {"baseline": "b"}
        shown = report["comparisons"]["auc"]["paradigms"]["baseline"]
        assert shown["selected_on_validation"] == pytest.approx(0.55)

    def test_swapping_test_metrics_does_not_change_the_selection(self) -> None:
        """Only validation metrics decide which run is selected."""
        val = {"a": {"auc": 0.60}, "b": {"auc": 0.90}}
        first = _report(baseline=_paradigm(val, {"a": {"auc": 0.9}, "b": {"auc": 0.1}}))
        second = _report(
            baseline=_paradigm(val, {"a": {"auc": 0.1}, "b": {"auc": 0.9}})
        )

        assert first["metadata"]["selected_runs"] == second["metadata"]["selected_runs"]


class TestHeadlineNumbers:
    """Validates that the report leads with mean and spread across seeds."""

    def test_mean_and_std_come_from_the_test_statistics(self) -> None:
        """Mean and std of every metric are those of the whole test sweep."""
        runs = {"a": {"auc": 0.6}, "b": {"auc": 0.8}}

        report = _report(baseline=_paradigm({"a": {}, "b": {}}, runs))

        shown = report["comparisons"]["auc"]["paradigms"]["baseline"]
        assert shown["mean"] == pytest.approx(0.7)
        assert shown["std"] == pytest.approx(0.1)

    def test_no_entry_is_labelled_best(self) -> None:
        """No key in the report claims a best run chosen on test."""
        report = _report(baseline=_paradigm({"a": {}}))

        assert "best" not in str(report).lower()

    def test_deltas_use_the_means(self) -> None:
        """Percentage deltas compare the mean of each paradigm."""
        baseline = _paradigm({"a": {"auc": 0.5}}, {"a": {"auc": 0.5}})
        ctl = _paradigm({"a": {"auc": 0.6}}, {"a": {"auc": 0.6}})

        report = _report(baseline=baseline, ctl=ctl)

        deltas = report["comparisons"]["auc"]["delta_pct_of_means"]
        assert deltas["ctl_vs_baseline"] == pytest.approx(20.0)

    def test_missing_optional_paradigms_are_omitted(self) -> None:
        """Only paradigms that were evaluated appear, and only their deltas."""
        report = _report(baseline=_paradigm({"a": {}}), ctl=_paradigm({"a": {}}))

        auc = report["comparisons"]["auc"]
        assert set(auc["paradigms"]) == {"baseline", "ctl"}
        assert set(auc["delta_pct_of_means"]) == {"ctl_vs_baseline"}


class TestInputValidation:
    """Validates the refusal of inconsistent inputs."""

    def test_test_payload_marked_as_validation_raises(self) -> None:
        """A file computed on the wrong cohort is refused."""
        val, _ = _paradigm({"a": {}})
        with pytest.raises(ValueError, match="cohort"):
            build_comparative_report(
                test_results={"baseline": val}, val_results={"baseline": val}
            )

    def test_different_paradigms_in_val_and_test_raise(self) -> None:
        """Validation and test results must cover the same paradigms."""
        val, test = _paradigm({"a": {}})
        with pytest.raises(ValueError, match="paradigms"):
            build_comparative_report(
                test_results={"baseline": test}, val_results={"ctl": val}
            )

    def test_paradigms_trained_on_different_splits_are_refused(self) -> None:
        """Results from different splits are not comparable, so they cannot share
        a report. The error names both split IDs."""
        with pytest.raises(ValueError, match="split_a.*split_b|split_b.*split_a"):
            _report(
                baseline=_paradigm({"a": {}}, split_id="split_a"),
                ctl=_paradigm({"a": {}}, split_id="split_b"),
            )

    def test_validation_and_test_of_different_configurations_are_refused(self) -> None:
        """A paradigm's validation and test files must describe one configuration,
        or the run selected on validation would be looked up in another."""
        val = _payload("val", {"a": {}}, config="config_a")
        test = _payload("test", {"a": {}}, config="config_b")
        with pytest.raises(ValueError, match="configuration"):
            build_comparative_report(
                test_results={"baseline": test}, val_results={"baseline": val}
            )

    def test_selected_run_missing_from_test_results_raises(self) -> None:
        """A run chosen on validation must exist in the test results."""
        val = _payload("val", {"a": {"auc": 0.9}})
        test = _payload("test", {"other": {}})
        with pytest.raises(ValueError, match="a"):
            build_comparative_report(
                test_results={"baseline": test}, val_results={"baseline": val}
            )


class TestResolveConfiguration:
    """Validates choosing the configuration a paradigm contributes to a report."""

    def _results(self, layout: ResultsLayout, paradigm: str, config: str, *cohorts):
        """Creates (empty) results files for a configuration and cohorts."""
        for cohort in cohorts:
            path = Path(layout.metrics_path(paradigm, config, cohort))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")

    def test_the_only_evaluated_configuration_is_used(self, tmp_path: Path) -> None:
        """With one configuration that has both cohorts, no choice is needed."""
        layout = ResultsLayout(str(tmp_path))
        self._results(layout, "ctl", "aaa", "val", "test")

        assert resolve_configuration(layout, "ctl") == "aaa"

    def test_configurations_without_both_cohorts_do_not_count(
        self, tmp_path: Path
    ) -> None:
        """A configuration evaluated on one cohort only cannot be reported."""
        layout = ResultsLayout(str(tmp_path))
        self._results(layout, "ctl", "aaa", "test")

        assert resolve_configuration(layout, "ctl") is None

    def test_no_results_gives_none(self, tmp_path: Path) -> None:
        """A paradigm without results contributes nothing."""
        assert resolve_configuration(ResultsLayout(str(tmp_path)), "qtl") is None

    def test_several_configurations_need_an_explicit_choice(
        self, tmp_path: Path
    ) -> None:
        """Choosing between configurations silently would be a hidden decision."""
        layout = ResultsLayout(str(tmp_path))
        self._results(layout, "qtl", "aaa", "val", "test")
        self._results(layout, "qtl", "bbb", "val", "test")

        with pytest.raises(ValueError, match="configuration"):
            resolve_configuration(layout, "qtl")
        assert resolve_configuration(layout, "qtl", "bbb") == "bbb"

    def test_an_unavailable_choice_raises(self, tmp_path: Path) -> None:
        """A requested configuration must have both cohorts evaluated."""
        layout = ResultsLayout(str(tmp_path))
        self._results(layout, "qtl", "aaa", "val", "test")

        with pytest.raises(ValueError, match="zzz"):
            resolve_configuration(layout, "qtl", "zzz")


def test_percentage_delta_is_zero_for_a_zero_base() -> None:
    """A zero baseline gives 0.0 instead of dividing by zero."""
    assert calculate_percentage_delta(0.0, 0.5) == 0.0
    assert calculate_percentage_delta(0.5, 0.6) == pytest.approx(20.0)
