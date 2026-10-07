"""Where the artifacts of a run live on disk.

`ResultsLayout` is the only code that knows the directory scheme

    <root>/<kind>/<paradigm>/<config_id>/<file>

with the kinds `checkpoints`, `histories`, `metrics`, and `plots`. Every path
is derived from the run specification (or its IDs), so nothing recovers
configuration from a file name. The seeds of one configuration share a
directory and differ only in the file name (`seed_<n>`).
"""

import glob
import json
import os

from dementia_boost.core.identity import Paradigm, RunSpec, config_id, label

DEFAULT_RESULTS_ROOT = "./data/results"
RESULT_COHORTS = ("val", "test")
CONFIG_NAME = "config.json"


class ConfigCollisionError(ValueError):
    """Raised when two different specs map to the same configuration ID."""


class ResultsLayout:
    """Builds and checks the artifact paths of runs.

    Attributes:
        root: Base directory holding the artifact trees.
    """

    def __init__(self, root: str = DEFAULT_RESULTS_ROOT) -> None:
        """Initializes the layout.

        Args:
            root: Base directory of the artifact trees. Defaults to
                "./data/results".
        """
        self.root = root

    def checkpoint_path(self, spec: RunSpec) -> str:
        """Returns the checkpoint file of a run.

        Args:
            spec: The run specification.

        Returns:
            `<root>/checkpoints/<paradigm>/<config_id>/seed_<n>.pt`.
        """
        return os.path.join(
            self._config_dir("checkpoints", spec.paradigm, config_id(spec)),
            f"seed_{spec.seed}.pt",
        )

    def history_path(self, spec: RunSpec) -> str:
        """Returns the training history file of a run.

        Args:
            spec: The run specification.

        Returns:
            `<root>/histories/<paradigm>/<config_id>/seed_<n>.json`.
        """
        return os.path.join(
            self._config_dir("histories", spec.paradigm, config_id(spec)),
            f"seed_{spec.seed}.json",
        )

    def config_path(self, spec: RunSpec) -> str:
        """Returns the `config.json` of a run's configuration.

        Args:
            spec: The run specification.

        Returns:
            `<root>/histories/<paradigm>/<config_id>/config.json`.
        """
        return os.path.join(
            self._config_dir("histories", spec.paradigm, config_id(spec)), CONFIG_NAME
        )

    def history_dir(self, paradigm: Paradigm | str, configuration: str) -> str:
        """Returns the directory holding the histories of one configuration.

        Args:
            paradigm: The paradigm.
            configuration: The `config_id`.

        Returns:
            `<root>/histories/<paradigm>/<config_id>`.
        """
        return self._config_dir("histories", paradigm, configuration)

    def history_files(
        self, paradigm: Paradigm | str | None = None, configuration: str | None = None
    ) -> list[str]:
        """Lists the training history files on disk.

        Only the directory scheme is used to find the files; what a history is
        comes from its content, not its name.

        Args:
            paradigm: Restrict to one paradigm. Defaults to all.
            configuration: Restrict to one `config_id`. Defaults to all.

        Returns:
            The sorted paths of the `seed_*.json` files, without `config.json`
            and without temporary files.
        """
        pattern = os.path.join(
            glob.escape(self.root),
            "histories",
            Paradigm(paradigm).value if paradigm is not None else "*",
            configuration if configuration is not None else "*",
            "seed_*.json",
        )
        return sorted(glob.glob(pattern))

    def metrics_path(
        self, paradigm: Paradigm | str, configuration: str, cohort: str
    ) -> str:
        """Returns the results file of one configuration on one cohort.

        Args:
            paradigm: The paradigm.
            configuration: The `config_id`.
            cohort: "val" or "test".

        Returns:
            `<root>/metrics/<paradigm>/<config_id>/<cohort>_results.json`.

        Raises:
            ValueError: If `cohort` is not "val" or "test".
        """
        if cohort not in RESULT_COHORTS:
            raise ValueError(
                f"Unknown cohort {cohort!r} for results; expected one of "
                f"{RESULT_COHORTS}."
            )
        return os.path.join(
            self._config_dir("metrics", paradigm, configuration),
            f"{cohort}_results.json",
        )

    def configs_with_metrics(self, paradigm: Paradigm | str, cohort: str) -> list[str]:
        """Lists the configurations that have a results file for a cohort.

        Args:
            paradigm: The paradigm.
            cohort: "val" or "test".

        Returns:
            The sorted `config_id`s whose results file exists.
        """
        pattern = os.path.join(
            glob.escape(self.root),
            "metrics",
            Paradigm(paradigm).value,
            "*",
            f"{cohort}_results.json",
        )
        return sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(pattern))

    def plots_dir(self, paradigm: Paradigm | str, configuration: str) -> str:
        """Returns the plot directory of one configuration.

        Args:
            paradigm: The paradigm.
            configuration: The `config_id`.

        Returns:
            `<root>/plots/<paradigm>/<config_id>`.
        """
        return self._config_dir("plots", paradigm, configuration)

    def is_done(self, spec: RunSpec) -> bool:
        """Tells whether a run's checkpoint already exists.

        Args:
            spec: The run specification.

        Returns:
            True if the checkpoint of exactly this spec is on disk. A spec that
            differs in any field has another directory, so it is never mistaken
            for a finished run.
        """
        return os.path.exists(self.checkpoint_path(spec))

    def write_config(self, spec: RunSpec) -> None:
        """Writes the `config.json` of a run's configuration, atomically.

        The file holds the configuration ID, its label, and the spec without
        the seed. Writing it again for another seed of the same configuration
        leaves it unchanged.

        Args:
            spec: The run specification.

        Raises:
            ConfigCollisionError: If a `config.json` already exists under this
                ID with a different spec.
        """
        path = self.config_path(spec)
        payload = self._config_payload(spec)

        if os.path.exists(path):
            with open(path) as handle:
                existing = json.load(handle)
            if existing["spec"] != payload["spec"]:
                raise ConfigCollisionError(
                    f"Configuration ID {payload['config_id']!r} already belongs to "
                    f"a different spec: {existing['spec']} versus {payload['spec']}."
                )
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)
        temp_path = f"{path}.tmp"
        with open(temp_path, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp_path, path)

    def _config_dir(
        self, kind: str, paradigm: Paradigm | str, configuration: str
    ) -> str:
        """Joins the root, artifact kind, paradigm, and configuration ID."""
        return os.path.join(self.root, kind, Paradigm(paradigm).value, configuration)

    @staticmethod
    def _config_payload(spec: RunSpec) -> dict:
        """Builds the content of `config.json` for a spec."""
        data = spec.to_dict()
        del data["seed"]
        return {
            "config_id": config_id(spec),
            "label": label(spec, include_seed=False),
            "spec": data,
        }
