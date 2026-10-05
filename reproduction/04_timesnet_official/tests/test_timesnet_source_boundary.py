"""Contracts for the immutable author TimesNet source boundary."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPRODUCTION_ROOT = PROJECT_ROOT / "reproduction" / "04_timesnet_official"
PROVENANCE = REPRODUCTION_ROOT / "provenance" / "upstream.json"
FETCH_SCRIPT = REPRODUCTION_ROOT / "scripts" / "fetch_upstream.sh"
VERIFY_SCRIPT = REPRODUCTION_ROOT / "scripts" / "verify_upstream.sh"
OFFICIAL_URL = "https://github.com/thuml/Time-Series-Library.git"
OFFICIAL_COMMIT = "2665a3143dae12d1cbcc31ddd396bbff48773bce"


def test_tests_directory_does_not_claim_the_shared_tests_package() -> None:
    """This suite relies on unique module names, not a globally shared package name."""

    assert not (Path(__file__).parent / "__init__.py").exists()


def test_module_filename_is_unique_across_reproductions() -> None:
    """A TimesNet test must not reuse another reproduction's module name."""

    assert Path(__file__).name.startswith("test_timesnet_")


def test_fetcher_and_provenance_pin_paper_era_tslib_commit() -> None:
    """The reproduction must not silently follow a moving TSLib branch head."""

    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))

    assert provenance["repository"] == OFFICIAL_URL
    assert provenance["commit"] == OFFICIAL_COMMIT
    assert provenance["official_ettm1_script"] == (
        "scripts/long_term_forecast/ETT_script/TimesNet_ETTm1.sh"
    )
    source = FETCH_SCRIPT.read_text(encoding="utf-8")
    assert OFFICIAL_URL in source
    assert OFFICIAL_COMMIT in source


def test_fetcher_initializes_an_empty_repository_before_fetching_only_the_pin() -> None:
    """Fetching one historic commit must not require cloning all TSLib history first."""

    source = FETCH_SCRIPT.read_text(encoding="utf-8")

    assert "git init" in source
    assert "fetch --depth 1 origin" in source
    assert "git clone --no-checkout" not in source


def test_verifier_explicitly_rejects_dirty_or_wrong_author_tree() -> None:
    """Compatibility code must stay outside the author clone."""

    source = VERIFY_SCRIPT.read_text(encoding="utf-8")

    assert "remote get-url origin" in source
    assert "rev-parse HEAD" in source
    assert "status --porcelain" in source
    assert "diff --quiet" in source
    assert "diff --cached --quiet" in source


def test_project_ignore_rules_keep_author_source_and_runs_out_of_git() -> None:
    """The upstream clone, environments, and cloud run artifacts are refetchable inputs."""

    ignore_rules = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "reproduction/04_timesnet_official/upstream/" in ignore_rules
    assert "reproduction/04_timesnet_official/.venv/" in ignore_rules
    assert "reproduction/04_timesnet_official/runs/" in ignore_rules
