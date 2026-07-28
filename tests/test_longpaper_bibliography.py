from __future__ import annotations

from pathlib import Path

import pytest

from papers.transactional_control_algebra_long.scripts import (
    check_bibliography as bibliography,
)

ARTICLE = """\
@article{alpha2020example,
  title = {An Example},
  author = {Alpha, Alice},
  year = {2020},
  url = {https://doi.org/10.0000/example}
}
"""

PREPRINT = """\
@misc{beta2024preprint,
  title = {A Preprint},
  author = {Beta, Bob},
  year = {2024},
  eprint = {2401.01234},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2401.01234}
}
"""


@pytest.fixture
def valid_bibliography_fixture(tmp_path: Path) -> tuple[Path, Path]:
    bibliography_path = tmp_path / "references.bib"
    source_map_path = tmp_path / "sources.md"
    bibliography_path.write_text(ARTICLE + "\n" + PREPRINT, encoding="utf-8")
    source_map_path.write_text(
        """\
# Sources

## Citation-key map for the paper scaffold

- Examples: `alpha2020example`, `beta2024preprint`.

## Guardrail
""",
        encoding="utf-8",
    )
    return bibliography_path, source_map_path


def test_checked_in_bibliography_contract() -> None:
    result = bibliography.check()

    assert result["status"] == "PASS"
    assert result["entry_count"] == 23
    assert result["unique_key_count"] == 23
    assert result["mapped_key_mentions"] == 24
    assert result["mapped_unique_key_count"] == 23


def test_valid_fixture_passes(
    valid_bibliography_fixture: tuple[Path, Path],
) -> None:
    bibliography_path, source_map_path = valid_bibliography_fixture

    result = bibliography.check(bibliography_path, source_map_path)

    assert result["entry_count"] == 2
    assert result["unique_key_count"] == 2


def test_unbalanced_entry_fails(
    valid_bibliography_fixture: tuple[Path, Path],
) -> None:
    bibliography_path, source_map_path = valid_bibliography_fixture
    bibliography_path.write_text(
        ARTICLE + "\n@misc{broken,\n  title = {Never closes}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        bibliography.BibliographyError,
        match="Unbalanced bibliography entry",
    ):
        bibliography.check(bibliography_path, source_map_path)


def test_duplicate_key_fails(
    valid_bibliography_fixture: tuple[Path, Path],
) -> None:
    bibliography_path, source_map_path = valid_bibliography_fixture
    bibliography_path.write_text(ARTICLE + "\n" + ARTICLE, encoding="utf-8")

    with pytest.raises(
        bibliography.BibliographyError,
        match="Duplicate citation key",
    ):
        bibliography.check(bibliography_path, source_map_path)


def test_missing_required_field_fails(
    valid_bibliography_fixture: tuple[Path, Path],
) -> None:
    bibliography_path, source_map_path = valid_bibliography_fixture
    bibliography_path.write_text(
        ARTICLE.replace("  title = {An Example},\n", ""),
        encoding="utf-8",
    )

    with pytest.raises(
        bibliography.BibliographyError,
        match="missing required fields: title",
    ):
        bibliography.check(bibliography_path, source_map_path)


def test_misc_without_arxiv_eprint_fails(
    valid_bibliography_fixture: tuple[Path, Path],
) -> None:
    bibliography_path, source_map_path = valid_bibliography_fixture
    bibliography_path.write_text(
        PREPRINT.replace("  eprint = {2401.01234},\n", ""),
        encoding="utf-8",
    )

    with pytest.raises(
        bibliography.BibliographyError,
        match="has no valid arXiv eprint",
    ):
        bibliography.check(bibliography_path, source_map_path)


def test_citation_key_map_must_match_exactly(
    valid_bibliography_fixture: tuple[Path, Path],
) -> None:
    bibliography_path, source_map_path = valid_bibliography_fixture
    source_map_path.write_text(
        """\
## Citation-key map for the paper scaffold

- Examples: `alpha2020example`, `unknown2025entry`.
""",
        encoding="utf-8",
    )

    with pytest.raises(
        bibliography.BibliographyError,
        match="Citation-key map mismatch",
    ):
        bibliography.check(bibliography_path, source_map_path)
