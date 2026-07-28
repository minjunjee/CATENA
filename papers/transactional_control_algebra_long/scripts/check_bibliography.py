from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

PAPER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIBLIOGRAPHY = PAPER_ROOT / "references.bib"
DEFAULT_SOURCE_MAP = PAPER_ROOT / "RELATED_WORK_PRIMARY_SOURCES.md"
CITATION_MAP_HEADING = "## Citation-key map for the paper scaffold"
REQUIRED_FIELDS = frozenset({"title", "author", "year", "url"})
SPECIAL_ENTRY_TYPES = frozenset({"comment", "preamble", "string"})
KEY_PATTERN = re.compile(r"[A-Za-z0-9_.:+/-]+")
ARXIV_PATTERN = re.compile(r"(?:\d{4}\.\d{4,5}|[A-Za-z.-]+/\d{7})")


class BibliographyError(ValueError):
    """Raised when the checked bibliography contract is violated."""


@dataclass(frozen=True)
class BibEntry:
    entry_type: str
    key: str
    fields: dict[str, str]


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _skip_space_and_comments(text: str, offset: int) -> int:
    while offset < len(text):
        if text[offset].isspace():
            offset += 1
            continue
        if text[offset] == "%":
            newline = text.find("\n", offset)
            return len(text) if newline < 0 else _skip_space_and_comments(text, newline + 1)
        break
    return offset


def _scan_entry_body(
    text: str,
    offset: int,
    opener: str,
) -> tuple[str, int]:
    closer = "}" if opener == "{" else ")"
    container_depth = 1
    brace_depth = 0
    quoted = False
    index = offset

    while index < len(text):
        character = text[index]
        if quoted:
            if character == "\\":
                index += 2
                continue
            if character == '"':
                quoted = False
            index += 1
            continue

        at_container_level = brace_depth == 0 and container_depth == 1
        if character == '"' and at_container_level:
            quoted = True
        elif character == "{":
            if opener == "{":
                container_depth += 1
            else:
                brace_depth += 1
        elif character == "}":
            if opener == "{":
                container_depth -= 1
                if container_depth == 0:
                    return text[offset:index], index + 1
            else:
                brace_depth -= 1
                if brace_depth < 0:
                    raise BibliographyError(
                        f"Unmatched closing brace at line {_line_number(text, index)}"
                    )
        elif opener == "(" and brace_depth == 0:
            if character == "(":
                container_depth += 1
            elif character == closer:
                container_depth -= 1
                if container_depth == 0:
                    return text[offset:index], index + 1
        index += 1

    if quoted:
        detail = "unterminated quoted value"
    elif brace_depth:
        detail = "unbalanced nested braces"
    else:
        detail = f"missing closing {closer}"
    raise BibliographyError(
        f"Unbalanced bibliography entry starting at line "
        f"{_line_number(text, offset - 1)}: {detail}"
    )


def _split_top_level(text: str, separator: str) -> list[str]:
    parts: list[str] = []
    start = 0
    brace_depth = 0
    parenthesis_depth = 0
    quoted = False
    index = 0

    while index < len(text):
        character = text[index]
        if quoted:
            if character == "\\":
                index += 2
                continue
            if character == '"':
                quoted = False
            index += 1
            continue

        if character == '"' and brace_depth == 0 and parenthesis_depth == 0:
            quoted = True
        elif character == "{":
            brace_depth += 1
        elif character == "}":
            brace_depth -= 1
            if brace_depth < 0:
                raise BibliographyError("Unmatched closing brace inside entry")
        elif character == "(" and brace_depth == 0:
            parenthesis_depth += 1
        elif character == ")" and brace_depth == 0:
            parenthesis_depth -= 1
            if parenthesis_depth < 0:
                raise BibliographyError("Unmatched closing parenthesis inside entry")
        elif (
            character == separator
            and brace_depth == 0
            and parenthesis_depth == 0
        ):
            parts.append(text[start:index])
            start = index + 1
        index += 1

    if quoted or brace_depth or parenthesis_depth:
        raise BibliographyError("Unbalanced field value inside entry")
    parts.append(text[start:])
    return parts


def _parse_fields(chunks: list[str], key: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for position, chunk in enumerate(chunks):
        stripped = chunk.strip()
        if not stripped and position == len(chunks) - 1:
            continue
        if not stripped:
            raise BibliographyError(f"Empty field in entry {key}")
        assignment = _split_top_level(stripped, "=")
        if len(assignment) != 2:
            raise BibliographyError(
                f"Malformed field assignment in entry {key}: {stripped!r}"
            )
        name = assignment[0].strip().lower()
        value = assignment[1].strip()
        if not name or not value:
            raise BibliographyError(f"Empty field name or value in entry {key}")
        if name in fields:
            raise BibliographyError(f"Duplicate field {name!r} in entry {key}")
        fields[name] = value
    return fields


def _parse_entry(entry_type: str, body: str) -> BibEntry | None:
    normalized_type = entry_type.lower()
    if normalized_type in SPECIAL_ENTRY_TYPES:
        return None

    chunks = _split_top_level(body, ",")
    if len(chunks) < 2:
        raise BibliographyError(f"Entry @{entry_type} has no key/field separator")
    key = chunks[0].strip()
    if not KEY_PATTERN.fullmatch(key):
        raise BibliographyError(f"Invalid or empty citation key: {key!r}")
    fields = _parse_fields(chunks[1:], key)
    return BibEntry(
        entry_type=normalized_type,
        key=key,
        fields=fields,
    )


def parse_bibliography(text: str) -> list[BibEntry]:
    entries: list[BibEntry] = []
    offset = 0
    while True:
        offset = _skip_space_and_comments(text, offset)
        if offset >= len(text):
            break
        if text[offset] != "@":
            raise BibliographyError(
                f"Unexpected content outside entry at line {_line_number(text, offset)}"
            )
        entry_line = _line_number(text, offset)
        offset += 1
        type_start = offset
        while offset < len(text) and (text[offset].isalnum() or text[offset] in "_-"):
            offset += 1
        entry_type = text[type_start:offset]
        if not entry_type:
            raise BibliographyError(f"Missing entry type at line {entry_line}")
        offset = _skip_space_and_comments(text, offset)
        if offset >= len(text) or text[offset] not in "{(":
            raise BibliographyError(
                f"Entry @{entry_type} at line {entry_line} has no opening delimiter"
            )
        opener = text[offset]
        body, offset = _scan_entry_body(text, offset + 1, opener)
        entry = _parse_entry(entry_type, body)
        if entry is not None:
            entries.append(entry)
    if not entries:
        raise BibliographyError("Bibliography has no entries")
    return entries


def _unwrapped(value: str) -> str:
    result = value.strip()
    while len(result) >= 2:
        if (result[0], result[-1]) in {("{", "}"), ('"', '"')}:
            result = result[1:-1].strip()
        else:
            break
    return result


def validate_entries(entries: list[BibEntry]) -> None:
    seen: set[str] = set()
    for entry in entries:
        if entry.key in seen:
            raise BibliographyError(f"Duplicate citation key: {entry.key}")
        seen.add(entry.key)

        missing = sorted(
            field
            for field in REQUIRED_FIELDS
            if field not in entry.fields or not _unwrapped(entry.fields[field])
        )
        if missing:
            raise BibliographyError(
                f"Entry {entry.key} is missing required fields: {', '.join(missing)}"
            )

        if entry.entry_type == "misc":
            archive_prefix = _unwrapped(entry.fields.get("archiveprefix", ""))
            eprint = _unwrapped(entry.fields.get("eprint", ""))
            if archive_prefix.lower() != "arxiv":
                raise BibliographyError(
                    f"@misc entry {entry.key} must declare archivePrefix={{arXiv}}"
                )
            if not ARXIV_PATTERN.fullmatch(eprint):
                raise BibliographyError(
                    f"@misc entry {entry.key} has no valid arXiv eprint"
                )


def parse_citation_key_map(text: str) -> list[str]:
    heading_match = re.search(
        rf"^{re.escape(CITATION_MAP_HEADING)}\s*$",
        text,
        flags=re.MULTILINE,
    )
    if heading_match is None:
        raise BibliographyError(
            f"Source map is missing heading: {CITATION_MAP_HEADING}"
        )
    next_heading = re.search(
        r"^##\s+",
        text[heading_match.end() :],
        flags=re.MULTILINE,
    )
    section_end = (
        len(text)
        if next_heading is None
        else heading_match.end() + next_heading.start()
    )
    section = text[heading_match.end() : section_end]
    keys = re.findall(r"`([^`\n]+)`", section)
    if not keys:
        raise BibliographyError("Citation-key map contains no keys")
    invalid = sorted({key for key in keys if not KEY_PATTERN.fullmatch(key)})
    if invalid:
        raise BibliographyError(
            f"Citation-key map contains invalid keys: {', '.join(invalid)}"
        )
    return keys


def validate_key_map(entries: list[BibEntry], mapped_keys: list[str]) -> None:
    bibliography_keys = {entry.key for entry in entries}
    mapping_keys = set(mapped_keys)
    unmapped = sorted(bibliography_keys - mapping_keys)
    unknown = sorted(mapping_keys - bibliography_keys)
    if unmapped or unknown:
        raise BibliographyError(
            "Citation-key map mismatch: "
            f"unmapped_bibliography_keys={unmapped}, "
            f"unknown_mapped_keys={unknown}"
        )


def check(
    bibliography_path: Path = DEFAULT_BIBLIOGRAPHY,
    source_map_path: Path = DEFAULT_SOURCE_MAP,
) -> dict[str, object]:
    entries = parse_bibliography(
        bibliography_path.read_text(encoding="utf-8")
    )
    validate_entries(entries)
    mapped_keys = parse_citation_key_map(
        source_map_path.read_text(encoding="utf-8")
    )
    validate_key_map(entries, mapped_keys)
    return {
        "status": "PASS",
        "bibliography": str(bibliography_path.resolve()),
        "source_map": str(source_map_path.resolve()),
        "entry_count": len(entries),
        "unique_key_count": len({entry.key for entry in entries}),
        "mapped_key_mentions": len(mapped_keys),
        "mapped_unique_key_count": len(set(mapped_keys)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the CATENA long-paper BibTeX and its citation-key map "
            "using only the Python standard library."
        )
    )
    parser.add_argument(
        "--bibliography",
        type=Path,
        default=DEFAULT_BIBLIOGRAPHY,
    )
    parser.add_argument(
        "--source-map",
        type=Path,
        default=DEFAULT_SOURCE_MAP,
    )
    args = parser.parse_args(argv)
    try:
        result = check(args.bibliography, args.source_map)
    except (BibliographyError, OSError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
