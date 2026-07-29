from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import cast

import torch

from catena.core.schema import Operation

ALLOWED_SAFE_FIELDS = (
    "entity_description",
    "domain",
    "current_relation",
    "incoming_evidence",
    "prior_version",
    "evidence_version",
    "observation_day",
    "evidence_timestamp_day",
    "prior_valid_from_day",
    "prior_valid_to_day",
    "evidence_valid_from_day",
    "evidence_valid_to_day",
    "scope",
    "source",
    "provenance",
    "incoming_value_token",
    "template_surface",
)

BANNED_SURFACE_CUES = (
    "add",
    "delete",
    "revoke",
    "invalidate",
    "replace",
    "supersede",
)

MINIMUM_NUMERIC_NAMESPACE = 5_000_000_000_000
_CURRENT_RELATION_PREFIX = "relation_at::"
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True, slots=True)
class SafeSemanticRecord:
    """The complete and exclusive model-visible structured transaction record."""

    entity_description: str
    domain: str
    current_relation: str
    incoming_evidence: str
    prior_version: int
    evidence_version: int
    observation_day: int
    evidence_timestamp_day: int
    prior_valid_from_day: int
    prior_valid_to_day: int
    evidence_valid_from_day: int
    evidence_valid_to_day: int
    scope: str
    source: str
    provenance: str
    incoming_value_token: str
    template_surface: str

    def __post_init__(self) -> None:
        for field_name in (
            "entity_description",
            "domain",
            "current_relation",
            "incoming_evidence",
            "scope",
            "source",
            "provenance",
            "incoming_value_token",
            "template_surface",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a nonempty string.")
        for field_name in (
            "prior_version",
            "evidence_version",
            "observation_day",
            "evidence_timestamp_day",
            "prior_valid_from_day",
            "prior_valid_to_day",
            "evidence_valid_from_day",
            "evidence_valid_to_day",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer.")
            if value < 0:
                raise ValueError(f"{field_name} must be nonnegative.")
        if self.prior_valid_from_day > self.prior_valid_to_day:
            raise ValueError("The prior validity interval is reversed.")
        if self.evidence_valid_from_day > self.evidence_valid_to_day:
            raise ValueError("The evidence validity interval is reversed.")
        _current_scope(self)
        banned = find_banned_surface_cues(self)
        if banned:
            raise ValueError(f"Safe semantic record contains banned surface cues: {banned}.")


if tuple(field.name for field in fields(SafeSemanticRecord)) != ALLOWED_SAFE_FIELDS:
    raise AssertionError("SafeSemanticRecord fields differ from the frozen allow-list.")


@dataclass(frozen=True, slots=True)
class RawDemand:
    erase: bool
    write: bool

    @property
    def operation(self) -> Operation:
        return {
            (False, False): Operation.PRESERVE,
            (False, True): Operation.ADD,
            (True, False): Operation.INVALIDATE,
            (True, True): Operation.SUPERSEDE,
        }[(self.erase, self.write)]


@dataclass(frozen=True, slots=True)
class SemanticMemorySpec:
    num_associations: int = 16
    key_dim: int = 32
    value_dim: int = 32
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.num_associations <= 1:
            raise ValueError("num_associations must be greater than one.")
        if self.key_dim < self.num_associations:
            raise ValueError("key_dim must support the requested orthonormal keys.")
        if self.value_dim <= 0:
            raise ValueError("value_dim must be positive.")
        if self.dtype not in {torch.float32, torch.float64}:
            raise ValueError("Only float32 and float64 semantic memory are supported.")


@dataclass(frozen=True, slots=True)
class PrivateSemanticEpisode:
    """Evaluation-only state kept outside the model-visible semantic record."""

    example_id: str
    transaction_id: str
    episode_id: str
    namespace_name: str
    numeric_seed: int
    checkpoint_seed: int
    operation: Operation
    old_value_token: str
    keys: torch.Tensor
    values: torch.Tensor
    state: torch.Tensor
    target_state: torch.Tensor
    affected_index: int
    new_value: torch.Tensor

    def __post_init__(self) -> None:
        if self.numeric_seed < MINIMUM_NUMERIC_NAMESPACE:
            raise ValueError("Semantic episode seed lies below the frozen numeric namespace.")
        if not 0 <= self.affected_index < self.keys.shape[0]:
            raise ValueError("affected_index lies outside the key table.")
        if self.keys.ndim != 2 or self.values.ndim != 2:
            raise ValueError("keys and values must be matrices.")
        if self.keys.shape[0] != self.values.shape[0]:
            raise ValueError("keys and values must contain the same number of associations.")
        expected_state_shape = (self.keys.shape[1], self.values.shape[1])
        if self.state.shape != expected_state_shape:
            raise ValueError("state has an incompatible shape.")
        if self.target_state.shape != self.state.shape:
            raise ValueError("target_state must match state shape.")
        if self.new_value.shape != (self.values.shape[1],):
            raise ValueError("new_value has an incompatible shape.")
        for name in ("keys", "values", "state", "target_state", "new_value"):
            tensor = getattr(self, name)
            if not bool(torch.isfinite(tensor).all().item()):
                raise FloatingPointError(f"{name} contains a non-finite value.")


@dataclass(frozen=True, slots=True)
class SemanticExample:
    safe_record: SafeSemanticRecord
    private_episode: PrivateSemanticEpisode

    @property
    def seed(self) -> int:
        return self.private_episode.checkpoint_seed

    @property
    def domain(self) -> str:
        return self.safe_record.domain

    @property
    def template(self) -> str:
        return self.safe_record.template_surface

    @property
    def operation(self) -> Operation:
        return self.private_episode.operation

    @property
    def example_id(self) -> str:
        return self.private_episode.example_id

    @property
    def state(self) -> torch.Tensor:
        return self.private_episode.state

    @property
    def target_state(self) -> torch.Tensor:
        return self.private_episode.target_state

    @property
    def keys(self) -> torch.Tensor:
        return self.private_episode.keys

    @property
    def affected_index(self) -> int:
        return self.private_episode.affected_index

    @property
    def new_value(self) -> torch.Tensor:
        return self.private_episode.new_value


@dataclass(frozen=True, slots=True)
class SemanticNamespaceRegistry:
    integer_root: int
    split_stride: int
    seed_stride: int
    split_offsets: tuple[tuple[str, int], ...]
    dry_run: bool
    prior_numeric_seed_max: int

    def __post_init__(self) -> None:
        if self.integer_root < MINIMUM_NUMERIC_NAMESPACE:
            raise ValueError("The semantic namespace root must be at least 5e12.")
        if self.integer_root <= self.prior_numeric_seed_max:
            raise ValueError("The semantic namespace overlaps a prior numeric seed range.")
        if self.split_stride <= 0 or self.seed_stride <= 0:
            raise ValueError("Namespace strides must be positive.")
        if self.split_stride % self.seed_stride:
            raise ValueError("split_stride must be an integer multiple of seed_stride.")
        names = [name for name, _ in self.split_offsets]
        offsets = [offset for _, offset in self.split_offsets]
        if len(names) != len(set(names)) or len(offsets) != len(set(offsets)):
            raise ValueError("Namespace names and offsets must be unique.")
        if any(not name or offset < 0 for name, offset in self.split_offsets):
            raise ValueError("Namespace entries are invalid.")
        if self.dry_run and any(name.startswith("e05b_") for name in names):
            raise ValueError("Dry namespace registry must not expose E05b namespaces.")

    @classmethod
    def from_config(
        cls,
        namespace_config: Mapping[str, object],
        *,
        dry_run: bool,
    ) -> SemanticNamespaceRegistry:
        offsets_key = "dry_split_offsets" if dry_run else "main_split_offsets"
        raw_offsets = namespace_config.get(offsets_key)
        if not isinstance(raw_offsets, dict):
            raise TypeError(f"{offsets_key} must be a mapping.")
        offsets: list[tuple[str, int]] = []
        for name, value in raw_offsets.items():
            if not isinstance(name, str) or isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("Namespace offsets must map strings to integers.")
            offsets.append((name, value))
        required_ints = (
            "integer_root",
            "split_stride",
            "seed_stride",
            "forbid_overlap_with_prior_numeric_seed_max",
        )
        parsed: dict[str, int] = {}
        for name in required_ints:
            value = namespace_config.get(name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"namespace.{name} must be an integer.")
            parsed[name] = value
        return cls(
            integer_root=parsed["integer_root"],
            split_stride=parsed["split_stride"],
            seed_stride=parsed["seed_stride"],
            split_offsets=tuple(sorted(offsets)),
            dry_run=dry_run,
            prior_numeric_seed_max=parsed["forbid_overlap_with_prior_numeric_seed_max"],
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.split_offsets)

    @property
    def maximum_seed_slots(self) -> int:
        return self.split_stride // self.seed_stride

    def numeric_seed(self, namespace_name: str, *, seed_slot: int, index: int) -> int:
        offset_by_name = dict(self.split_offsets)
        if namespace_name not in offset_by_name:
            raise KeyError(f"Namespace {namespace_name!r} is not open in this registry.")
        if isinstance(seed_slot, bool) or not isinstance(seed_slot, int):
            raise TypeError("seed_slot must be an integer.")
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an integer.")
        if not 0 <= seed_slot < self.maximum_seed_slots:
            raise ValueError("seed_slot lies outside its split namespace.")
        if not 0 <= index < self.seed_stride:
            raise ValueError("index lies outside its seed namespace.")
        result = (
            self.integer_root
            + offset_by_name[namespace_name] * self.split_stride
            + seed_slot * self.seed_stride
            + index
        )
        if result < MINIMUM_NUMERIC_NAMESPACE:
            raise AssertionError("Generated numeric namespace is below 5e12.")
        return result


def safe_record_field_names() -> tuple[str, ...]:
    return tuple(field.name for field in fields(SafeSemanticRecord))


def find_banned_surface_cues(record: SafeSemanticRecord) -> tuple[str, ...]:
    observed: set[str] = set()
    for field_name in ALLOWED_SAFE_FIELDS:
        value = getattr(record, field_name)
        if not isinstance(value, str):
            continue
        tokens = set(_TOKEN_RE.findall(value.lower()))
        observed.update(token for token in BANNED_SURFACE_CUES if token in tokens)
    return tuple(sorted(observed))


def _current_scope(record: SafeSemanticRecord) -> str:
    if not record.current_relation.startswith(_CURRENT_RELATION_PREFIX):
        raise ValueError("current_relation does not encode a raw current scope.")
    current_scope = record.current_relation.removeprefix(_CURRENT_RELATION_PREFIX)
    if not current_scope:
        raise ValueError("current_relation contains an empty current scope.")
    return current_scope


def derive_raw_demand(record: SafeSemanticRecord) -> RawDemand:
    """Derive demand only from the frozen raw validity/version predicates."""

    erase = record.prior_valid_to_day < record.observation_day
    write = (
        record.evidence_valid_from_day
        <= record.observation_day
        <= record.evidence_valid_to_day
        and record.evidence_version > record.prior_version
        and record.scope == _current_scope(record)
    )
    return RawDemand(erase=erase, write=write)


def derive_operation(record: SafeSemanticRecord) -> Operation:
    return derive_raw_demand(record).operation


def _opaque_token(kind: str, numeric_seed: int, salt: str = "") -> str:
    digest = hashlib.sha256(f"{kind}\0{numeric_seed}\0{salt}".encode()).hexdigest()[:20]
    return f"{kind}_{digest}"


def _stream_seed(numeric_seed: int, stream: str) -> int:
    digest = hashlib.sha256(f"{numeric_seed}\0{stream}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def build_safe_record_for_operation(
    *,
    operation: Operation,
    numeric_seed: int,
    domain: str,
    template_surface: str,
    entity_description: str | None = None,
    incoming_value_token: str | None = None,
) -> SafeSemanticRecord:
    """Construct raw predicates whose derived demand is exactly ``operation``."""

    if numeric_seed < MINIMUM_NUMERIC_NAMESPACE:
        raise ValueError("numeric_seed lies below the frozen E05 namespace.")
    observation_day = 64 + numeric_seed % 128
    prior_version = 1 + numeric_seed % 8
    expected_erase, expected_write = operation.demand
    prior_valid_from_day = observation_day - 32
    prior_valid_to_day = observation_day - 1 if expected_erase else observation_day + 8
    current_scope = f"{domain}_scope"

    if expected_write:
        evidence_version = prior_version + 1 + numeric_seed % 2
        evidence_valid_from_day = observation_day - 4
        evidence_valid_to_day = observation_day + 12
        evidence_scope = current_scope
    else:
        failure_variant = numeric_seed % 3
        evidence_version = prior_version + 1
        evidence_valid_from_day = observation_day - 4
        evidence_valid_to_day = observation_day + 12
        evidence_scope = current_scope
        if failure_variant == 0:
            evidence_version = prior_version
        elif failure_variant == 1:
            evidence_valid_from_day = observation_day - 12
            evidence_valid_to_day = observation_day - 1
        else:
            evidence_scope = _opaque_token("scope", numeric_seed)

    record = SafeSemanticRecord(
        entity_description=(
            entity_description
            if entity_description is not None
            else _opaque_token("entity", numeric_seed)
        ),
        domain=domain,
        current_relation=f"{_CURRENT_RELATION_PREFIX}{current_scope}",
        incoming_evidence=f"statement_{numeric_seed % 4}",
        prior_version=prior_version,
        evidence_version=evidence_version,
        observation_day=observation_day,
        evidence_timestamp_day=max(observation_day - 1 - numeric_seed % 3, 0),
        prior_valid_from_day=prior_valid_from_day,
        prior_valid_to_day=prior_valid_to_day,
        evidence_valid_from_day=evidence_valid_from_day,
        evidence_valid_to_day=evidence_valid_to_day,
        scope=evidence_scope,
        source=f"source_{numeric_seed % 5}",
        provenance=f"trace_{numeric_seed % 7}",
        incoming_value_token=(
            incoming_value_token
            if incoming_value_token is not None
            else _opaque_token("value", numeric_seed)
        ),
        template_surface=template_surface,
    )
    if derive_operation(record) is not operation:
        raise AssertionError("Raw semantic predicates do not derive the requested operation.")
    return record


def _orthonormal_keys(spec: SemanticMemorySpec, numeric_seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(
        _stream_seed(numeric_seed, "orthonormal-keys")
    )
    coordinates = torch.randperm(spec.key_dim, generator=generator)[
        : spec.num_associations
    ]
    signs = torch.where(
        torch.rand(spec.num_associations, generator=generator) < 0.5,
        -torch.ones(spec.num_associations, dtype=spec.dtype),
        torch.ones(spec.num_associations, dtype=spec.dtype),
    )
    keys = torch.zeros(
        spec.num_associations,
        spec.key_dim,
        dtype=spec.dtype,
    )
    keys[torch.arange(spec.num_associations), coordinates] = signs
    return keys


def _normalized_rows(
    rows: int,
    columns: int,
    *,
    numeric_seed: int,
    stream: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(_stream_seed(numeric_seed, stream))
    values = torch.randn(rows, columns, generator=generator, dtype=dtype)
    return cast(
        torch.Tensor,
        values
        / torch.linalg.vector_norm(values, dim=-1, keepdim=True).clamp_min(1e-12),
    )


def _example_digest(numeric_seed: int, checkpoint_seed: int) -> str:
    return hashlib.sha256(f"{numeric_seed}\0{checkpoint_seed}".encode()).hexdigest()[:24]


def semantic_private_identifiers(
    *,
    numeric_seed: int,
    checkpoint_seed: int,
) -> dict[str, str]:
    if numeric_seed < MINIMUM_NUMERIC_NAMESPACE:
        raise ValueError("numeric_seed lies below the frozen E05 namespace.")
    digest = _example_digest(numeric_seed, checkpoint_seed)
    return {
        "example_id": f"x_{digest}",
        "transaction_id": f"t_{digest}",
        "episode_id": f"p_{digest}",
        "old_value_token": _opaque_token("private", numeric_seed),
    }


def build_semantic_example(
    *,
    safe_record: SafeSemanticRecord,
    namespace_name: str,
    numeric_seed: int,
    checkpoint_seed: int,
    memory_spec: SemanticMemorySpec,
) -> SemanticExample:
    """Build a deterministic orthonormal-memory example without stored candidates."""

    operation = derive_operation(safe_record)
    keys = _orthonormal_keys(memory_spec, numeric_seed)
    values = _normalized_rows(
        memory_spec.num_associations,
        memory_spec.value_dim,
        numeric_seed=numeric_seed,
        stream="memory-values",
        dtype=memory_spec.dtype,
    )
    new_value = _normalized_rows(
        1,
        memory_spec.value_dim,
        numeric_seed=numeric_seed,
        stream="incoming-value",
        dtype=memory_spec.dtype,
    )[0]
    address_generator = torch.Generator(device="cpu").manual_seed(
        _stream_seed(numeric_seed, "visible-address")
    )
    affected_index = int(
        torch.randint(
            memory_spec.num_associations,
            (1,),
            generator=address_generator,
        ).item()
    )
    state = keys.transpose(0, 1) @ values
    visible_address = keys[affected_index]
    visible_old_read = visible_address @ state
    erase_candidate = torch.outer(visible_address, visible_old_read)
    write_candidate = torch.outer(visible_address, new_value)
    demand = derive_raw_demand(safe_record)
    target_state = (
        state
        - float(demand.erase) * erase_candidate
        + float(demand.write) * write_candidate
    )
    identifiers = semantic_private_identifiers(
        numeric_seed=numeric_seed,
        checkpoint_seed=checkpoint_seed,
    )
    private = PrivateSemanticEpisode(
        example_id=identifiers["example_id"],
        transaction_id=identifiers["transaction_id"],
        episode_id=identifiers["episode_id"],
        namespace_name=namespace_name,
        numeric_seed=numeric_seed,
        checkpoint_seed=checkpoint_seed,
        operation=operation,
        old_value_token=identifiers["old_value_token"],
        keys=keys,
        values=values,
        state=state,
        target_state=target_state,
        affected_index=affected_index,
        new_value=new_value,
    )
    return SemanticExample(safe_record=safe_record, private_episode=private)


def build_balanced_semantic_examples(
    *,
    namespace_registry: SemanticNamespaceRegistry,
    namespace_name: str,
    checkpoint_seed: int,
    seed_slot: int,
    operations: Sequence[Operation],
    domains: Sequence[str],
    templates: Sequence[str],
    count_per_cell: int,
    memory_spec: SemanticMemorySpec,
) -> list[SemanticExample]:
    if count_per_cell <= 0:
        raise ValueError("count_per_cell must be positive.")
    if not operations or not domains or not templates:
        raise ValueError("A balanced semantic dataset requires nonempty cell axes.")
    result: list[SemanticExample] = []
    index = 0
    for operation in operations:
        for domain in domains:
            for template_surface in templates:
                for _ in range(count_per_cell):
                    numeric_seed = namespace_registry.numeric_seed(
                        namespace_name,
                        seed_slot=seed_slot,
                        index=index,
                    )
                    safe_record = build_safe_record_for_operation(
                        operation=operation,
                        numeric_seed=numeric_seed,
                        domain=domain,
                        template_surface=template_surface,
                    )
                    result.append(
                        build_semantic_example(
                            safe_record=safe_record,
                            namespace_name=namespace_name,
                            numeric_seed=numeric_seed,
                            checkpoint_seed=checkpoint_seed,
                            memory_spec=memory_spec,
                        )
                    )
                    index += 1
    ids = [example.example_id for example in result]
    episode_ids = [example.private_episode.episode_id for example in result]
    if len(ids) != len(set(ids)) or len(episode_ids) != len(set(episode_ids)):
        raise AssertionError("Semantic dataset contains duplicate identifiers.")
    return result


def semantic_vocabularies(
    examples: Sequence[SemanticExample],
) -> dict[str, frozenset[str]]:
    return {
        "entity": frozenset(example.safe_record.entity_description for example in examples),
        "old_value": frozenset(
            example.private_episode.old_value_token for example in examples
        ),
        "new_value": frozenset(
            example.safe_record.incoming_value_token for example in examples
        ),
    }


def assert_disjoint_semantic_vocabularies(
    datasets: Mapping[str, Sequence[SemanticExample]],
) -> None:
    vocabularies = {
        name: semantic_vocabularies(examples) for name, examples in datasets.items()
    }
    names = sorted(vocabularies)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            for vocabulary_name in ("entity", "old_value", "new_value"):
                overlap = (
                    vocabularies[left_name][vocabulary_name]
                    & vocabularies[right_name][vocabulary_name]
                )
                if overlap:
                    raise ValueError(
                        f"{vocabulary_name} vocabulary overlaps between "
                        f"{left_name} and {right_name}."
                    )
