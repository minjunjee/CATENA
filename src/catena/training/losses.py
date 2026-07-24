from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LossWeights:
    affected_kl: float = 1.0
    retention_kl: float = 0.6
    gold_ce: float = 0.5
    identity_kl: float = 0.2
    composition_kl: float = 0.0
    temperature: float = 1.0


def categorical_kl(student_logits, teacher_logits, temperature: float = 1.0):
    import torch.nn.functional as F

    t = float(temperature)
    teacher_prob = F.softmax(teacher_logits.float() / t, dim=-1)
    student_log_prob = F.log_softmax(student_logits.float() / t, dim=-1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (t * t)


def gold_cross_entropy(student_logits, gold_indices):
    import torch.nn.functional as F

    return F.cross_entropy(student_logits.float(), gold_indices)


def transport_objective(
    *,
    affected_student,
    affected_teacher,
    retention_student,
    retention_teacher,
    gold_logits,
    gold_indices,
    identity_student=None,
    identity_teacher=None,
    composition_student=None,
    composition_reference=None,
    weights: LossWeights = LossWeights(),
):
    import torch

    components: dict[str, torch.Tensor] = {}
    components["affected_kl"] = categorical_kl(
        affected_student, affected_teacher, weights.temperature
    )
    components["retention_kl"] = categorical_kl(
        retention_student, retention_teacher, weights.temperature
    )
    components["gold_ce"] = gold_cross_entropy(gold_logits, gold_indices)
    total = (
        weights.affected_kl * components["affected_kl"]
        + weights.retention_kl * components["retention_kl"]
        + weights.gold_ce * components["gold_ce"]
    )
    if identity_student is not None and identity_teacher is not None:
        components["identity_kl"] = categorical_kl(
            identity_student, identity_teacher, weights.temperature
        )
        total = total + weights.identity_kl * components["identity_kl"]
    if composition_student is not None and composition_reference is not None:
        components["composition_kl"] = categorical_kl(
            composition_student, composition_reference, weights.temperature
        )
        total = total + weights.composition_kl * components["composition_kl"]
    components["total"] = total
    return total, components
