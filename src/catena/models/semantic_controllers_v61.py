from __future__ import annotations

from enum import StrEnum

import torch
from torch import nn

from catena.models.memory import GateOutput


class SemanticRoute(StrEnum):
    """The only architectural difference in the matched E05 controller pair."""

    FACTORIZED = "factorized"
    SHARED = "shared"


class MatchedSemanticControllerV61(nn.Module):
    """Two-path semantic gate controller with an exactly matched compute graph.

    Both variants own and evaluate the same parameter tensors.  The fixed
    two-by-two path-routing matrix is the only difference:

    * ``FACTORIZED`` keeps the two latent paths separate.
    * ``SHARED`` replaces both routed paths by their common average.

    The routing matrix is a non-persistent buffer so paired models can load one
    identical state dict without accidentally overwriting their constraint.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        route: SemanticRoute,
    ) -> None:
        super().__init__()
        if isinstance(input_dim, bool) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")
        if isinstance(hidden_dim, bool) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer.")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.route = SemanticRoute(route)

        self.input_weight = nn.Parameter(
            torch.empty(2, self.input_dim, self.hidden_dim)
        )
        self.input_bias = nn.Parameter(torch.empty(2, self.hidden_dim))
        self.hidden_weight = nn.Parameter(
            torch.empty(2, self.hidden_dim, self.hidden_dim)
        )
        self.hidden_bias = nn.Parameter(torch.empty(2, self.hidden_dim))
        self.head_weight = nn.Parameter(torch.empty(2, self.hidden_dim))
        self.head_bias = nn.Parameter(torch.empty(2))

        routing = (
            torch.eye(2, dtype=torch.float32)
            if self.route is SemanticRoute.FACTORIZED
            else torch.full((2, 2), 0.5, dtype=torch.float32)
        )
        self.register_buffer("_routing", routing, persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for path in range(2):
            nn.init.kaiming_uniform_(
                self.input_weight[path],
                a=5**0.5,
            )
            nn.init.kaiming_uniform_(
                self.hidden_weight[path],
                a=5**0.5,
            )
        input_bound = self.input_dim**-0.5
        hidden_bound = self.hidden_dim**-0.5
        nn.init.uniform_(self.input_bias, -input_bound, input_bound)
        nn.init.uniform_(self.hidden_bias, -hidden_bound, hidden_bound)
        nn.init.uniform_(self.head_weight, -hidden_bound, hidden_bound)
        nn.init.uniform_(self.head_bias, -hidden_bound, hidden_bound)

    @property
    def routing_matrix(self) -> torch.Tensor:
        return self._routing.detach().clone()

    def registered_dense_multiply_adds_per_example(self) -> int:
        """Return the protocol-counted dense multiply-add budget.

        The fixed routing multiply is deliberately included for both variants;
        identity routing is not optimized away in ``forward``.
        """

        return (
            2 * self.input_dim * self.hidden_dim
            + 2 * self.hidden_dim * self.hidden_dim
            + 4 * self.hidden_dim
            + 2 * self.hidden_dim
        )

    def forward(self, features: torch.Tensor) -> GateOutput:
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError(
                f"features must have shape [batch,{self.input_dim}], got "
                f"{tuple(features.shape)}."
            )
        if not bool(torch.isfinite(features).all().item()):
            raise FloatingPointError("semantic controller input is non-finite.")

        hidden = torch.einsum("bi,pih->bph", features, self.input_weight)
        hidden = torch.nn.functional.gelu(hidden + self.input_bias)
        hidden = torch.einsum("bpi,pih->bph", hidden, self.hidden_weight)
        hidden = torch.nn.functional.gelu(hidden + self.hidden_bias)
        routed = torch.einsum(
            "pq,bqh->bph",
            self._routing.to(dtype=hidden.dtype),
            hidden,
        )
        logits = torch.sum(routed * self.head_weight, dim=-1) + self.head_bias
        gates = torch.sigmoid(logits)
        return GateOutput(erase=gates[:, 0], write=gates[:, 1])


def assert_matched_semantic_pair(
    factorized: MatchedSemanticControllerV61,
    shared: MatchedSemanticControllerV61,
) -> None:
    if factorized.route is not SemanticRoute.FACTORIZED:
        raise ValueError("factorized model has the wrong route.")
    if shared.route is not SemanticRoute.SHARED:
        raise ValueError("shared model has the wrong route.")
    factorized_parameters = sum(
        parameter.numel() for parameter in factorized.parameters()
    )
    shared_parameters = sum(parameter.numel() for parameter in shared.parameters())
    if factorized_parameters != shared_parameters:
        raise AssertionError("semantic controller parameter counts differ.")
    if (
        factorized.registered_dense_multiply_adds_per_example()
        != shared.registered_dense_multiply_adds_per_example()
    ):
        raise AssertionError("semantic controller dense compute budgets differ.")
    factorized_state = factorized.state_dict()
    shared_state = shared.state_dict()
    if factorized_state.keys() != shared_state.keys():
        raise AssertionError("semantic controller state schemas differ.")
    for name in factorized_state:
        if not torch.equal(factorized_state[name], shared_state[name]):
            raise AssertionError(
                f"semantic controller common initialization differs at {name}."
            )
