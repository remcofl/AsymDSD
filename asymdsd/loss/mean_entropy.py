import torch
import torch.distributed as dist
from torch import nn


class MeanEntropyLoss(nn.Module):
    def __init__(self, dim: int, use_momentum: bool = False) -> None:
        super().__init__()
        self.use_momentum = use_momentum
        if use_momentum:
            self.register_buffer("momentum_prob", torch.full((dim,), 1.0 / dim))

    @torch.no_grad()
    def update_momentum_entropy(
        self, batch_prob: torch.Tensor, momentum: float
    ) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_prob, op=dist.ReduceOp.AVG)

        self.momentum_prob = momentum * self.momentum_prob + (1 - momentum) * batch_prob

    # Simply computes the mean entropy over the batch
    def forward(
        self, logits: torch.Tensor, *, momentum: float | None = None
    ) -> torch.Tensor:
        # MC estimate of marginal p(z) = E_(x~p(x))[(p_model(z|x)]
        probs = nn.functional.softmax(logits, dim=1).mean(dim=0)

        if self.use_momentum:
            # TODO: Consider updating momentum after computing mean entropy.
            if momentum is None:
                raise ValueError("Momentum value must be provided.")
            self.update_momentum_entropy(probs, momentum)

            # Note probs in the case of momentum does not require gradient.
            target_probs = self.momentum_prob

            # Estimate of marginal H(p(z))
            # Both the density and log prob are optimized wrt better EMA estimate.
            mean_entropy = -(probs * target_probs.log()).sum()
            mean_entropy -= (target_probs * probs.log()).sum()

            # Note this is actuall becomes cross entropy. Therefore KL divergence is included.
            # This value is not bounded.
            mean_entropy /= 2
        else:
            # Estimate of marginal H(p(z))
            mean_entropy = -(probs * probs.log()).sum()

        mean_entropy = (probs ** (-probs)).log().sum()
        return mean_entropy
