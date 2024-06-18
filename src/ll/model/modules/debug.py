from logging import getLogger

import torch
import torch.distributed

log = getLogger(__name__)


class DebugModuleMixin:
    @torch.jit.unused
    def breakpoint(self, rank_zero_only: bool = True):
        if (
            not rank_zero_only
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        ):
            breakpoint()

        if rank_zero_only and torch.distributed.is_initialized():
            _ = torch.distributed.barrier()

    @torch.jit.unused
    def ensure_finite(
        self,
        tensor: torch.Tensor,
        name: str | None = None,
        throw: bool = False,
    ):
        name_parts: list[str] = ["Tensor"]
        if name is not None:
            name_parts.append(name)
        name = " ".join(name_parts)

        not_finite = ~torch.isfinite(tensor)
        if not_finite.any():
            msg = f"{name} has {not_finite.sum().item()}/{not_finite.numel()} non-finite values."
            if throw:
                raise RuntimeError(msg)
            else:
                log.warning(msg)
            return False
        return True
