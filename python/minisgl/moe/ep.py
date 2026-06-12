import torch
import torch.distributed as dist
from minisgl.distributed import get_ep_info
from minisgl.distributed.impl import ep_all_to_all, get_ep_group
from minisgl.moe.base import BaseMoeBackend
from minisgl.moe.fused import fused_experts_impl, fused_topk


def _dispatch_combine(
    local_hidden: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    local_gating: torch.Tensor,
    topk: int,
    renormalize: bool,
    activation: str,
    ep_size: int,
) -> torch.Tensor:
    """Run one EP dispatch -> local expert compute -> combine for this rank's
    token slice. Every collective here is called unconditionally so that ranks
    with an empty slice (n_local == 0) stay in lock-step and do not deadlock."""
    n_local, hidden_size = local_hidden.shape
    num_local_experts = w1.shape[0]
    num_pairs = n_local * topk

    if n_local > 0:
        topk_weights, topk_ids = fused_topk(
            hidden_states=local_hidden,
            gating_output=local_gating,
            topk=topk,
            renormalize=renormalize,
        )
        flat_ids = topk_ids.view(-1)
        dest_rank = flat_ids.to(torch.int64) // num_local_experts
        local_ids = (flat_ids % num_local_experts).to(torch.int32)

        token_idx = (
            torch.arange(n_local, device=local_hidden.device)
            .unsqueeze(1)
            .expand(-1, topk)
            .reshape(-1)
        )
        sort_idx = torch.argsort(dest_rank, stable=True)
        sorted_token_idx = token_idx[sort_idx]
        sorted_local_ids = local_ids[sort_idx]
        send_hidden = local_hidden[sorted_token_idx].contiguous()
        send_counts = torch.bincount(dest_rank, minlength=ep_size)
    else:
        # Empty slice: still must participate in every collective below.
        topk_weights = local_hidden.new_zeros(0, topk)
        sort_idx = local_hidden.new_zeros(0, dtype=torch.int64)
        sorted_local_ids = local_hidden.new_zeros(0, dtype=torch.int32)
        send_hidden = local_hidden.new_empty(0, hidden_size)
        send_counts = torch.zeros(ep_size, dtype=torch.int64, device=local_hidden.device)

    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts, group=get_ep_group())

    send_splits = send_counts.tolist()
    recv_splits = recv_counts.tolist()
    total_recv = sum(recv_splits)

    recv_hidden = local_hidden.new_empty(total_recv, hidden_size)
    ep_all_to_all(recv_hidden, send_hidden, recv_splits, send_splits)

    recv_ids = sorted_local_ids.new_empty(total_recv)
    ep_all_to_all(recv_ids, sorted_local_ids, recv_splits, send_splits)

    if total_recv > 0:
        unit_weights = torch.ones(total_recv, 1, dtype=torch.float32, device=local_hidden.device)
        local_out = fused_experts_impl(
            recv_hidden,
            w1,
            w2,
            unit_weights,
            recv_ids.unsqueeze(1),
            activation=activation,
            apply_router_weight_on_input=False,
        )
    else:
        local_out = local_hidden.new_empty(0, hidden_size)

    combined = local_hidden.new_empty(num_pairs, hidden_size)
    ep_all_to_all(combined, local_out, send_splits, recv_splits)

    if n_local == 0:
        return local_hidden.new_empty(0, hidden_size)

    result = local_hidden.new_empty(num_pairs, hidden_size)
    result[sort_idx] = combined
    result = result.view(n_local, topk, hidden_size)
    weights = topk_weights.to(local_hidden.dtype).unsqueeze(-1)
    return (result * weights).sum(dim=1)


class EPMoe(BaseMoeBackend):
    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
    ) -> torch.Tensor:
        assert not apply_router_weight_on_input, "EP backend does not support router-weight-on-input"
        ep_info = get_ep_info()
        ep_size, ep_rank = ep_info.size, ep_info.rank
        num_tokens, hidden_size = hidden_states.shape

        # hidden_states / gating_output are TP-replicated (identical on every EP
        # rank), so each rank owns a contiguous slice of tokens with no extra
        # communication. This makes the dispatched work 1/ep_size per rank
        # instead of recomputing every token on every rank.
        chunk = (num_tokens + ep_size - 1) // ep_size
        start = min(ep_rank * chunk, num_tokens)
        end = min(start + chunk, num_tokens)

        local_out = _dispatch_combine(
            local_hidden=hidden_states[start:end],
            w1=w1,
            w2=w2,
            local_gating=gating_output[start:end],
            topk=topk,
            renormalize=renormalize,
            activation=activation,
            ep_size=ep_size,
        )

        # All-gather the per-rank output slices back into the full, replicated
        # activation expected by the rest of the (TP) model. Pad to `chunk` so
        # every rank contributes an equal-sized block, then trim the padding.
        padded = hidden_states.new_zeros(chunk, hidden_size)
        padded[: end - start] = local_out
        gathered = hidden_states.new_empty(ep_size * chunk, hidden_size)
        dist.all_gather_into_tensor(gathered, padded, group=get_ep_group())
        return gathered[:num_tokens]
