from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import pp_layer_range, try_get_pp_info
from minisgl.distributed.impl import pp_recv, pp_send
from minisgl.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP as Qwen3MLP
from .utils import RopeAttn as Qwen3Attn

if TYPE_CHECKING:
    from .config import ModelConfig


class Qwen3DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = Qwen3Attn(config, layer_id, has_qk_norm=True)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class Qwen3Model(BaseOP):
    def __init__(self, config: ModelConfig):
        pp_info = try_get_pp_info()
        self.pp_rank = pp_info.rank if pp_info is not None else 0
        self.pp_size = pp_info.size if pp_info is not None else 1
        self.is_first = self.pp_rank == 0
        self.is_last = self.pp_rank == self.pp_size - 1

        start, end = pp_layer_range(config.num_layers, self.pp_rank, self.pp_size)
        self._hidden_size = config.hidden_size
        self._dtype = torch.get_default_dtype()

        # Only the first stage embeds tokens; only the last stage applies the
        # final norm. Decoder layers are local (ids 0..n-1) indexing this
        # stage's KV cache; their checkpoint weights are remapped by the loader.
        self.embed_tokens = (
            VocabParallelEmbedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
            )
            if self.is_first
            else None
        )
        self.layers = OPList(
            [Qwen3DecoderLayer(config, local_id) for local_id in range(end - start)]
        )
        self.norm = (
            RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
            if self.is_last
            else None
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.is_first:
            x = self.embed_tokens.forward(input_ids)
        else:
            # Receive the materialized hidden stream from the previous stage.
            x = torch.empty(
                input_ids.shape[0], self._hidden_size, dtype=self._dtype, device=input_ids.device
            )
            pp_recv(x, src=self.pp_rank - 1)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        if self.is_last:
            return self.norm.forward(x, residual)[0]
        # Materialize the residual stream and hand it to the next stage.
        hidden = x + residual if residual is not None else x
        pp_send(hidden.contiguous(), dst=self.pp_rank + 1)
        return hidden


class Qwen3ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3Model(config)
        self.is_last = self.model.is_last
        if config.tie_word_embeddings and self.model.pp_size > 1:
            raise NotImplementedError("Pipeline parallelism does not support tied word embeddings")
        self.lm_head = (
            ParallelLMHead(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                tie_word_embeddings=config.tie_word_embeddings,
                tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            )
            if self.is_last
            else None
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        if not self.is_last:
            return output  # non-last stages do not sample; value is unused
        return self.lm_head.forward(output)


__all__ = ["Qwen3ForCausalLM"]
