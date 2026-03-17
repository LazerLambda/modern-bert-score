import gc
from typing import Any, List

import torch
from sentence_transformers import SentenceTransformer
from torch.nn import functional as F
from transformers import AutoTokenizer

try:
    from vllm import LLM

    VLLM_AVAILABLE = True
except ImportError:
    LLM = object  # To prevent NameError if vllm is not installed
    VLLM_AVAILABLE = False


# TODO: Cache reference embeddings
class Inference:

    model = None

    def inference(
        self, candidates: List[str], references: List[str], **kwargs: Any
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        raise NotImplementedError("Method must be implemented in Subclass.")


class STInference(Inference):
    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        batch_size: int = 64,
        **kwargs: Any,
    ):
        self.model = SentenceTransformer(
            model_name_or_path=model_id, device=device, **kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id
        )  # TODO: Maybe switch to PreTrainedTokenizerFast for clarity?
        self.batch_size = batch_size
        self.eps: float = 1e-12

    def inference(
        self, candidates: List[str], references: List[str], **kwargs: Any
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:

        embds_refs = self.model.encode(
            references,
            output_value="token_embeddings",
            convert_to_tensor=True,
            **kwargs,
        )
        embds_refs = [
            F.normalize(e, p=2, dim=-1, eps=self.eps) for e in embds_refs
        ]
        embds_cnds = self.model.encode(
            candidates,
            output_value="token_embeddings",
            convert_to_tensor=True,
            **kwargs,
        )
        embds_cnds = [
            F.normalize(e, p=2, dim=-1, eps=self.eps) for e in embds_cnds
        ]

        return embds_cnds, embds_refs


class VLLMInference(Inference):
    def __init__(self, **kwargs: Any):
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not installed. To use the vLLM backend, please "
                "install it with `pip install vllm` or "
                "`pip install 'modern-bert-score[vllm]'`."
            )
        # Backward compatibility for old callsites that pass task="embed".
        task = kwargs.pop("task", None)
        if task == "embed":
            kwargs.setdefault("runner", "pooling")
            kwargs.setdefault("convert", "embed")

        try:
            self.model = LLM(**kwargs)
        except Exception as exc:
            message = str(exc)
            if (
                "Model architectures" in message
                and "ModernBertForMaskedLM" in message
            ):
                raise RuntimeError(
                    "vLLM does not accept the masked-LM ModernBERT checkpoint "
                    "directly. Export an encoder-only checkpoint first with "
                    "prepare_model.py, which rewrites the saved config to "
                    "advertise ModernBertModel, then load that local path in "
                    "VLLMInference. If you do not need vLLM specifically, use "
                    "STInference for the original HF checkpoint."
                ) from exc
            raise
        self.eps: float = 1e-12

    def inference(
        self, candidates: List[str], references: List[str], **kwargs: Any
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        outputs_cands = self.model.encode(
            candidates, pooling_task="token_embed", **kwargs
        )
        outputs_refs = self.model.encode(
            references, pooling_task="token_embed", **kwargs
        )
        collector: List[torch.Tensor] = []
        for output in outputs_cands:
            embeds = output.outputs.data
            collector.append(embeds)
        for output in outputs_refs:
            embeds = output.outputs.data
            collector.append(embeds)

        collector = [
            F.normalize(e, p=2, dim=-1, eps=self.eps) for e in collector
        ]  # TODO: Check superflous?
        return collector[0 : len(candidates)], collector[len(candidates) :]

    def __del__(self) -> None:
        # In case engine shuts down ungracefully and model is already freed.
        if self.model:
            del self.model
        gc.collect()
        torch.cuda.empty_cache()
