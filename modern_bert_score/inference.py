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
    """Abstract base class for inference backends."""

    model: Any = None

    def inference(
        self, candidates: List[str], references: List[str], **kwargs: Any
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Computes embeddings for candidates and references.

        Args:
            candidates: A list of candidate strings.
            references: A list of reference strings.
            **kwargs: Additional arguments passed to the underlying model.

        Returns:
            A tuple containing two lists of tensors (candidate_embeddings, reference_embeddings).
        """
        raise NotImplementedError("Method must be implemented in Subclass.")


class STInference(Inference):
    """Inference backend using SentenceTransformers."""

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        batch_size: int = 64,
        **kwargs: Any,
    ):
        """Initializes the SentenceTransformers inference backend.

        Args:
            model_id: The model identifier or path.
            device: The device to load the model on (e.g., 'cpu', 'cuda').
            batch_size: Batch size for inference.
            **kwargs: Additional arguments for SentenceTransformer.
        """
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
        """Computes embeddings using SentenceTransformers.

        Args:
            candidates: A list of candidate strings.
            references: A list of reference strings.
            **kwargs: Additional arguments passed to `model.encode`.

        Returns:
            A tuple containing two lists of tensors (candidate_embeddings, reference_embeddings).
        """

        if self.model is None:
            raise RuntimeError("Model not loaded.")
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
    """Inference backend using vLLM."""

    def __init__(self, **kwargs: Any):
        """Initializes the vLLM inference backend.

        Args:
            **kwargs: Arguments passed to the vLLM `LLM` constructor.
        """
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not installed. To use the vLLM backend, please "
                "install it with `pip install vllm` or "
                "`pip install 'modern-bert-score[vllm]'`."
            )
        # Backward compatibility for old callsites that pass task="embed".
        kwargs = self._prepare_args(kwargs)
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

    @staticmethod
    def _prepare_args(kwargs: Any) -> Any:
        """Prepares arguments for vLLM, setting defaults for embedding tasks."""
        task = kwargs.pop("task", None)
        if task == "embed":
            kwargs.setdefault("runner", "pooling")
            kwargs.setdefault("convert", "embed")
        return kwargs

    def inference(
        self, candidates: List[str], references: List[str], **kwargs: Any
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Computes embeddings using vLLM.

        Args:
            candidates: A list of candidate strings.
            references: A list of reference strings.
            **kwargs: Additional arguments passed to `model.encode`.

        Returns:
            A tuple containing two lists of tensors (candidate_embeddings, reference_embeddings).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")
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

    def cleanup(self) -> None:
        """Cleans up the vLLM model and frees GPU memory."""
        if hasattr(self, "model") and self.model:
            del self.model
        gc.collect()
        torch.cuda.empty_cache()
