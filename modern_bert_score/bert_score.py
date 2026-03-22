from collections import Counter, defaultdict
from functools import partial, reduce
from itertools import chain, islice
from math import log
from multiprocessing import Pool
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from modern_bert_score.consts import BASELINES
from modern_bert_score.inference import STInference, VLLMInference


class BertScore:
    inference_engine: Optional[Union[STInference, VLLMInference]]

    def __init__(
        self,
        model_id: str = "answerdotai/ModernBERT-base",
        idf_weighting: bool = False,
        baseline_rescaling: bool = False,
        custom_baseline: Optional[Tuple[float, float, float]] = None,
        device: str = "cpu",  # TODO Enum cuda, mlx, cpu?
        backend: str = "default",  # TODO Enum default, vllm, onnx, etc
        sentence_transformers_args: Optional[Dict[str, Any]] = None,
        vllm_args: Optional[Dict[str, Any]] = None,
    ):
        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_id)
        self.idf_weighting: bool = idf_weighting
        if baseline_rescaling:
            if model_id in BASELINES.keys():
                self.baseline: Tuple[float, float, float] = BASELINES[model_id]
                if isinstance(self.baseline, tuple):
                    self.baseline = self.baseline
            elif custom_baseline is not None:
                self.baseline = custom_baseline
            else:
                raise ValueError(
                    f"Baseline rescaling enabled but no baseline found for model {model_id} and no custom baseline provided."
                )
        self.baseline_rescaling: bool = baseline_rescaling
        if backend == "vllm":
            self.inference_engine = VLLMInference(
                model=model_id,
                runner="pooling",
                convert="embed",
                **(vllm_args or {}),
            )
        elif backend == "default":
            self.inference_engine = STInference(
                model_id, device=device, **(sentence_transformers_args or {})
            )
        else:
            raise ValueError(f"Unsupported backend {backend}")

    def __call__(
        self, candidates: Union[str, List[str]], references: Union[str, List[str]], **kwargs: Any
    ) -> List[Tuple[float, float, float]]:
        assert reduce(
            lambda acc, x: acc and isinstance(x, str), candidates, True
        ), "Candidates must be a list of strings or a string."
        assert reduce(
            lambda acc, x: acc and isinstance(x, str), references, True
        ), "References must be a list of strings or a string."
        if isinstance(candidates, str):
            candidates = [candidates]
        if isinstance(references, str):
            references = [references]
        if len(candidates) != len(references):
            raise ValueError(
                "Number of candidates and references must be the same."
            )
        if len(candidates) == 0:
            return []
        candidates = [c.strip() for c in candidates]
        references = [r.strip() for r in references]
        if self.idf_weighting:
            idf_dict_ref, input_ids_ref = self.get_idf_dict(references)
            input_ids_cand = self._tokenize_data(candidates)
        else:
            idf_dict_ref = None

        if self.inference_engine is None:
            raise ValueError(
                "Inference engine not initialized. Check backend "
                "configuration."
            )
        cand_embs, ref_embs = self.inference_engine.inference(
            candidates, references, **kwargs
        )
        if self.idf_weighting:
            scores = [
                self.bert_score(
                    candidates=c,
                    references=r,
                    idf_dict_ref=idf_dict_ref,
                    input_ids_cand=ids_cand,
                    input_ids_ref=ids_ref,
                )
                for c, r, ids_cand, ids_ref in zip(
                    cand_embs, ref_embs, input_ids_cand, input_ids_ref
                )
            ]
        else:
            scores = [
                self.bert_score(candidates=c, references=r)
                for c, r in zip(cand_embs, ref_embs)
            ]
        if self.baseline_rescaling:
            rescaled_scores = []
            for p, r, f1 in scores:
                rescaled_p = (p - self.baseline[0]) / (1 - self.baseline[0])
                rescaled_r = (r - self.baseline[1]) / (1 - self.baseline[1])
                rescaled_f1 = (f1 - self.baseline[2]) / (1 - self.baseline[2])
                rescaled_scores.append((rescaled_p, rescaled_r, rescaled_f1))
            return rescaled_scores
        return scores

    @staticmethod
    def _check_nan(f1: torch.Tensor) -> torch.Tensor:
        if torch.isnan(f1):
            f1 = torch.Tensor([0.0])
        return f1


    def bert_score(
        self,
        candidates: torch.Tensor,
        references: torch.Tensor,
        idf_dict_ref: Optional[Dict[int, float]] = None,
        input_ids_cand: Optional[List[int]] = None,
        input_ids_ref: Optional[List[int]] = None,
    ) -> Tuple[float, float, float]:
        has_idf_dict = idf_dict_ref is not None
        has_input_ids = (
            input_ids_cand is not None and input_ids_ref is not None
        )
        if has_idf_dict != has_input_ids:
            raise ValueError(
                "`idf_dict` and `input_ids` must either both be provided or "
                "both be None."
            )
        # TODO: w cuda?
        assert len(candidates.shape) == 2 and len(references.shape) == 2
        candidates = candidates[1:-1]  # remove CLS and SEP
        references = references[1:-1]
        similarities: torch.tensor = candidates @ references.T
        r_bert = similarities.max(dim=0).values.cpu()
        p_bert = similarities.max(dim=1).values.cpu()
        if idf_dict_ref and input_ids_cand and input_ids_ref:
            idf_weights_cand = torch.tensor(
                [idf_dict_ref[tok_id] for tok_id in input_ids_cand]
            )
            idf_weights_ref = torch.tensor(
                [idf_dict_ref[tok_id] for tok_id in input_ids_ref]
            )
            idf_weights_cand /= idf_weights_cand.sum()
            idf_weights_ref /= idf_weights_ref.sum()
            p_bert = (p_bert * idf_weights_cand).sum()
            r_bert = (r_bert * idf_weights_ref).sum()
        else:
            r_bert = r_bert.mean()
            p_bert = p_bert.mean()
        f1 = 2 * p_bert * r_bert / (p_bert + r_bert)
        # handle p_bert + r_bert == 0
        f1 = self._check_nan(f1)
        return p_bert.item(), r_bert.item(), f1.item()

    @staticmethod
    def _batchify(
        iterable: List[str], batch_size: int
    ) -> Generator[List[str], None, None]:
        iterator = iter(iterable)
        while True:
            batch = list(islice(iterator, batch_size))
            if not batch:
                break
            yield batch

    @staticmethod
    def _process_batch(
        batch: List[str],
        tokenizer: PreTrainedTokenizerFast,
        ignore_counter: bool = False,
    ) -> Tuple[Counter[int], List[List[int]]]:
        stripped_batch = [sample.strip() for sample in batch]

        encoded_batch = tokenizer(
            stripped_batch,
            add_special_tokens=True,
            truncation=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        encoded_batch = [e[1:-1] for e in encoded_batch]  # remove CLS and SEP
        if ignore_counter:
            return Counter(), encoded_batch
        else:
            batch_count: Counter[int] = Counter(
                chain.from_iterable(map(set, encoded_batch))
            )
            return batch_count, encoded_batch

    def _tokenize_data(self, corpus: List[str], batch_size: int = 100_000, nthreads: int = 4) -> List[List[int]]:
        collected_input_ids: List[List[int]] = []

        process_partial = partial(
            self._process_batch, tokenizer=self.tokenizer, ignore_counter=True
        )
        batches = self._batchify(corpus, batch_size)

        if nthreads > 0:
            with Pool(nthreads) as pool:
                for batch_result in pool.imap(
                    process_partial, batches, chunksize=1
                ):
                    _, batch_input_ids = batch_result
                    collected_input_ids.extend(batch_input_ids)
        else:
            for batch_result in map(process_partial, batches):
                _, batch_input_ids = batch_result
                collected_input_ids.extend(batch_input_ids)
        return collected_input_ids

    def get_idf_dict(
        self,
        corpus: List[str],
        nthreads: int = 4,
        batch_size: int = 100_000,
    ) -> Tuple[Dict[int, float], List[List[int]]]:  # TODO: Return dict
        """Build an IDF (Inverse-Document-Frequency) dictionary for a corpus.

        When ``return_input_ids`` is true, this also returns the tokenized
        ``input_ids`` for each corpus entry in the same order as ``corpus``.
        """
        idf_count: Counter[int] = Counter()
        collected_input_ids: List[List[int]] = []
        num_docs = len(corpus)

        process_partial = partial(
            self._process_batch, tokenizer=self.tokenizer
        )
        batches = self._batchify(corpus, batch_size)

        if nthreads > 0:
            with Pool(nthreads) as pool:
                for batch_result in pool.imap(
                    process_partial, batches, chunksize=1
                ):
                    batch_count, batch_input_ids = batch_result
                    collected_input_ids.extend(batch_input_ids)
                    idf_count.update(batch_count)
        else:
            for batch_result in map(process_partial, batches):
                batch_count, batch_input_ids = batch_result
                collected_input_ids.extend(batch_input_ids)
                idf_count.update(batch_count)

        idf_dict: Dict[int, float] = defaultdict(
            lambda: log((num_docs + 1) / (1))
        )
        idf_dict.update(
            {
                idx: log((num_docs + 1) / (count + 1))
                for idx, count in idf_count.items()
            }
        )
        return idf_dict, collected_input_ids

def ModernBERTBaseScore(**kwargs: Any) -> "BertScore":
    """BertScore with ModernBERT-base-ModBERTScore-12"""
    kwargs.pop("model_id", None)
    return BertScore(model_id="LazerLambda/ModernBERT-base-ModBERTScore-12", **kwargs)


def ModernBERTLargeScore(**kwargs: Any) -> "BertScore":
    """BertScore with ModernBERT-large-ModBERTScore-19"""
    kwargs.pop("model_id", None)
    return BertScore(model_id="LazerLambda/ModernBERT-large-ModBERTScore-19", **kwargs)


def RobertaBaseScore(**kwargs: Any) -> "BertScore":
    """BertScore with roberta-base-ModBERTScore-10"""
    kwargs.pop("model_id", None)
    return BertScore(model_id="LazerLambda/roberta-base-ModBERTScore-10", **kwargs)


def RobertaLargeScore(**kwargs: Any) -> "BertScore":
    """BertScore with roberta-large-ModBERTScore-17"""
    kwargs.pop("model_id", None)
    return BertScore(model_id="LazerLambda/roberta-large-ModBERTScore-17", **kwargs)


def RobertaLargeMNLIScore(**kwargs: Any) -> "BertScore":
    """BertScore with roberta-large-mnli-ModBERTScore-19"""
    kwargs.pop("model_id", None)
    return BertScore(model_id="LazerLambda/roberta-large-mnli-ModBERTScore-19", **kwargs)
