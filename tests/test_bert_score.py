import unittest
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import logging as transformers_logging

from modern_bert_score.bert_score import BertScore
from modern_bert_score.inference import VLLM_AVAILABLE

# Suppress transformers warnings about missing weights during testing
transformers_logging.set_verbosity_error()

TEST_MODEL= "LazerLambda/BERT-Tiny-L-2-H-128-A-2-ModBERTScore-TEST"

class TestBertScore(unittest.TestCase):

    test_model: str = TEST_MODEL

    def setUp(self):
        self.candidates = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
            "The cat is on the table",
        ]
        self.references = [
            "Hello, my name is",
            "The head of the United States is",
            "The capital of Japan is",
            "The future of Work is",
            "The dog is on the table",
        ]

    def test_against_original(self):
        """Test basic functionality."""
        original_p_r_f1 = [
            torch.tensor([1.0000, 0.9423, 0.8705, 0.9105, 0.9637]),
            torch.tensor([1.0000, 0.9423, 0.8705, 0.9105, 0.9637]),
            torch.tensor([1.0000, 0.9423, 0.8705, 0.9105, 0.9637])
        ]
        original_p_r_f1 = torch.stack(original_p_r_f1).T
        original_p_r_f1_idf = [
            torch.tensor([1.0000, 0.9233, 0.8167, 0.8609, 0.9369]),
            torch.tensor([1.0000, 0.9358, 0.8304, 0.8861, 0.9505]),
            torch.tensor([1.0000, 0.9295, 0.8235, 0.8733, 0.9436])
        ]
        original_p_r_f1_idf = torch.stack(original_p_r_f1_idf).T
        bs = BertScore(model_id=self.test_model, backend="default")
        p_r_f1 = bs(self.candidates, self.references)
        bs.idf_weighting = True
        p_r_f1_idf = bs(self.candidates, self.references)
        for (p, r, f1), (p_exp, r_exp, f1_exp) in zip(p_r_f1, original_p_r_f1):
            self.assertTrue(torch.allclose(torch.tensor(p), p_exp, atol=1e-4))
            self.assertTrue(torch.allclose(torch.tensor(r), r_exp, atol=1e-4))
            self.assertTrue(torch.allclose(torch.tensor(f1), f1_exp, atol=1e-4))
        for (p, r, f1), (p_exp, r_exp, f1_exp) in zip(p_r_f1_idf, original_p_r_f1_idf):
            self.assertTrue(torch.allclose(torch.tensor(p), p_exp, atol=1e-4))
            self.assertTrue(torch.allclose(torch.tensor(r), r_exp, atol=1e-4))
            self.assertTrue(torch.allclose(torch.tensor(f1), f1_exp, atol=1e-4))

    def test_id(self):
        cand1 = ["Hello World!"]
        ref1 = ["Hello World!"]
        bs = BertScore(model_id=self.test_model, backend="default")
        p_r_f1 = bs(cand1, ref1)
        self.assertEqual(p_r_f1[0][0], 1.0)
        self.assertEqual(p_r_f1[0][1], 1.0)
        self.assertEqual(p_r_f1[0][2], 1.0)

        cand2 = ["Hello World!"]
        ref2 = [" Hello World! "]
        p_r_f1 = bs(cand2, ref2)
        self.assertEqual(p_r_f1[0][0], 1.0)
        self.assertEqual(p_r_f1[0][1], 1.0)
        self.assertEqual(p_r_f1[0][2], 1.0)

    def test_unequal_length(self):
        cand1 = ["Hello World!"]
        ref1 = ["Hello World!", "Hello World!"]
        bs = BertScore(model_id=self.test_model, backend="default")
        with self.assertRaises(ValueError):
            bs(cand1, ref1)

        cand2 = ["Hello World!", "Hello World!"]
        ref2 = ["Hello World!"]
        bs = BertScore(model_id=self.test_model, backend="default")
        with self.assertRaises(ValueError):
            bs(cand2, ref2)

    def test_unequal_input(self):
        cand1 = ["Hello World!"]
        ref1 = ["Bye World!"]
        bs = BertScore(model_id=self.test_model, backend="default")
        p_r_f1 = bs(cand1, ref1)
        self.assertTrue(p_r_f1[0][0] < 1.0)
        self.assertTrue(p_r_f1[0][1] < 1.0)
        self.assertTrue(p_r_f1[0][2] < 1.0)

    def test_empty_input(self):
        cand1 = []
        ref1 = []
        bs = BertScore(model_id=self.test_model, backend="default")
        p_r_f1 = bs(cand1, ref1)
        self.assertEqual(p_r_f1, [])

    def test_empty_inference_engine(self):
        bs = BertScore(model_id=self.test_model, backend="default")
        bs.inference_engine = None
        with self.assertRaises(ValueError):
             bs(self.candidates, self.references)

    def test_check_nan(self):
        f1_nan = torch.tensor(torch.nan)
        f1_checked = BertScore._check_nan(f1_nan)
        self.assertEqual(f1_checked, 0.0)

    def test_exception_idf_bertscore(self):
        cand1 = ["Hello World!"]
        ref1 = ["Hello World!"]
        bs = BertScore(model_id=self.test_model, backend="default")
        with self.assertRaises(ValueError):
            bs.bert_score(
                candidates=torch.rand(3, 4),
                references=torch.rand(3, 4),
                input_ids_cand=[0, 1, 2],
                input_ids_ref=[0, 1, 2],
            )

    def test_single(self):
        cand1 = "Hello World!"
        ref1 = "Hello World!"
        bs = BertScore(model_id=self.test_model, backend="default")
        bs(cand1, ref1)

    def test_tokenize_batch(self):
        bs = BertScore(model_id=self.test_model, backend="default")
        counter, input_ids = bs._process_batch(
            ["Hello World!", "Hello World!"],
            bs.tokenizer,
            ignore_counter=True
        )
        assert len(counter) == 0
        counter, input_ids = bs._process_batch(
            ["Hello World!", "Hello World!"],
            bs.tokenizer,
            ignore_counter=False
        )
        assert counter is not None

    def test_tokenize_data(self):
        bs = BertScore(model_id=self.test_model, backend="default")
        input_ids = bs._tokenize_data(["Hello World!", "Hello World!"], nthreads=4)
        input_ids = bs._tokenize_data(["Hello World!", "Hello World!"], nthreads=0)

    def test_get_idf_dict(self):
        bs = BertScore(model_id=self.test_model, backend="default")
        input_ids = bs.get_idf_dict(["Hello World!", "Hello World!"], nthreads=4)
        input_ids = bs.get_idf_dict(["Hello World!", "Hello World!"], nthreads=0)

    def test_base_line(self):
        cand1 = ["Hello World!"]
        ref1 = ["Hello World!"]
        bs = BertScore(model_id=self.test_model, backend="default", baseline_rescaling=True, custom_baseline=(0.5, 0.5, 0.5))
        p_r_f1 = bs(cand1, ref1)
        self.assertEqual(p_r_f1[0][0], 1.0)
        self.assertEqual(p_r_f1[0][1], 1.0)
        self.assertEqual(p_r_f1[0][2], 1.0)

    def test_baseline_rescaling_exception(self):
        """Test that ValueError is raised when baseline is missing."""
        with self.assertRaises(ValueError):
            BertScore(model_id=self.test_model, backend="default", baseline_rescaling=True)

    def test_baseline_rescaling_injection(self):
        """Test baseline rescaling with injected baseline for the test model."""
        from modern_bert_score.consts import BASELINES
        # Inject test model baseline (P, R, F1) tuple. F1 is at index 2.
        BASELINES[self.test_model] = (0.5, 0.5, 0.5)
        try:
            bs = BertScore(model_id=self.test_model, backend="default", baseline_rescaling=True)
            self.assertEqual(bs.baseline, (0.5, 0.5, 0.5))
            # Test computation
            cand1 = ["Hello World!"]
            ref1 = ["Hello World!"]
            p_r_f1 = bs(cand1, ref1)
            self.assertEqual(p_r_f1[0][2], 1.0)
        finally:
            del BASELINES[self.test_model]

    def test_inference_base_not_implemented_error(self):
        from modern_bert_score.inference import Inference
        with self.assertRaises(NotImplementedError):
            dummy = Inference()
            dummy.inference(["Hello"], ["Hello"])

    @patch("modern_bert_score.bert_score.AutoTokenizer")
    @patch("modern_bert_score.bert_score.STInference")
    def test_initialization_default(self, mock_st_inference, mock_tokenizer):
        """Test initialization with default backend."""
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = MagicMock()

        bs = BertScore(model_id=self.test_model, backend="default")

        mock_tokenizer.from_pretrained.assert_called_with(self.test_model)
        mock_st_inference.assert_called_once()
        self.assertEqual(bs.inference_engine, mock_st_inference.return_value)

    @patch("modern_bert_score.bert_score.AutoTokenizer")
    @patch("modern_bert_score.bert_score.VLLMInference")
    def test_initialization_vllm(self, mock_vllm_inference, mock_tokenizer):
        """Test initialization with vllm backend."""
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = MagicMock()

        bs = BertScore(model_id=self.test_model, backend="vllm")

        mock_vllm_inference.assert_called_once()
        self.assertEqual(bs.inference_engine, mock_vllm_inference.return_value)

    @patch("modern_bert_score.bert_score.AutoTokenizer")
    @patch("modern_bert_score.inference.VLLM_AVAILABLE", False)
    def test_initialization_vllm_without_vllm_installed(self, mock_tokenizer):
        """Test that using vllm backend without vllm installed raises ImportError."""
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        with self.assertRaises(ImportError) as cm:
            BertScore(model_id=self.test_model, backend="vllm")

        self.assertIn("vLLM is not installed", str(cm.exception))
        self.assertIn("pip install 'modern-bert-score[vllm]'", str(cm.exception))

    @patch("modern_bert_score.inference.LLM")
    @patch("modern_bert_score.inference.VLLM_AVAILABLE", True)
    @patch("modern_bert_score.bert_score.AutoTokenizer")
    def test_initialization_vllm_masked_lm_error(self, mock_tokenizer, mock_llm):
        """Test that appropriate error is raised when vLLM rejects MaskedLM architecture.

        This test uses mocks for vLLM components, so it will run (and pass) regardless
        of whether the `vllm` package is installed in the test environment.
        """
        mock_tokenizer.from_pretrained.return_value = MagicMock()

        # Simulate vLLM raising exception about architecture
        mock_llm.side_effect = Exception(
            "ValueError: Model architectures ['ModernBertForMaskedLM'] are not supported for now. "
            "Supported architectures: ['ModernBertModel', ...]"
        )

        with self.assertRaises(RuntimeError) as cm:
            BertScore(model_id=self.test_model, backend="vllm")

        self.assertIn(
            "vLLM does not accept the masked-LM ModernBERT checkpoint directly",
            str(cm.exception),
        )

    @patch("modern_bert_score.bert_score.AutoTokenizer")
    def test_initialization_invalid_backend(self, mock_tokenizer):
        """Test initialization with invalid backend raises ValueError."""
        with self.assertRaises(ValueError):
            BertScore(model_id=self.test_model, backend="invalid")

    @patch("modern_bert_score.bert_score.AutoTokenizer")
    @patch("modern_bert_score.bert_score.STInference")
    def test_call_simple(self, mock_st_inference, mock_tokenizer):
        """Test scoring call with default backend."""
        mock_engine = MagicMock()
        mock_st_inference.return_value = mock_engine

        # Mock embeddings: List[Tensor]
        # Need shape [seq_len, hidden_dim].
        # BertScore slices [1:-1] (removes CLS and SEP), so we need at least 3 tokens.
        c_emb = torch.rand(3, 4)
        r_emb = torch.rand(3, 4)

        mock_engine.inference.return_value = ([c_emb], [r_emb])

        bs = BertScore(model_id=self.test_model, backend="default")
        results = bs(self.candidates, self.references)

        self.assertEqual(len(results), 1)
        p, r, f1 = results[0]
        self.assertIsInstance(p, float)
        self.assertIsInstance(r, float)
        self.assertIsInstance(f1, float)

        # Verify inference called correctly
        mock_engine.inference.assert_called_with(self.candidates, self.references)

    @patch("modern_bert_score.bert_score.AutoTokenizer")
    @patch("modern_bert_score.bert_score.STInference")
    def test_input_validation(self, mock_st_inference, mock_tokenizer):
        """Test input validation for candidates/references."""
        bs = BertScore(model_id=self.test_model, backend="default")

        # Mismatched lengths
        with self.assertRaises(ValueError):
            bs(["a"], ["b", "c"])

@pytest.fixture(scope="module")
def vllm_bert_score():
    # This setup runs once for the module
    bs = BertScore(
        model_id=TEST_MODEL,
        backend="vllm",
        vllm_args={"gpu_memory_utilization": 0.3, "enforce_eager": True, "distributed_executor_backend": "mp", "task": "embed"}
    )
    return bs

# 2. The actual test function
@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not installed")
def test_vllm_only_feature(vllm_bert_score):
    """Test that requests the fixture above."""
    cand1 = ["Hello World!"]
    ref1 = ["Hello World!"]

    # Use the fixture passed as an argument
    p_r_f1 = vllm_bert_score(cand1, ref1)

    # Use standard asserts instead of self.assertEqual
    assert p_r_f1[0][0] == 1.0
    assert p_r_f1[0][1] == 1.0
    assert p_r_f1[0][2] == 1.0

    # Test kwargs passed to vLLMInference
    # kwargs = {"task": "embed", "gpu_memory_utilization": 0.3, "enforce_eager": True, "distributed_executor_backend": "mp"}
