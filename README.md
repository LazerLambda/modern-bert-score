## BERTScore for the Inference Era

Reimplementation of the BERTScore metric introduced by [Zhang et al. 2019](https://arxiv.org/abs/1904.09675) with [SentenceTransformers](https://www.sbert.net/) and [vLLM](https://vllm.ai/).


![BERTScore](zhang_19_figure_1.png)



```bash
uv run python prepare_model.py \
	--model-id answerdotai/ModernBERT-base \
	--output-dir models/ModernBERT-base-vllm \
	--max-layer-index 21
```
