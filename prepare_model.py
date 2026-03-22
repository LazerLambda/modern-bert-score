#!/usr/bin/env python3
"""Prepare a trimmed BERT-style encoder checkpoint.

This script loads a Hugging Face model, keeps encoder layers up to an
inclusive index, removes LM-head components by exporting only the base encoder,
and saves the prepared model to disk.
"""

from __future__ import annotations

import argparse
from typing import Optional

import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer


def _get_nested_attr(obj: object, path: str) -> Optional[object]:
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _find_encoder_layers(
    model: torch.nn.Module,
) -> tuple[str, torch.nn.ModuleList]:
    candidate_paths = [
        "layers",
        "encoder.layer",
        "encoder.layers",
        "bert.encoder.layer",
        "roberta.encoder.layer",
        "deberta.encoder.layer",
        "deberta_v2.encoder.layer",
        "electra.encoder.layer",
        "distilbert.transformer.layer",
        "transformer.layer",
    ]

    for path in candidate_paths:
        layers = _get_nested_attr(model, path)
        if isinstance(layers, torch.nn.ModuleList):
            return path, layers

    raise ValueError(
        "Could not find an encoder layer stack on this model. "
        "Expected a BERT-style architecture with a ModuleList of layers."
    )


def _load_model(
    model_id: str, trust_remote_code: bool, revision: Optional[str]
) -> torch.nn.Module:
    common_kwargs = {
        "trust_remote_code": trust_remote_code,
    }
    if revision is not None:
        common_kwargs["revision"] = revision

    try:
        model = AutoModelForMaskedLM.from_pretrained(model_id, **common_kwargs)
    except Exception:
        model = AutoModel.from_pretrained(model_id, **common_kwargs)
    return model


def trim_encoder_layers(
    model: torch.nn.Module, max_layer_index: int
) -> tuple[int, int, str]:
    base_model = model.base_model if hasattr(model, "base_model") else model
    layer_path, layers = _find_encoder_layers(base_model)

    if max_layer_index < 0:
        raise ValueError("max_layer_index must be >= 0")

    original_count = len(layers)
    target_count = max_layer_index + 1

    if target_count > original_count:
        raise ValueError(
            f"max_layer_index={max_layer_index} is out of range for model "
            f"with {original_count} layers"
        )

    del layers[target_count:]

    for key in ("num_hidden_layers", "num_layers", "n_layers"):
        if hasattr(base_model.config, key):
            setattr(base_model.config, key, target_count)

    return original_count, target_count, layer_path


def prepare_base_model_config(base_model: torch.nn.Module) -> None:
    architecture_name = base_model.__class__.__name__
    base_model.config.architectures = [architecture_name]

    if hasattr(base_model, "base_model_prefix"):
        base_model.config._name_or_path = getattr(
            base_model.config, "_name_or_path", ""
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a BERT-style model, keep layers [0..max_layer_index], "
            "drop LM head by saving only the base encoder, and save output."
        )
    )
    parser.add_argument(
        "--model-id", required=True, help="HF model id or local model path"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Where to save the prepared model"
    )
    parser.add_argument(
        "--max-layer-index",
        required=True,
        type=int,
        help="Last encoder layer index to keep (inclusive)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code when loading model/tokenizer",
    )
    parser.add_argument(
        "--revision", default=None, help="Optional model revision"
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the prepared model to the Hub",
    )
    parser.add_argument(
        "--hub-model-id",
        help="The Hub repo ID to push to (required if --push-to-hub is set)",
    )
    parser.add_argument(
        "--hub-token",
        help="The token to use to push to the Model Hub.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
    )
    model = _load_model(args.model_id, args.trust_remote_code, args.revision)

    original_count, target_count, layer_path = trim_encoder_layers(
        model, args.max_layer_index
    )

    # Save only the base encoder to ensure LM head is removed.
    base_model = model.base_model if hasattr(model, "base_model") else model
    prepare_base_model_config(base_model)
    base_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Saved prepared model to: {args.output_dir}")
    print(f"Layer path: {layer_path}")
    print(f"Encoder layers: {original_count} -> {target_count}")
    print(f"Architectures: {base_model.config.architectures}")

    if args.push_to_hub:
        if args.hub_model_id is None:
            raise ValueError(
                "--hub-model-id is required when --push-to-hub is set"
            )
        print(f"Pushing to Hub: {args.hub_model_id}")
        base_model.push_to_hub(args.hub_model_id, token=args.hub_token)
        tokenizer.push_to_hub(args.hub_model_id, token=args.hub_token)


if __name__ == "__main__":
    main()
