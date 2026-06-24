# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
from transformers import AutoTokenizer

import nncf
from nncf.torch.function_hook.wrapper import get_hook_storage
from nncf.torch.model_creation import load_from_config
from nncf.torch.quantization.layers import AsymmetricLoraQuantizer  # noqa: F401


def load_checkpoint(model: nn.Module, ckpt_file: Path) -> nn.Module:
    """
    Loads the state of a tuned model from a checkpoint. This function restores the placement of Fake Quantizers (FQs)
    with absorbable LoRA adapters and loads their parameters.

    :param model: The model to load the checkpoint into.
    :param ckpt_file: Path to the checkpoint file.
    :returns: The model with the loaded NNCF state from checkpoint.
    """
    ckpt = torch.load(ckpt_file, weights_only=False, map_location="cpu")
    model = load_from_config(model, ckpt["nncf_config"])
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    hook_storage = get_hook_storage(model)
    hook_storage.load_state_dict(ckpt["nncf_state_dict"])
    return model


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)  # noqa: F821
    parser.add_argument(
        "--pretrained",
        "-p",
        type=str,
        default="Qwen/Qwen3-8B",
        help="The model id or path of a pretrained HF model configuration.",
    )
    # Model params
    parser.add_argument(
        "-c",
        "--ckpt_file",
        default=None,
        type=str,
        help="Single checkpoint file to process",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="stripped",
        type=str,
        help="output directory",
    )
    return parser


def main(argv) -> float:
    """
    Fine-tunes the specified model and returns the difference between initial and best validation perplexity in Torch,
    and the test perplexity for best model exported to OpenVINO.
    """
    parser = get_argument_parser()
    args = parser.parse_args(argv)

    ckpt_file = Path(args.ckpt_file)
    if not ckpt_file.exists():
        msg = f"not found checkpoint: {ckpt_file}"
        raise FileNotFoundError(msg)
    print(f"Processing checkpoint: {ckpt_file}")
    save_dir = Path(args.output_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    # Use AutoProcessor for multimodal models (e.g. Gemma3) so that
    # preprocessor_config.json is included in the saved directory.
    # Fall back to AutoTokenizer for text-only models.
    try:
        processor = AutoProcessor.from_pretrained(args.pretrained)
        processor.save_pretrained(save_dir)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
        tokenizer.save_pretrained(save_dir)

    model = AutoModelForCausalLM.from_pretrained(args.pretrained, device_map="cpu")
    model = load_checkpoint(model, ckpt_file)
    model = nncf.strip(model, strip_format=nncf.StripFormat.IN_PLACE)
    model.save_pretrained(save_dir)
    # For multimodal models (e.g. Qwen3.6-35B-A3B), AutoModelForCausalLM saves only
    # the text sub-config (Qwen3_5MoeTextConfig) which vLLM rejects with TypeError.
    # Overwrite config.json with the full top-level config (Qwen3_5MoeConfig) so
    # vLLM can load the directory.  Use language_model_only=True in eval model_args
    # to skip loading vision-encoder weights (only text weights are present here).
    orig_config = AutoConfig.from_pretrained(args.pretrained)
    if type(orig_config) is not type(model.config):
        orig_config.save_pretrained(save_dir)
        print(
            f"Overwrote text config ({type(model.config).__name__}) with full model "
            f"config ({type(orig_config).__name__}) for vLLM compatibility."
        )
    print(f"Saved stripped model to: {save_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
