# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This logic is largely copied from the Hendrycks' MATH release (math_equivalence), and borrowed from:
- https://github.com/microsoft/ProphetNet/tree/master/CRITIC
- https://github.com/openai/prm800k
- https://github.com/microsoft/ToRA/blob/main/src/eval/grader.py
- https://github.com/deepseek-ai/DeepSeek-Math/blob/main/evaluation/eval/eval_utils.py
- https://github.com/VITA-Group/SEAL/tree/main
"""

import argparse
import json
import os
import random
import re
from collections import Counter

import torch
from datasets import load_dataset
from tqdm import tqdm
from tqdm import trange
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from nncf.quantization.advanced_parameters import KVCacheCompressionMode
from nncf.quantization.advanced_parameters import KVCacheCompressionParameters
from nncf.quantization.advanced_parameters import KVCacheRefinedSelection
from nncf.quantization.algorithms.kv_cache_management.torch_backend import KVCacheCompressor
from reasoning_parser import extract_answer
from reasoning_parser import parallel_math_equal
from reasoning_parser import strip_string

# disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
OUTPUT_LENGTHS = []


def run_evaluation(res_path, save=False, k=None, output_dir=None):
    with open(res_path) as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]

    for example in tqdm(data):
        if "model_generation" not in example:
            example["model_generation"] = example["model_output"]
        if k is not None:
            example["model_generation"] = example["model_generation"][:k]
        gt_cot = example["answer"]
        gt_ans = extract_answer(gt_cot, data_name="omni-math")
        gt_cot = str(gt_cot).strip()
        gt_ans = strip_string(gt_ans, skip_unit=False)
        all_pred = [extract_answer(p, data_name="omni-math") for p in example["model_generation"]]
        all_pred = [strip_string(p, skip_unit=False) for p in all_pred]
        all_eval = parallel_math_equal(all_pred, gt_ans, timeout=5)
        effective_pred = [p for p, o in zip(all_pred, example["model_generation"]) if "boxed" in o]
        if len(effective_pred) == 0:
            effective_pred = all_pred
        counter = Counter(effective_pred)
        pred = counter.most_common(1)[0][0]
        index = all_pred.index(pred)
        eval = all_eval[index]
        example["all_pred"] = all_pred
        example["all_eval"] = all_eval
        example["mv_pred"] = pred
        example["mv_eval"] = eval
        example["mv_index"] = index

    acc = sum([example["mv_eval"] for example in data]) / len(data)
    print(f"Accuracy: {acc:.3f}")

    correct_avg_len = []
    incorrect_avg_len = []

    for i, example in enumerate(data):
        if example["mv_eval"]:
            correct_avg_len.append(OUTPUT_LENGTHS[i])
        else:
            incorrect_avg_len.append(OUTPUT_LENGTHS[i])

    if len(correct_avg_len) != 0:
        print(f"Correct avg len: {sum(correct_avg_len) / len(correct_avg_len):.2f}", end=", ")
    if len(incorrect_avg_len) != 0:
        print(f"Incorrect avg len: {sum(incorrect_avg_len) / len(incorrect_avg_len):.2f}")

    if save:
        out_file = os.path.join(output_dir, "math_eval.jsonl")
        with open(out_file, "w") as f:
            for example in data:
                f.write(json.dumps(example) + "\n")

        metric_file = os.path.join(output_dir, "metrics.json")
        with open(metric_file, "w") as f:
            json.dump({"acc": acc}, f)


def trim_output(output):
    instruction_prefix = "Answer the following question"
    question_prefix = "Question:"
    comment_prefix = "Comment:"  # for some reason, Llama 13B likes to generate these comments indefinitely

    for prefix in [instruction_prefix, question_prefix, comment_prefix]:
        if prefix in output:
            output = output.split(prefix)[0]

    return output


def extract_box(pred_str):
    ans = pred_str.split("boxed")[-1]
    if len(ans) == 0:
        return ""
    elif ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()

    return a


def main(args):
    random.seed(42)

    print("Loading data...")
    test_data = []
    if args.dataset == "MATH500":
        data = load_dataset("HuggingFaceH4/MATH-500", split="test")
        for example in data:
            gt = extract_box(example["solution"])
            test_data.append(
                {
                    "question": example["problem"],
                    "answer": example["solution"],
                    "gt": gt,
                }
            )
    elif args.dataset == "GSM":
        data_path = "data/gsm/test.jsonl"
        with open(data_path) as fin:
            for line in fin:
                example = json.loads(line)
                answer = example["answer"].split("####")[1].strip()
                answer = re.sub(r"(\d),(\d)", r"\1\2", answer)
                test_data.append(
                    {
                        "question": example["question"],
                        "answer": example["answer"].split("####")[0].strip(),
                        "gt": answer,
                    }
                )
    else:
        error_message = f"Unknown dataset {args.dataset}"
        raise ValueError(error_message)

    if args.max_examples and len(test_data) > args.max_examples:
        test_data = test_data[: args.max_examples]

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer if args.tokenizer else args.model)

    # set padding side to left for batch generation
    tokenizer.padding_side = "left"

    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prefix = (
        "Answer the following questions. You should think step-by-step and put your final answer within \\boxed{}.\n"
    )
    prompts = []
    for i, example in enumerate(test_data):
        prompt = prefix + "Question: " + example["question"].strip() + "\nAnswer: "
        if args.use_chat_format:
            if "deepseek" in args.model:
                messages = [{"role": "user", "content": prefix + "Question: " + example["question"].strip()}]
            else:
                messages = [
                    {"role": "system", "content": prefix},
                    {"role": "user", "content": "Question: " + example["question"].strip()},
                ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if args.remove_bos and tokenizer.bos_token is not None and prompt.startswith(tokenizer.bos_token):
                prompt = prompt[len(tokenizer.bos_token) :]
        prompts.append(prompt)
    with open(os.path.join(args.save_dir, "example_prompt.txt"), "w") as fout:
        fout.write(prompts[0])

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", attn_implementation="eager")
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.eval()

    if args.enable_eviction:
        if args.algorithm == "snapkv":
            algorthm = KVCacheCompressionMode.SNAPKV
        elif args.algorithm == "rpc":
            algorthm = KVCacheCompressionMode.RPC
        elif args.algorithm == "rkv":
            algorthm = KVCacheCompressionMode.RKV
        else:
            algorthm = KVCacheCompressionMode.H2O
        refined_algorithm = None
        if args.refined_algorithm is not None:
            if args.refined_algorithm == "criticalkv":
                refined_algorithm = KVCacheRefinedSelection.CRITICALKV
            elif args.refined_algorithm == "diversekv":
                refined_algorithm = KVCacheRefinedSelection.DIVERSEKV
            else:
                refined_algorithm = KVCacheRefinedSelection.KVCRUSH

        eviction_parameters = KVCacheCompressionParameters(
            algorithm=algorthm,
            window_size=args.window_size,
            strategy=args.strategy,
            group_size=args.group_size,
            start_size=args.start_size,
            recent_size=args.recent_size,
            intermediate_size=args.intermediate_size,
            score_aggregation=args.score_aggregation,
            apply_rerotation=False,
            refined_size=args.refined_size,
            refined_algorithm=refined_algorithm,
            mix_lambda=args.mix_lambda,
            prefill_impl=args.prefill_impl,
        )
        compress = KVCacheCompressor(eviction_parameters=eviction_parameters)

    outputs = []
    prompts_with_eviction = 0
    avg_prompt_len = []
    for i in trange(0, len(prompts), args.batch_size):
        batch = prompts[i : i + args.batch_size]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
        tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}
        avg_prompt_len.append(tokenized_batch["input_ids"].shape[1])
        from contextlib import nullcontext

        with torch.no_grad(), compress(model) if args.enable_eviction else nullcontext():
            output = model.generate(
                **tokenized_batch,
                do_sample=False,
                max_new_tokens=args.max_tokens,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_len = tokenized_batch["input_ids"].shape[1]
        OUTPUT_LENGTHS.append(output.shape[1])
        if output.shape[1] > args.intermediate_size + args.recent_size + args.start_size:
            prompts_with_eviction += 1
        output = [tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in output]
        outputs.extend(output)

    outputs = [[trim_output(o)] for o in outputs]
    print(f"Average prompt length: {sum(avg_prompt_len) / len(avg_prompt_len):.2f}")
    print(f"Average length: {sum(OUTPUT_LENGTHS) / len(OUTPUT_LENGTHS):.2f}")
    print(f"Prompts with eviction: {prompts_with_eviction}/{len(OUTPUT_LENGTHS)}")

    predictions = [
        {
            "prompt": prompt,
            "problem": example["question"],
            "answer": example["gt"],
            "solution": example["answer"],
            "model_generation": output,
        }
        for example, output, prompt in zip(test_data, outputs, prompts)
    ]

    with open(os.path.join(args.save_dir, "predictions.jsonl"), "w") as fout:
        for prediction in predictions:
            fout.write(json.dumps(prediction) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="results/gsm")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--use_chat_format", action="store_true")
    parser.add_argument("--dataset", type=str, default="MATH500")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--remove_bos", action="store_true", default=True)

    parser.add_argument("--enable_eviction", action="store_true")
    parser.add_argument("--algorithm", default="snapkv", choices=["snapkv", "h2o", "rpc", "rkv"])
    parser.add_argument("--strategy", default="per_group", choices=["per_token", "per_group"])
    parser.add_argument("--refined_algorithm", default=None, choices=["criticalkv", "kvcrush", "diversekv"])
    parser.add_argument("--intermediate_size", type=int, default=512)
    parser.add_argument("--recent_size", type=int, default=128)
    parser.add_argument("--start_size", type=int, default=32)
    parser.add_argument("--refined_size", type=int, default=0)
    parser.add_argument(
        "--anchor", type=str, default="alternate", choices=["alternate", "mean", "zeros", "ones", "random"]
    )
    parser.add_argument("--score_aggregation", type=str, default="sum", choices=["sum", "norm_sum"])
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--mix_lambda", type=float, default=1.0)
    parser.add_argument("--prefill_impl", default="dense", choices=["dense", "tri-shape", "x-attention"])

    args = parser.parse_args()

    args.save_dir = os.path.join(args.save_dir, "base")

    if args.remove_bos:
        args.save_dir = args.save_dir + "_remove_bos"

    if args.max_examples or args.start:
        start = 0 if args.start is None else args.start
        end = start + args.max_examples if args.max_examples is not None else -1
        args.save_dir = os.path.join(args.save_dir, f"{start}_{end}")

    print(args.save_dir)
    main(args)
    run_evaluation(os.path.join(args.save_dir, "predictions.jsonl"), save=True, k=None, output_dir=args.save_dir)
