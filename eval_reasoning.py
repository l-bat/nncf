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
import multiprocessing
import os
import queue
import random
import re
from collections import Counter
from math import isclose
from typing import Union

import evaluate
import regex
import torch
from datasets import load_dataset
from latex2sympy2 import latex2sympy
from sympy import N
from sympy import simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from tqdm import tqdm
from tqdm import trange
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from nncf.quantization.advanced_parameters import KVCacheCompressionMode
from nncf.quantization.advanced_parameters import KVCacheCompressionParameters
from nncf.quantization.advanced_parameters import KVCacheRefinedSelection
from nncf.quantization.algorithms.kv_cache_management.torch_backend import KVCacheCompressor
from reasoning_parser import extract_answer
from reasoning_parser import strip_string

# disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
OUTPUT_LENGTHS = []

exact_match = evaluate.load("exact_match")


def choice_answer_clean(pred: str):
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]
    pred = pred[-1]
    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")
    return pred


def parse_digits(num):
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except ValueError:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except ValueError:
                pass
    return None


def is_digit(num):
    # paired with parse_digits
    return parse_digits(num) is not None


def str_to_pmatrix(input_str):
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []

    for m in matrix_str:
        m = m.strip("{}")
        pmatrix = r"\begin{pmatrix}" + m.replace(",", "\\") + r"\end{pmatrix}"
        pmatrix_list.append(pmatrix)

    return ", ".join(pmatrix_list)


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    # print("Judge:", prediction, reference)
    if prediction is None or reference is None:
        return False
    if str(prediction.strip().lower()) == str(reference.strip().lower()):
        return True
    if reference in ["A", "B", "C", "D", "E"] and choice_answer_clean(prediction) == reference:
        return True

    try:  # 1. numerical equal
        if is_digit(prediction) and is_digit(reference):
            prediction = parse_digits(prediction)
            reference = parse_digits(reference)
            # number questions
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if numeric_equal(prediction, item):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except Exception:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    # pmatrix (amps)
    if "pmatrix" in prediction and "pmatrix" not in reference:
        reference = str_to_pmatrix(reference)

    # deal with [], (), {}
    pred_str, ref_str = prediction, reference
    if (prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")) or (
        prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str.lower() == ref_str.lower():
        return True

    # [a, b] vs. [c, d], return a==c and b==d
    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts) and all(
            [math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close) for i in range(len(pred_parts))]
        ):
            return True
    if (
        (prediction.startswith("\\begin{pmatrix}") or prediction.startswith("\\begin{bmatrix}"))
        and (prediction.endswith("\\end{pmatrix}") or prediction.endswith("\\end{bmatrix}"))
        and (reference.startswith("\\begin{pmatrix}") or reference.startswith("\\begin{bmatrix}"))
        and (reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}"))
    ):
        pred_lines = [
            line.strip()
            for line in prediction[len("\\begin{pmatrix}") : -len("\\end{pmatrix}")].split("\\\\")
            if line.strip()
        ]
        ref_lines = [
            line.strip()
            for line in reference[len("\\begin{pmatrix}") : -len("\\end{pmatrix}")].split("\\\\")
            if line.strip()
        ]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) == len(ref_parts):
                    if not all(
                        [
                            math_equal(
                                pred_parts[i],
                                ref_parts[i],
                                include_percentage,
                                is_close,
                            )
                            for i in range(len(pred_parts))
                        ]
                    ):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            return True
    elif prediction.count("=") == 1 and len(prediction.split("=")[0].strip()) <= 2 and "=" not in reference:
        if math_equal(prediction.split("=")[1], reference, include_percentage, is_close):
            return True
    elif reference.count("=") == 1 and len(reference.split("=")[0].strip()) <= 2 and "=" not in prediction:
        if math_equal(prediction, reference.split("=")[1], include_percentage, is_close):
            return True

    # symbolic equal with sympy
    if timeout:
        if call_with_timeout(symbolic_equal_process, prediction, reference):
            return True
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False


def math_equal_process(param):
    return math_equal(param[-2], param[-1])


def numeric_equal(prediction: float, reference: float):
    # Note that relative tolerance has significant impact
    # on the result of the synthesized GSM-Hard dataset
    # if reference.is_integer():
    #     return isclose(reference, round(prediction), abs_tol=1e-4)
    # else:
    # prediction = round(prediction, len(str(reference).split(".")[-1]))
    return isclose(reference, prediction, rel_tol=1e-4)


def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\"))
            except Exception:
                try:
                    return f(s)
                except Exception:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    # direct equal
    try:
        if str(a) == str(b) or a == b:
            return True
    except Exception:
        pass

    # simplify equal
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except Exception:
        pass

    # equation equal
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except Exception:
        pass

    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except Exception:
        pass

    # matrix
    try:
        # if a and b are matrix
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except Exception:
        pass

    return False


def symbolic_equal_process(a, b, output_queue):
    result = symbolic_equal(a, b)
    output_queue.put(result)


def call_with_timeout(func, *args, timeout=1, **kwargs):
    output_queue = multiprocessing.Queue()
    process_args = args + (output_queue,)
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return False

    return output_queue.get()


def math_equal_with_timeout(pred, gt_ans, timeout):
    def target(result_queue):
        try:
            result_queue.put(math_equal(pred, gt_ans))
        except Exception as e:
            result_queue.put(e)

    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=target, args=(result_queue,))
    process.start()

    process.join(timeout)

    if process.is_alive():
        print(f"Timeout occurred for prediction: {pred}")
        process.terminate()
        process.join()
        return False

    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        print("Result queue timed out")
        return False

    if isinstance(result, Exception):
        print(f"Error occurred: {result}")
        return False

    return result


def parallel_math_equal(all_pred, gt_ans, timeout=20):
    results = []
    for pred in all_pred:
        results.append(math_equal_with_timeout(pred, gt_ans, timeout))
    return results


def eval_main(res_path, save=False, k=None, output_dir=None):
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


def extract_last_number(pred_str):
    o = re.sub(r"(\d),(\d)", r"\1\2", pred_str)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", o)
    if numbers:
        ans = numbers[-1]
    else:
        ans = None
    return ans


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
    eval_main(os.path.join(args.save_dir, "predictions.jsonl"), save=True, k=None, output_dir=args.save_dir)
