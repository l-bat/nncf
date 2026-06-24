#!/bin/bash

# Single checkpoint evaluation script
# Usage: ./eval_single_checkpoint.sh <checkpoint_path>
# Example: ./eval_single_checkpoint.sh output_llama_3b-instruct_1/nncf_init_fq_lora.pth

PRETRAINED="Qwen/Qwen3-4B"
# PRETRAINED="Qwen/Qwen3.6-35B-A3B"
# PRETRAINED="meta-llama/Llama-3.2-3B-Instruct"
TENSOR_PARALLEL_SIZE=2  # tp=1: no worker subprocesses, no IPC/shm leaks; Qwen/Qwen3.6-35B-A3B INT4 ~20GB fits on one A100 80GB

export VLLM_USE_FLASHINFER_SAMPLER=0  # "Qwen/Qwen3.6-35B-A3B" doesn't work with flashinfer sampler.
export VLLM_USE_V1=0                  # vLLM v1 shm_broadcast IPC crashes on this system; use v0 engine.
export VLLM_WORKER_MULTIPROC_METHOD=spawn  # prevent CUDA fork-safety issues in v0 worker processes
export NCCL_P2P_DISABLE=1             # NVLink/PCIe P2P peer access hangs with spawned workers on this system
export HF_HOME=/data/lt_hf_cache      # A100 HF dir
export HF_DATASETS_CACHE=/data/lt_hf_cache/datasets

# Parse arguments
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <checkpoint_path> [pretrained_model] [tensor_parallel_size]"
    exit 1
fi

CKPT_PATH="$1"
if [[ $# -ge 2 ]]; then
    PRETRAINED="$2"
fi
if [[ $# -ge 3 ]]; then
    TENSOR_PARALLEL_SIZE="$3"
fi

# Validate checkpoint exists
if [[ ! -f "$CKPT_PATH" ]]; then
    echo "Error: Checkpoint not found: $CKPT_PATH"
    exit 1
fi

# Get checkpoint info
CKPT_BASENAME=$(basename "$CKPT_PATH" .pth)
CKPT_DIR=$(dirname "$CKPT_PATH")
OUTPUT_DIR="$CKPT_DIR/stripped_$CKPT_BASENAME"
EVAL_DIR="$CKPT_DIR/eval_$CKPT_BASENAME"

echo "=========================================="
echo "Checkpoint: $CKPT_PATH"
echo "Pretrained: $PRETRAINED"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "Output directory: $OUTPUT_DIR"
echo "Eval directory: $EVAL_DIR"
echo "=========================================="

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$EVAL_DIR"

# Strip and save the model
echo "Step 1/5: Converting checkpoint to full model..."
python save_stripped.py -p "$PRETRAINED" -c "$CKPT_PATH" -o "$OUTPUT_DIR"
if [[ $? -ne 0 ]]; then
    echo "Error: save_stripped.py failed"
    exit 1
fi

# # Run GSM8K evaluation
# echo "Step 2/5: Running GSM8K evaluation..."
# lm_eval \
#     --model hf \
#     --model_args "pretrained=$OUTPUT_DIR" \
#     --tasks gsm8k \
#     --output_path "$EVAL_DIR" \
#     --apply_chat_template \
#     --batch_size 64


# Run GSM8K evaluation
echo "Step 2/4: Running GSM8K evaluation..."
lm_eval \
    --model vllm \
    --model_args "pretrained=$OUTPUT_DIR,dtype=auto,gpu_memory_utilization=0.8,tensor_parallel_size=$TENSOR_PARALLEL_SIZE,language_model_only=True,disable_custom_all_reduce=True" \
    --tasks gsm8k \
    --output_path "$EVAL_DIR" \
    --batch_size auto
    # --apply_chat_template \ # Disable for Qwen3 4B+ models


# # Run Lambada evaluation
echo "Step 3/4: Running Lambada evaluation..."
lm_eval \
    --model vllm \
    --model_args "pretrained=$OUTPUT_DIR,dtype=auto,gpu_memory_utilization=0.8,tensor_parallel_size=$TENSOR_PARALLEL_SIZE,language_model_only=True,disable_custom_all_reduce=True" \
    --tasks lambada_openai \
    --output_path "$EVAL_DIR" \
    --batch_size auto

# # Run MMLU evaluation
echo "Step 4/4: Running MMLU evaluation..."
lm_eval \
    --model vllm \
    --model_args "pretrained=$OUTPUT_DIR,dtype=auto,gpu_memory_utilization=0.8,tensor_parallel_size=$TENSOR_PARALLEL_SIZE,language_model_only=True,disable_custom_all_reduce=True" \
    --tasks mmlu \
    --output_path "$EVAL_DIR" \
    --batch_size auto

echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: $EVAL_DIR"
echo "=========================================="
