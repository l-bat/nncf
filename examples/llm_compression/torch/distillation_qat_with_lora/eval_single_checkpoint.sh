#!/bin/bash

# Single checkpoint evaluation script
# Usage: ./eval_single_checkpoint.sh <checkpoint_path>
# Example: ./eval_single_checkpoint.sh output_llama_3b-instruct_1/nncf_init_fq_lora.pth

# PRETRAINED="Qwen/Qwen3-1.7B"
PRETRAINED="google/gemma-3-4b-it"
# PRETRAINED="meta-llama/Llama-3.2-3B-Instruct"
TENSOR_PARALLEL_SIZE=2

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
echo "Step 2/5: Running GSM8K evaluation..."
    # --model_args "pretrained=$OUTPUT_DIR,dtype=auto,device_map=auto" \
lm_eval \
    --model hf \
    --model_args "pretrained=$OUTPUT_DIR" \
    --tasks gsm8k \
    --output_path "$EVAL_DIR" \
    --apply_chat_template \
    --batch_size 64

# # Run Lambada evaluation
# echo "Step 3/5: Running Lambada evaluation..."
lm_eval \
    --model hf \
    --model_args "pretrained=$OUTPUT_DIR" \
    --tasks lambada_openai \
    --output_path "$EVAL_DIR" \
    --batch_size 64

# Run MMLU evaluation
echo "Step 5/5: Running MMLU evaluation..."
lm_eval \
    --model hf \
    --model_args "pretrained=$OUTPUT_DIR" \
    --tasks mmlu \
    --output_path "$EVAL_DIR" \
    --batch_size 32


# # Run GSM8K evaluation
# echo "Step 2/5: Running GSM8K evaluation..."
#     # --model_args "pretrained=$OUTPUT_DIR,dtype=auto,device_map=auto" \
# lm_eval \
#     --model vllm \
#     --model_args "pretrained=$OUTPUT_DIR,dtype=auto,tensor_parallel_size=$TENSOR_PARALLEL_SIZE" \
#     --tasks gsm8k \
#     --output_path "$EVAL_DIR" \
#     --batch_size auto
#     # --apply_chat_template \

# # # Run Lambada evaluation
# # echo "Step 3/5: Running Lambada evaluation..."
# lm_eval \
#     --model vllm \
#     --model_args "pretrained=$OUTPUT_DIR,dtype=auto,tensor_parallel_size=$TENSOR_PARALLEL_SIZE" \
#     --tasks lambada_openai \
#     --output_path "$EVAL_DIR" \
#     --batch_size auto

# # # Run CommonsenseQA evaluation
# # echo "Step 4/5: Running CommonsenseQA evaluation..."
# # lm_eval \
# #     --model vllm \
# #     --model_args "pretrained=$OUTPUT_DIR" \
# #     --tasks commonsense_qa \
# #     --output_path "$EVAL_DIR" \
# #     --batch_size auto

# # # Run MMLU evaluation
# echo "Step 5/5: Running MMLU evaluation..."
# lm_eval \
#     --model vllm \
#     --model_args "pretrained=$OUTPUT_DIR,dtype=auto,tensor_parallel_size=$TENSOR_PARALLEL_SIZE" \
#     --tasks mmlu \
#     --output_path "$EVAL_DIR" \
#     --batch_size auto

# echo "=========================================="
# echo "Evaluation complete!"
# echo "Results saved to: $EVAL_DIR"
# echo "=========================================="


# #!/bin/bash

# # Single checkpoint evaluation script
# # Usage: ./eval_single_checkpoint.sh <checkpoint_path>
# # Example: ./eval_single_checkpoint.sh output_llama_3b-instruct_1/nncf_init_fq_lora.pth

# PRETRAINED="Qwen/Qwen3.5-4B"
# # PRETRAINED="meta-llama/Llama-3.2-3B-Instruct"
# TENSOR_PARALLEL_SIZE=2

# # Parse arguments
# if [[ $# -eq 0 ]]; then
#     echo "Usage: $0 <checkpoint_path> [pretrained_model] [tensor_parallel_size]"
#     exit 1
# fi

# CKPT_PATH="$1"
# if [[ $# -ge 2 ]]; then
#     PRETRAINED="$2"
# fi
# if [[ $# -ge 3 ]]; then
#     TENSOR_PARALLEL_SIZE="$3"
# fi

# # Validate checkpoint exists
# if [[ ! -f "$CKPT_PATH" ]]; then
#     echo "Error: Checkpoint not found: $CKPT_PATH"
#     exit 1
# fi

# # Get checkpoint info
# CKPT_BASENAME=$(basename "$CKPT_PATH" .pth)
# CKPT_DIR=$(dirname "$CKPT_PATH")
# OUTPUT_DIR="$CKPT_DIR/stripped_$CKPT_BASENAME"
# EVAL_DIR="$CKPT_DIR/eval_$CKPT_BASENAME"

# echo "=========================================="
# echo "Checkpoint: $CKPT_PATH"
# echo "Pretrained: $PRETRAINED"
# echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
# echo "Output directory: $OUTPUT_DIR"
# echo "Eval directory: $EVAL_DIR"
# echo "=========================================="

# # Create directories
# mkdir -p "$OUTPUT_DIR"
# mkdir -p "$EVAL_DIR"

# # Strip and save the model
# echo "Step 1/5: Converting checkpoint to full model..."
# python save_stripped.py -p "$PRETRAINED" -c "$CKPT_PATH" -o "$OUTPUT_DIR"
# if [[ $? -ne 0 ]]; then
#     echo "Error: save_stripped.py failed"
#     exit 1
# fi

# # Run GSM8K evaluation
# echo "Step 2/5: Running GSM8K evaluation..."

#     # --model_args "pretrained=$OUTPUT_DIR,dtype=auto,tensor_parallel_size=$TENSOR_PARALLEL_SIZE,enable_thinking=False" \
# lm_eval \
#     --model hf \
#     --model_args "pretrained=$OUTPUT_DIR,enable_thinking=False" \
#     --tasks gsm8k \
#     --apply_chat_template \
#     --num_fewshot 5 \
#     --output_path "$EVAL_DIR" \
#     --batch_size auto

# # Run Lambada evaluation
# echo "Step 3/5: Running Lambada evaluation..."
#     # --model_args "pretrained=$OUTPUT_DIR,dtype=auto,tensor_parallel_size=$TENSOR_PARALLEL_SIZE,trust_remote_code=True" \
# lm_eval \
#     --model hf \
#     --model_args "pretrained=$OUTPUT_DIR" \
#     --tasks lambada_openai \
#     --output_path "$EVAL_DIR" \
#     --batch_size auto

# # Run CommonsenseQA evaluation
# echo "Step 4/5: Running CommonsenseQA evaluation..."
#     # --model_args "pretrained=$OUTPUT_DIR,dtype=auto,tensor_parallel_size=$TENSOR_PARALLEL_SIZE,trust_remote_code=True" \
# lm_eval \
#     --model hf \
#     --model_args "pretrained=$OUTPUT_DIR" \
#     --tasks commonsense_qa \
#     --output_path "$EVAL_DIR" \
#     --batch_size auto

# # Run MMLU evaluation
# echo "Step 5/5: Running MMLU evaluation..."
#     # --model_args "pretrained=$OUTPUT_DIR,dtype=auto,tensor_parallel_size=$TENSOR_PARALLEL_SIZE,trust_remote_code=True" \
# lm_eval \
#     --model hf \
#     --model_args "pretrained=$OUTPUT_DIR" \
#     --tasks mmlu \
#     --output_path "$EVAL_DIR" \
#     --batch_size auto

# echo "=========================================="
# echo "Evaluation complete!"
# echo "Results saved to: $EVAL_DIR"
# echo "=========================================="
