#!/bin/bash

# Output directories to process
OUTPUT_DIRS=("output_cosine")
PRETRAINED="meta-llama/Llama-3.2-3B-Instruct"
# PRETRAINED="Qwen/Qwen3-4B"

# Checkpoint files to evaluate
CKPT_FILES=(
    "last/nncf_checkpoint_epoch15.pth"
    "last/nncf_checkpoint_epoch10.pth"
    "last/nncf_checkpoint_epoch5.pth"
    "last/nncf_checkpoint_epoch2.pth"
    "last/nncf_checkpoint_epoch1.pth"
)
# CKPT_FILES=("nncf_checkpoint_after_first_epoch.pth") #"nncf_checkpoint_svd_lora_se_2bit.pth") #"nncf_checkpoint_svd_lora_se_tune_scales.pth")
LOG_FILE="lm_eval.log"

# ── Parse CLI arguments ─────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output_dirs) shift; OUTPUT_DIRS=(); while [[ $# -gt 0 && ! "$1" == --* ]]; do OUTPUT_DIRS+=("$1"); shift; done ;;
        --pretrained) PRETRAINED="$2"; shift 2 ;;
        --log_file)   LOG_FILE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--output_dirs <dir1> <dir2> ...] [--pretrained <model>] [--log_file <file>]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

run_lm_eval_gsm8k_sampling() {
    local model_dir="$1"
    local eval_output_path="$2"
    local log_file=$eval_output_path/$LOG_FILE
    echo "Running lm-eval on mmlu for $model_dir... Log file: $log_file"
    lm_eval \
        --model vllm \
        --model_args "{\"pretrained\":\"$model_dir\",\"enable_thinking\":false,\"dtype\":\"auto\",\"tensor_parallel_size\":2}" \
        --tasks gsm8k \
        --fewshot_as_multiturn \
        --apply_chat_template \
        --output_path "$eval_output_path" \
        --gen_kwargs do_sample=True,temperature=0.7,top_p=0.8,top_k=20,min_p=0 \
        --batch_size auto >> "$log_file" 2>&1
}

run_lm_eval_gsm8k_andrei() {
    local model_dir="$1"
    local eval_output_path="$2"
    local log_file=$eval_output_path/$LOG_FILE
    echo "Running lm-eval on mmlu for $model_dir... Log file: $log_file"
    lm_eval \
        --model vllm \
        --model_args "{\"pretrained\":\"$model_dir\",\"dtype\":\"auto\",\"tensor_parallel_size\":2}" \
        --tasks gsm8k \
        --output_path "$eval_output_path" \
        --batch_size auto >> "$log_file" 2>&1
}

run_lm_eval_lambada() {
    local model_dir="$1"
    local eval_output_path="$2"
    local log_file=$eval_output_path/$LOG_FILE
    echo "Running lm-eval on mmlu for $model_dir... Log file: $log_file"
    lm_eval \
        --model vllm \
        --model_args "{\"pretrained\":\"$model_dir\",\"dtype\":\"auto\",\"tensor_parallel_size\":2}" \
        --tasks lambada_openai \
        --output_path "$eval_output_path" \
        --batch_size auto >> "$log_file" 2>&1
}

run_lm_eval_mmlu() {
    local model_dir="$1"
    local eval_output_path="$2"
    local log_file=$eval_output_path/$LOG_FILE
    echo "Running lm-eval on mmlu for $model_dir... Log file: $log_file"
    lm_eval \
        --model vllm \
        --model_args "{\"pretrained\":\"$model_dir\",\"dtype\":\"auto\",\"tensor_parallel_size\":2}" \
        --tasks mmlu \
        --output_path "$eval_output_path" \
        --batch_size 1 >> "$log_file" 2>&1
}

echo "Started evaluation. Log file: $(realpath "$LOG_FILE")"
for OUTPUT_DIR in "${OUTPUT_DIRS[@]}"; do
    CKPT_DIR="$OUTPUT_DIR/last/stripped"

    # # Run training
    # echo "Running training with output_dir: $(realpath "$OUTPUT_DIR")"

    # python main.py --pretrained $PRETRAINED --fq_lr 1e-4 --fq_weight_decay 1e-4 --lora_weight_decay 1e-4 --lora_lr 1e-4 --cosine_epochs 5 --constant_epochs 0 --warmup_epochs 0 --min_lr_ratio 0.1 --resume --output_dir "$OUTPUT_DIR" >> "$LOG_FILE" 2>&1

    # First evaluation (after training, before stripping any checkpoint)
    # echo "Running lm-eval after training..."
    # run_lm_eval "$CKPT_DIR" "$LOG_FILE"

    # Loop through checkpoint files
    for CKPT_FILE in "${CKPT_FILES[@]}"; do
        # Derive a subfolder name from the checkpoint filename (without extension)
        CKPT_BASENAME=$(basename "$CKPT_FILE" .pth)
        CKPT_EVAL_DIR="$OUTPUT_DIR/eval_results/$CKPT_BASENAME"
        mkdir -p "$CKPT_EVAL_DIR"

        CKPT_PATH="$OUTPUT_DIR/$CKPT_FILE"
        if [[ ! -f "$CKPT_PATH" ]]; then
            echo "Skipping: checkpoint not found: $CKPT_PATH"
            continue
        fi

        echo "Saving stripped checkpoint: $CKPT_FILE to $CKPT_DIR"
        # python save_stripped.py -p $PRETRAINED -c "$OUTPUT_DIR/last/$CKPT_FILE" -o "$CKPT_DIR"
        python save_stripped.py -p $PRETRAINED -c "$CKPT_PATH" -o "$CKPT_DIR" || { echo "save_stripped.py failed for $CKPT_PATH, skipping eval"; continue; }

        run_lm_eval_gsm8k_andrei "$CKPT_DIR" "$CKPT_EVAL_DIR"
        run_lm_eval_lambada "$CKPT_DIR" "$CKPT_EVAL_DIR"
        # run_lm_eval_mmlu "$CKPT_DIR" "$CKPT_EVAL_DIR"
    done
done

echo "All experiments completed!"
