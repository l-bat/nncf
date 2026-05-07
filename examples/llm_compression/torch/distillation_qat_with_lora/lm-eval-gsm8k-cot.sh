####################################   RedHatAI/Qwen2.5-3B-quantized.w4a16    ####################################
####################################        vLLM                              ####################################
# 11 + 51 sec (22200 toks/s)
# lm-eval \
# --model vllm \
# --model_args pretrained="RedHatAI/Qwen2.5-3B-quantized.w4a16",dtype=auto,max_model_len=4096,max_gen_toks=1024,tensor_parallel_size=1 \
# --tasks gsm8k_cot \
# --fewshot_as_multiturn \
# --apply_chat_template \
# --num_fewshot 8 \
# --batch_size auto
# fm - 0.7149
# sm - 0.5883
####################################        HF                                 ####################################
# > 8 hours
# --tensor_parallel_size=1 \ #vLLM specific
# lm-eval \
# --model hf \
# --model_args pretrained="RedHatAI/Qwen2.5-3B-quantized.w4a16",dtype=auto,max_length=4096 \
# --gen_kwargs="max_gen_toks=1024" \
# --tasks gsm8k_cot \
# --fewshot_as_multiturn \
# --apply_chat_template \
# --num_fewshot 8 \
# --batch_size auto

####################################   Qwen/Qwen2.5-3B    ####################################
####################################        vLLM          ####################################
# 11 + 45 sec (25100 toks/s)
# fm - 0.7672
# sm - 0.6611
# lm-eval \
# --model vllm \
# --model_args pretrained="Qwen/Qwen2.5-3B",dtype=auto,max_model_len=4096,max_gen_toks=1024,tensor_parallel_size=1 \
# --tasks gsm8k_cot \
# --fewshot_as_multiturn \
# --apply_chat_template \
# --num_fewshot 8 \
# --batch_size auto
####################################         HF           ####################################
# ~50min
# --tensor_parallel_size=1 \ #vLLM specific
# lm-eval \
# --model hf \
# --model_args pretrained="Qwen/Qwen2.5-3B",dtype=auto,max_length=4096 \
# --gen_kwargs="max_gen_toks=1024" \
# --tasks gsm8k_cot \
# --fewshot_as_multiturn \
# --apply_chat_template \
# --num_fewshot 8 \
# --batch_size auto


####################################   STiFLeR7/Qwen2.5-3B-GPTQ    ####################################
####################################        vLLM                   ####################################
# 11 + 53 sec (25600 toks/s)
# fm - 0.6846
# sm - 0.5754
# [gpu_model_runner.py:3259] Starting to load model STiFLeR7/Qwen2.5-3B-GPTQ...
# [gptq_marlin.py:359] Using MarlinLinearKernel for GPTQMarlinLinearMethod
# lm-eval \
# --model vllm \
# --model_args pretrained="STiFLeR7/Qwen2.5-3B-GPTQ",dtype=auto,max_model_len=4096,max_gen_toks=1024,tensor_parallel_size=1 \
# --tasks gsm8k_cot \
# --fewshot_as_multiturn \
# --apply_chat_template \
# --num_fewshot 8 \
# --batch_size auto
####################################         HF           ####################################
# --tensor_parallel_size=1 \ #vLLM specific
# ~50min
# requires gptq-model or auto-gptq
# lm-eval \
# --model hf \
# --model_args pretrained="STiFLeR7/Qwen2.5-3B-GPTQ",dtype=auto,max_length=4096 \
# --gen_kwargs="max_gen_toks=1024" \
# --tasks gsm8k_cot \
# --fewshot_as_multiturn \
# --apply_chat_template \
# --num_fewshot 8 \
# --batch_size auto


####################################   osllmai-community/Qwen2.5-3B-bnb-4bit    ####################################
####################################        bnb                                 ####################################

# bitsandbytes - nf4
# --apply_chat_template \
# --num_fewshot 8 \
# --fewshot_as_multiturn \
# 11 + 74 sec (13500 toks/s)
# fm - 0.7263
# sm - 0.5754
# lm-eval
# --model vllm \
# --model_args pretrained="osllmai-community/Qwen2.5-3B-bnb-4bit",dtype=auto,max_model_len=4096,max_gen_toks=1024,tensor_parallel_size=1 \
# --tasks gsm8k_cot \
# --batch_size auto

####################################   Qwen/Qwen2.5-3B-OpenVINO    ####################################
####################################         HF           ####################################
# optimum-cli export openvino -m=Qwen/Qwen2.5-3B --weight-format=int4 ./int4_default
# There's an error because of --batch_size=auto
# Caught exception: Exception from src/plugins/intel_cpu/src/node.cpp:820:
# [CPU] FullyConnectedCompressed node with name '__module.model.layers.0.mlp.down_proj/ov_ext::linear/MatMul' could not create a primitive descriptor for the inner product forward propagation primitive.
# optimum-cli export openvino -m=Qwen/Qwen2.5-3B --weight-format=int4 ./int4_default --task text-generation-with-past
# --tensor_parallel_size=1 \ #vLLM specific
# lm-eval \
# --model openvino \
# --model_args pretrained=./int4_default,dtype=auto,max_length=4096 \
# --gen_kwargs="max_gen_toks=1024" \
# --tasks gsm8k_cot \
# --fewshot_as_multiturn \
# --apply_chat_template \
# --num_fewshot 8 \
# --batch_size auto
####################################         HF           ####################################
# ~3 hours
# lm-eval \
# --model openvino \
# --model_args pretrained=./int4_default,dtype=auto,max_length=4096 \
# --gen_kwargs="max_gen_toks=1024" \
# --apply_chat_template \
# --fewshot_as_multiturn \
# --num_fewshot 8 \
# --tasks gsm8k_cot \
# --batch_size auto

####################################   Qwen/Qwen2.5-3B-Torch    ####################################
####################################         HF           ####################################
# bf16 - ~1.3h
# stripped directly - ~5h
# stripped w/ save-load - ~1.3h
# not stripped - ~10h
# https://jira.devtools.intel.com/secure/attachment/5783546/5783546_check_strip.py

# fm 0.4564
# sm 0.2335
# lm-eval \
# --model vllm \
# --model_args pretrained="kaitchup/Qwen3-8B-autoround-2bit-gptq",dtype=auto,max_model_len=4096,max_gen_toks=1024,tensor_parallel_size=1 \
# --tasks gsm8k_cot \
# --fewshot_as_multiturn \
# --apply_chat_template \
# --num_fewshot 8 \
# --batch_size auto

ROOT_DIR="/home/nlyaly/projects/nncf/examples/llm_compression/torch/distillation_qat_with_lora"
OUT_DIR="$ROOT_DIR/output"

# sm=0.2055
# fm=0.1221
# 1 epoch, num_samples=512, seq_len=512, lora_rank=64, nbs=1, bs=8
# CKPT_DIR="$OUT_DIR/last/stripped"
# lm_eval \
# --model vllm \
# --model_args pretrained=$CKPT_DIR,dtype=auto,tensor_parallel_size=2 \
# --tasks gsm8k \
# --fewshot_as_multiturn \
# --apply_chat_template \
# --batch_size auto

# sm=0.0000
# fm=0.0045
# 0 epoch, lora_rank=634
# CKPT_DIR="$OUT_DIR/initial_rank64/stripped"
# lm_eval \
# --model vllm \
# --model_args pretrained=$CKPT_DIR,dtype=auto,tensor_parallel_size=2 \
# --tasks gsm8k \
# --fewshot_as_multiturn \
# --apply_chat_template \
# --batch_size auto

# sm=0.2745
# fm=0.1820
# 10 epoch, num_samples=512, seq_len=512, lora_rank=64, nbs=1, bs=8, gs=64
# CKPT_DIR="$OUT_DIR/last/stripped"
# lm_eval \
# --model vllm \
# --model_args pretrained=$CKPT_DIR,dtype=auto,tensor_parallel_size=2 \
# --tasks gsm8k \
# --fewshot_as_multiturn \
# --apply_chat_template \
# --batch_size auto

# sm=0.4549
# fm=0.4632
# 4 epoch, num_samples=512, seq_len=512, lora_rank=64, nbs=1, bs=8, gs=32
# CKPT_DIR="$OUT_DIR/last/stripped_4epoch"
# lm_eval \
# --model vllm \
# --model_args pretrained=$CKPT_DIR,dtype=auto,tensor_parallel_size=2 \
# --tasks gsm8k \
# --fewshot_as_multiturn \
# --apply_chat_template \
# --batch_size auto

# fm=0.3707
# sm=0.3632
# SE, lora_rank=64, gs=32
# CKPT_DIR="$OUT_DIR/initial_rank64_gs32/stripped"
# lm_eval \
# --model vllm \
# --model_args pretrained=$CKPT_DIR,dtype=auto,tensor_parallel_size=2 \
# --tasks gsm8k \
# --fewshot_as_multiturn \
# --apply_chat_template \
# --batch_size auto

# fm=0.0265
# sm=0.0061
# SE, lora_rank=64, gs=32 (SE2)
# CKPT_DIR="$OUT_DIR/initial_rank64_gs32_SE2/stripped"
# lm_eval \
# --model vllm \
# --model_args pretrained=$CKPT_DIR,dtype=auto,tensor_parallel_size=2 \
# --tasks gsm8k \
# --fewshot_as_multiturn \
# --apply_chat_template \
# --batch_size auto


# fm=0.4579
# sm=0.4481
# https://huggingface.co/kaitchup/Qwen3-8B-autoround-2bit-gptq
# CKPT_DIR=kaitchup/Qwen3-8B-autoround-2bit-gptq
# lm_eval \
# --model vllm \
# --model_args pretrained=$CKPT_DIR,dtype=auto,tensor_parallel_size=2 \
# --tasks gsm8k \
# --fewshot_as_multiturn \
# --apply_chat_template \
# --batch_size auto



# fm=0.5004
# sm=0.4936
# 0 epoch, num_samples=512, seq_len=512, lora_rank=64, nbs=1, bs=8, gs=32, mostly int2
# CKPT_DIR="$OUT_DIR/last/stripped_0epoch"
# lm_eval \
# --model vllm \
# --model_args pretrained=$CKPT_DIR,dtype=auto,tensor_parallel_size=2 \
# --tasks gsm8k \
# --fewshot_as_multiturn \
# --apply_chat_template \
# --batch_size auto

# fm=0.5027
# sm=0.4966
# 1 epoch, num_samples=512, seq_len=512, lora_rank=64, nbs=1, bs=8, gs=32, mostly int2
# CKPT_DIR="$OUT_DIR/last/stripped_1epoch"
# lm_eval \
# --model vllm \
# --model_args pretrained=$CKPT_DIR,dtype=auto,tensor_parallel_size=2 \
# --tasks gsm8k \
# --fewshot_as_multiturn \
# --apply_chat_template \
# --batch_size auto

# fm=0.2199
# sm=0.3306
# 15 epoch, num_samples=512, seq_len=512, lora_rank=64, nbs=1, bs=8, gs=64/128, avg 3 (ar config)
# OUT_DIR="$ROOT_DIR/output_avg3_ar"
# CKPT_DIR="$OUT_DIR/last/stripped"
# lm_eval \
# --model vllm \
# --model_args pretrained=$CKPT_DIR,dtype=auto,tensor_parallel_size=2 \
# --tasks gsm8k \
# --fewshot_as_multiturn \
# --apply_chat_template \
# --batch_size auto


# fm=0.2214
# fm=0.1296
# lora_rank=64, gs=64/128, avg 3 (ar config)
# OUT_DIR="$ROOT_DIR/output_avg3_ar"
# CKPT_DIR="$OUT_DIR/last/stripped_init"
# lm_eval \
# --model vllm \
# --model_args pretrained=$CKPT_DIR,dtype=auto,tensor_parallel_size=2 \
# --tasks gsm8k \
# --fewshot_as_multiturn \
# --apply_chat_template \
# --batch_size auto

# fm=0.3146
# sm=0.2843
# 1 epoch, num_samples=512, seq_len=512, lora_rank=64, nbs=1, bs=8, gs=64/128, avg 3 (ar config)
# OUT_DIR="$ROOT_DIR/output_avg3_ar"
# CKPT_DIR="$OUT_DIR/last/stripped_after_first_epoch"
# lm_eval \
# --model vllm \
# --model_args pretrained=$CKPT_DIR,dtype=auto,tensor_parallel_size=2 \
# --tasks gsm8k \
# --fewshot_as_multiturn \
# --apply_chat_template \
# --batch_size auto

# fm=0.2153
# sm=0.0121
# 0 epoch, lora_rank=64, gs=32, avg 3 (ar config)
# OUT_DIR="$ROOT_DIR/output_avg3_ar"
# CKPT_DIR="$OUT_DIR/last/stripped"
# lm_eval \
# --model vllm \
# --model_args pretrained=$CKPT_DIR,dtype=auto,tensor_parallel_size=2 \
# --tasks gsm8k \
# --fewshot_as_multiturn \
# --apply_chat_template \
# --batch_size auto


# fm=0.2646
# sm=0.0417
# 2 epoch, num_samples=512, seq_len=512, lora_rank=64, nbs=1, bs=8, gs=32, avg 3 (ar config)
# OUT_DIR="$ROOT_DIR/output_avg3_ar"
# CKPT_DIR="$OUT_DIR/last/stripped"
# lm_eval \
# --model vllm \
# --model_args pretrained=$CKPT_DIR,dtype=auto,tensor_parallel_size=2 \
# --tasks gsm8k \
# --fewshot_as_multiturn \
# --apply_chat_template \
# --batch_size auto


# fm=0.5504
# sm=0.5383
# 2 epoch, num_samples=512, seq_len=512, lora_rank=64, nbs=1, bs=8, gs=32, sym, mostly int2
# CKPT_DIR="$OUT_DIR/last/stripped"
# lm_eval \
# --model vllm \
# --model_args pretrained=$CKPT_DIR,dtype=auto,tensor_parallel_size=2 \
# --tasks gsm8k \
# --fewshot_as_multiturn \
# --apply_chat_template \
# --batch_size auto

# fm=0.533
# sm=0.533
# 1 epoch, num_samples=512, seq_len=512, lora_rank=64, nbs=1, bs=8, gs=32, sym, mostly int2
# CKPT_DIR="$OUT_DIR/last/stripped_after_first_epoch"
# lm_eval \
# --model vllm \
# --model_args pretrained=$CKPT_DIR,dtype=auto,tensor_parallel_size=2 \
# --tasks gsm8k \
# --fewshot_as_multiturn \
# --apply_chat_template \
# --batch_size auto


# fm=0.3207
# sm=0.2229
# 2 epoch, num_samples=512, seq_len=512, lora_rank=64, nbs=1, bs=8, gs=64/128, sym, mostly int2
# fm=0.3692
# sm=0.3359
# 1 epoch
# fm=0.0167
# sm=0.0008
# 0 epoch
# CKPT_DIR="$OUT_DIR/last/stripped"
# lm_eval \
# --model vllm \
# --model_args pretrained=$CKPT_DIR,dtype=auto,tensor_parallel_size=2 \
# --tasks gsm8k \
# --fewshot_as_multiturn \
# --apply_chat_template \
# --batch_size auto

#### 20 epoch, num_samples=512, seq_len=512, lora_rank=64, nbs=1, bs=8, gs=64/128, sym, all down/v-proj
# fm=0.4549
# sm=0.3760
#### 1 epoch
# fm=0.4284
# sm=0.3677
#### 0 epoch
# fm=0.0743
# sm=0.0758
# CKPT_DIR="$OUT_DIR/last/stripped"
# lm_eval \
# --model vllm \
# --model_args pretrained=$CKPT_DIR,dtype=auto,tensor_parallel_size=2 \
# --tasks gsm8k \
# --fewshot_as_multiturn \
# --apply_chat_template \
# --batch_size auto

#### 2 epoch, num_samples=512, seq_len=512, lora_rank=64, nbs=1, bs=8, gs=128, sym, all int4
# fm=0.1948
# sm=0.0136
#### 1 epoch
# fm=0.2062
# sm=0.0121
#### 0 epoch
# fm=0.1842
# sm=0.0114



# --fewshot_as_multiturn \
# --apply_chat_template \
# --model_args "{\"pretrained\":\"${MODEL_DIR}\",\"enable_thinking\":false,\"dtype\":\"auto\",\"tensor_parallel_size\":2}" \
# --model_args pretrained=Qwen/Qwen3-8B,dtype=auto,tensor_parallel_size=2 \

# MODEL_DIR=Qwen/Qwen3-8B
# MODEL_DIR=output/last/stripped
# lm_eval \
# --model vllm \
# --model_args "{\"pretrained\":\"${MODEL_DIR}\",\"enable_thinking\":false,\"dtype\":\"auto\",\"tensor_parallel_size\":2}" \
# --tasks gsm8k \
# --batch_size auto \
# --fewshot_as_multiturn \
# --apply_chat_template \
# --gen_kwargs do_sample=True,temperature=0.7,top_p=0.8,top_k=20,min_p=0

# MODEL_DIR=meta-llama/Llama-3.2-3B-Instruct
# MODEL_DIR=Qwen/Qwen3-4B
# MODEL_DIR=output_1/nncf_init_fq_lora.pth
# MODEL_DIR=output_1/last/stripped
# MODEL_DIR=output_cosine_1/last/stripped
# --model_args "{\"pretrained\":\"${MODEL_DIR}\",\"dtype\":\"auto\",\"tensor_parallel_size\":1}" \
lm_eval \
--model vllm \
--model_args "pretrained=${MODEL_DIR},dtype=auto,tensor_parallel_size=1" \
--tasks gsm8k \
--batch_size auto

lm_eval \
--model vllm \
--model_args "pretrained=${MODEL_DIR},dtype=auto,tensor_parallel_size=1" \
--tasks lambada_openai \
--batch_size auto