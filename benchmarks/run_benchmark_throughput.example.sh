#!/bin/bash
# This script shows how to run the throughput benchmark with vLLM.
MODEL_ID="meta-llama/Llama-3.2-1B"

# optimum-rbln path
OPTIMUM_RBLN_MODEL_PATH="./.optimum-rbln-cache"
python3 compile_optimum_rbln.py $MODEL_ID --output-dir $OPTIMUM_RBLN_MODEL_PATH \
    --max-num-seqs 16 \
    --max-model-len 8192 \
    --block-size 4096 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 128 \
    --attn-impl flash_attn

VLLM_LOGGING_LEVEL=warning RBLN_PROFILER=0 USE_VLLM_MODEL=0 VLLM_USE_V1=0 \
python3 benchmark_throughput.py \
    --model $OPTIMUM_RBLN_MODEL_PATH \
    --backend vllm \
    --dataset-name random --input-len 1024 --output-len 1024 \
    --num-prompts 100 \
    --max-model-len 8192 \
    --block-size 4096 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 128 \
    --warmup-requests 1 \
    --output-json opt-rbln-results.json

rm -rf $OPTIMUM_RBLN_MODEL_PATH

# torch.compile path
VLLM_LOGGING_LEVEL=warning RBLN_PROFILER=0 RBLN_KERNEL_MODE=triton USE_VLLM_MODEL=1 VLLM_DISABLE_COMPILE_CACHE=1 VLLM_USE_V1=0 \
python3 benchmark_throughput.py \
    --model meta-llama/Llama-3.2-1B-instruct \
    --backend vllm \
    --dataset-name random --input-len 1024 --output-len 124 \
    --num-prompts 100 \
    --max-num-seqs 16 \
    --max-model-len 8192 \
    --block-size 4096 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 256 \
    --warmup-requests 1 \
    --random-range-ratio 0.1 \
    --async-engine \
    --output-json torch-compile-results-async-v0.json


VLLM_LOGGING_LEVEL=warning RBLN_PROFILER=0 RBLN_KERNEL_MODE=triton USE_VLLM_MODEL=1 VLLM_DISABLE_COMPILE_CACHE=1 VLLM_USE_V1=1 \
 vllm serve meta-llama/Llama-3.2-1B-instruct --max-model-len 8192 --max-num-seqs 16 --block-size 4096 --enable-chunked-prefill --max-num-batched-tokens 256


python benchmark_serving_structured_output.py \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --model meta-llama/Llama-3.2-1B-instruct  \
    --dataset xgrammar_bench \
    --output-len 640 \
    --save-results \
    --result-filename v1_serving-results-chat-completions.json \
    --num-prompts 100 \
    --max-concurrency 16


curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.2-1B-instruct",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "max_tokens": 100
    }'