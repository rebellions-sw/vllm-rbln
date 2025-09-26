VLLM_USE_V1=1 VLLM_RBLN_ENABLE_WARM_UP=0 VLLM_RBLN_SAMPLER=0 python run_decoder_only_simple.py # original (cpu)
VLLM_USE_V1=1 VLLM_RBLN_ENABLE_WARM_UP=0 VLLM_RBLN_SAMPLER=1 python run_decoder_only_simple.py # torch.compile (cpu) + no warmup
VLLM_USE_V1=1 VLLM_RBLN_ENABLE_WARM_UP=1 VLLM_RBLN_SAMPLER=1 python run_decoder_only_simple.py # torch.compile (cpu) + warmup
VLLM_USE_V1=1 VLLM_RBLN_ENABLE_WARM_UP=0 VLLM_RBLN_SAMPLER=2 python run_decoder_only_simple.py # new (rbln)