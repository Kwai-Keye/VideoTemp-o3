CUDA_VISIBLE_DEVICES=0 \
DECORD_EOF_RETRY_MAX=20480 \
VIDEO_MIN_PIXELS=50176 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=1024 \
swift deploy \
    --model_type qwen2_5_vl \
    --model /mmu_mllm_hdd_2/liuwenqi/code_field/rag/verl/checkpoints/sft_videotemp_o3_256/v0-20260309-161110/checkpoint-616 \
    --infer_backend vllm \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 81920 \
    --max_new_tokens 2048 \
    --vllm_limit_mm_per_prompt '{"image": 5, "video": 20}' \
    --temperature 0.1 \
    --vllm_mm_processor_cache_gb 0 \
    --served_model_name Qwen2.5-VL-7B \
    --port 8000
