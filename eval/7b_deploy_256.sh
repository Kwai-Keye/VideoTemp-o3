CUDA_VISIBLE_DEVICES=0 \
DECORD_EOF_RETRY_MAX=20480 \
VIDEO_MIN_PIXELS=200704 \
VIDEO_MAX_PIXELS=200704 \
FPS_MAX_FRAMES=256 \
swift deploy \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --infer_backend vllm \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_max_model_len 81920 \
    --max_new_tokens 2048 \
    --vllm_limit_mm_per_prompt '{"image": 5, "video": 20}' \
    --temperature 0 \
    --served_model_name Qwen2.5-VL-7B \
    --port 8000
