PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
MAX_PIXELS=501760 \
DECORD_EOF_RETRY_MAX=20480 \
VIDEO_MIN_PIXELS=50176 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=512 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
NPROC_PER_NODE=6 \
swift rlhf \
    --rlhf_type grpo \
    --model sft/ckpt/test \
    --train_type full \
    --external_plugins rl/video_crop_plugin.py \
    --reward_funcs acc_reward iou_reward tool_penalty format_reward \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 0.0.0.0 \
    --vllm_server_port 8100 \
    --vllm_server_pass_dataset true \
    --torch_dtype bfloat16 \
    --freeze_vit true \
    --freeze_aligner false \
    --freeze_llm false \
    --dataset rl/data/qa.jsonl rl/data/grounding.jsonl \
    --split_dataset_ratio 0 \
    --max_completion_length 4096 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 5e-6 \
    --gradient_accumulation_steps 12 \
    --save_only_model false \
    --save_strategy 'steps' \
    --save_steps 50 \
    --save_total_limit 20 \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 64 \
    --dataset_num_proc 64 \
    --num_generations 8 \
    --temperature 1.0 \
    --log_completions true \
    --log_entropy true \
    --steps_per_generation 12 \
    --num_iterations 1 \
    --attn_impl flash_attn \
    --deepspeed zero2 \
    --report_to tensorboard \
    --output_dir rl/ckpt/test \
