CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
VIDEO_MIN_PIXELS=50176 \
VIDEO_MAX_PIXELS=50176 \
DECORD_EOF_RETRY_MAX=20480 \
FPS_MAX_FRAMES=512 \
swift sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --model_type qwen2_5_vl \
    --train_type full \
    --dataset sft/data/wo_tool_call/activitynet.jsonl \
            sft/data/wo_tool_call/charades.jsonl \
            sft/data/wo_tool_call/vidchapters.jsonl \
            sft/data/wo_tool_call/video_r1_image_mc.jsonl \
            sft/data/wo_tool_call/video_r1_video.jsonl \
            sft/data/wi_tool_call/activitynet.jsonl \
            sft/data/wi_tool_call/qvhighlight.jsonl \
            sft/data/wi_tool_call/longvila.jsonl \
    --torch_dtype bfloat16 \
    --external_plugins sft/loss_scale_plugin.py \
    --loss_scale last_two_rounds \
    --freeze_vit True \
    --freeze_aligner False \
    --freeze_llm False \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 32 \
    --save_only_model False \
    --save_strategy epoch \
    --save_total_limit 10 \
    --logging_steps 2 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 16 \
    --attn_impl flash_attn \
    --deepspeed zero3 \
    --output_dir sft/ckpt/test
