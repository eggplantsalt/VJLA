CUDA_VISIBLE_DEVICES=0 python deploy/libero/run_libero_eval.py \
  --model_family instruct_vla \
  --pretrained_checkpoint  /storage/v-xiangxizheng/zy_workspace/VJLA/outputs/instructvla_finetune_v2/checkpoints/step-013500-epoch-01-loss=0.1093.pt \
  --task_suite_name libero_goal \
  --local_log_dir /storage/v-xiangxizheng/zy_workspace/VJLA/Libero/logs \
  --use_length -1 \
  --center_crop True \
  --prompt_mode image_text_primary
