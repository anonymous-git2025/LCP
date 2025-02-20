# flag="--exp_name attened_lang_no_pretrained_round_two
#       --run-type train
#       --exp-config vlnce_baselines/add_lang/add_lang.yaml
#       RANDOM_ALLOC_SCENE False
#       GPU_NUMBERS 3
#       SIMULATOR_GPU_IDS [5,6,7]
#       TORCH_GPU_IDS [5,6,7]
#       IL.batch_size 16
#       IL.lr 3e-5
#       IL.epochs 60
#       IL.schedule_ratio 0.50
#       IL.decay_time 40
#       WP_CKPT_LANG waypoint_predictor/waypoint_predict/checkpoints/att_lang_no_pretrained_second_round/snap/epoch0
#       LANG attened_lang
#       IL.load_from_ckpt True
#       IL.ckpt_to_load waypoint/waypoint/logs/checkpoints/attened_lang_no_pretrained_round_two/ckpt.0.pth
#       IL.is_requeue True
#       MODEL.LANG_ENCODER.lang_checkpoint logs/checkpoints/attened_lang_no_pretrained/ckpt.0.pth
#       CHECKPOINT_FOLDER logs/new_ckpt/
#       "
# python -m torch.distributed.launch --nproc_per_node=3 --master_port=$RANDOM run.py $flag

