flag="--exp_name attened_lang_no_pretrained_round_three
      --run-type eval
      --exp-config vlnce_baselines/add_lang/add_lang.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_ID 0
      TORCH_GPU_IDS [0]
      EVAL.SPLIT val_seen
      EVAL_CKPT_PATH_DIR waypoint/waypoint/logs/new_ckpt/attened_lang_no_pretrained_round_three/ckpt.0.pth
      WP_CKPT_LANG waypoint_predictor/waypoint_predict/checkpoints/att_lang_no_pretrained_third_round/snap/epoch0
      LANG attened_lang
      MODEL.LANG_ENCODER.lang_checkpoint logs/checkpoints/attened_lang_no_pretrained_round_two/ckpt.0.pth
      "
python run.py $flag