flag="--exp_name $3
      --run-type eval
      --exp-config vlnce_baselines/GT_waypoint/data_collect.yaml
      SIMULATOR_GPU_IDS [$2]
      TORCH_GPU_ID $2
      TORCH_GPU_IDS [$2]
      EVAL.SPLIT val_unseen
      EVAL_CKPT_PATH_DIR $1
      "
python run.py $flag