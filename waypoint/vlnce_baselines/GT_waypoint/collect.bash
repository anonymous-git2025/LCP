export GLOG_minloglevel=2
export MAGNUM_LOG=quiet


# EVALUATION
flag="--exp_name attened_lang_no_pretrained_round_three
      --run-type eval
      --exp-config vlnce_baselines/GT_waypoint/data_collect.yaml
      SIMULATOR_GPU_IDS [1]
      TORCH_GPU_ID 1
      TORCH_GPU_IDS [1]
      EVAL.SPLIT val_unseen
      EVAL_CKPT_PATH_DIR logs/checkpoints/ckpt.0.pth
      "
python run.py $flag

