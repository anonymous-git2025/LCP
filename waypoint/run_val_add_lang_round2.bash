flag="--exp_name $3
      --run-type eval
      --exp-config vlnce_baselines/add_lang/add_lang.yaml
      SIMULATOR_GPU_IDS [$2]
      TORCH_GPU_ID $2
      TORCH_GPU_IDS [$2]
      EVAL.SPLIT val_unseen
      EVAL_CKPT_PATH_DIR $1
      WP_CKPT $4
      WP_CKPT_LANG $5
      LANG $6
      MODEL.LANG_ENCODER.lang_checkpoint $7
      "
python run.py $flag