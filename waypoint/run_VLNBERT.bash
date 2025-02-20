# # TRAINING
# flag="--exp_name cont-cwp-vlnbert-ori
#       --run-type train
#       --exp-config run_VLNBERT.yaml
#       SIMULATOR_GPU_IDS [0]
#       TORCH_GPU_ID 0
#       TORCH_GPU_IDS [0]
#       IL.batch_size 16
#       IL.lr 1e-5
#       IL.epochs 60
#       IL.schedule_ratio 0.50
#       IL.decay_time 20
#       "
# python run.py $flag


# EVALUATION
# flag="--exp_name official_vlnbert
#       --run-type eval
#       --exp-config run_VLNBERT.yaml
#       SIMULATOR_GPU_IDS [1]
#       TORCH_GPU_ID 1
#       TORCH_GPU_IDS [1]
#       EVAL.SPLIT val_seen
#       EVAL_CKPT_PATH_DIR logs/checkpoints/cont-cwp-vlnbert-ori/vlnbert_ckpt_best.pth
#       "
# python run.py $flag



# INFERENCE
# flag="--exp_name official_vlnbert
#       --run-type inference
#       --exp-config run_VLNBERT.yaml
#       SIMULATOR_GPU_IDS [1]
#       TORCH_GPU_ID 1
#       TORCH_GPU_IDS [1]
#       EVAL.SAVE_RESULTS False
#       INFERENCE.PREDICTIONS_FILE test
#       INFERENCE.SPLIT test
#       INFERENCE.CKPT_PATH logs/checkpoints/cont-cwp-vlnbert-ori/vlnbert_ckpt_best.pth
#       "
# python run.py $flag