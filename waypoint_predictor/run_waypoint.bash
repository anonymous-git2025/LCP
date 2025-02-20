flag="--EXP_ID att_lang_no_pretrained_second_round

      --TRAINEVAL train
      --VIS 0

      --ANGLES 120
      --NUM_IMGS 12

      --EPOCH 300
      --BATCH_SIZE 64
      --LEARNING_RATE 1e-6

      --WEIGHT 0

      --TRM_LAYER 2
      --TRM_NEIGHBOR 1
      --HEATMAP_OFFSET 5
      --HIDDEN_DIM 768

      --lang attened_lang
      "

python waypoint_predictor_lang.py $flag


