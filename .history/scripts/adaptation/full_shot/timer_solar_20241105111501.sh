export CUDA_VISIBLE_DEVICES=2
model_name=timer
token_num=7
token_len=96
seq_len=$((token_num * token_len))
# training one model with a context length
python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ../../TS-Library/dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id Solar_finetune \
  --model $model_name \
  --data MultivariateDatasetBenchmark  \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len 96 \
  --e_layers 8 \
  --d_model 1024 \
  --d_ff 2048 \
  --batch_size 16 \
  --learning_rate 5e-5 \
  --train_epochs 10 \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --use_norm \
  --valid_last \
  --test_dir forecast_Solar_finetune_timer_MultivariateDatasetBenchmark_sl672_it96_ot96_lr5e-05_bt16_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0
  # --adaptation \
  # --pretrain_model_path ../../Large-Time-Series-Model/checkpoints/Timer_forecast_1.0.ckpt