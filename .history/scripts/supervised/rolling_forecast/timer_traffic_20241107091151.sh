export CUDA_VISIBLE_DEVICES=3
model_name=timer
token_num=7
token_len=96
seq_len=$((token_num * token_len))
# training one model with a context length
for test_pred_len in 96
do
python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ../../TS-Library/dataset/traffic/ \
  --data_path traffic.csv \
  --model_id Traffic_rel \
  --model $model_name \
  --data MultivariateDatasetBenchmark  \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len $test_pred_len \
  --e_layers 4 \
  --d_model 512 \
  --d_ff 1024 \
  --batch_size 4 \
  --learning_rate 0.0005 \
  --train_epochs 10 \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --use_norm \
  --valid_last
  # --test_dir forecast_Traffic_last_lradj4_timer_MultivariateDatasetBenchmark_sl672_it96_ot96_lr0.0005_bt4_wd0_el4_dm512_dff1024_nh8_cosTrue_test_0
done
  # --dp \
  # --devices 0,1
  # --adaptation \
  # --pretrain_model_path ../../Large-Time-Series-Model/checkpoints/Timer_forecast_1.0.ckpt
  # --pretrain_model_path checkpoints/forecast_Traffic_uni_timer_MultivariateDatasetBenchmark_sl672_it96_ot96_lr0.0005_bt4_wd0_el4_dm512_dff1024_nh8_cosTrue_test_0/checkpoint.pth

# testing the model on all forecast lengths
# for test_pred_len in 96 192 336 720
# do
# python -u run.py \
#   --task_name forecast \
#   --is_training 1 \
#   --root_path ../../TS-Library/dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id Traffic \
#   --model $model_name \
#   --data MultivariateDatasetBenchmark  \
#   --seq_len $seq_len \
#   --input_token_len $token_len \
#   --output_token_len $token_len \
#   --test_seq_len $seq_len \
#   --test_pred_len $test_pred_len \
#   --e_layers 5 \
#   --d_model 512 \
#   --d_ff 2048 \
#   --batch_size 4 \
#   --learning_rate 0.0005 \
#   --train_epochs 10 \
#   --gpu 0 \
#   --cosine \
#   --tmax 10 \
#   --use_norm \
#   --valid_last \
#   --test_dir forecast_Traffic_timer_Custom_Multi_sl672_it96_ot96_lr0.0005_bt4_wd0_el4_dm512_dff2048_nh8_cosTrue_test_0
# done