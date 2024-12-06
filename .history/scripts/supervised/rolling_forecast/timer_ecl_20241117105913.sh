export CUDA_VISIBLE_DEVICES=0
model_name=timer
token_num=7
token_len=96
seq_len=$((token_num * token_len))
# training one model with a context length
for test_pred_len in 720
do
python -u run.py \
  --task_name forecast \
  --is_training 0 \
  --root_path ../../TS-Library/dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_rel_rand \
  --model $model_name \
  --data MultivariateDatasetBenchmark  \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len 720 \
  --test_seq_len $seq_len \
  --test_pred_len $test_pred_len \
  --e_layers 8 \
  --d_model 1024 \
  --d_ff 2048 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --lradj type1 \
  --train_epochs 10 \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --use_norm \
  --valid_last \
  --test_dir forecast_utsd_timer_Utsd_sl672_it96_ot720_lr5e-05_bt16384_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0
done
# testing the model on all forecast lengths
# for test_pred_len in 96 192 336 720
# do
# python -u run.py \
#   --task_name forecast \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id ECL \
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
#   --test_dir forecast_ECL_timer_Custom_Multi_sl672_it96_ot96_lr0.0005_bt4_wd0_el5_dm512_dff2048_nh8_cosTrue_test_0
# done