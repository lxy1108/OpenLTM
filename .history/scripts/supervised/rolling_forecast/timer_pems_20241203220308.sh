export CUDA_VISIBLE_DEVICES=1
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
  --root_path ../../TS-Library/dataset/PEMS/ \
  --data_path PEMS04.npz \
  --model_id PEMS04 \
  --model $model_name \
  --data MultivariateDatasetBenchmark  \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len 720 \
  --test_seq_len $seq_len \
  --test_pred_len $test_pred_len \
  --e_layers 6 \
  --d_model 512 \
  --d_ff 2048 \
  --batch_size 16 \
  --learning_rate 0.00005 \
  --lradj type1 \
  --train_epochs 10 \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --use_norm \
  --valid_last \
  --test_dir forecast_PEMS03_timer_MultivariateDatasetBenchmark_sl672_it96_ot720_lr5e-05_bt16_wd0_el6_dm512_dff2048_nh8_cosTrue_test_0
done