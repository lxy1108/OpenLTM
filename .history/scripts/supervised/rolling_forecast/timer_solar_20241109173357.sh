export CUDA_VISIBLE_DEVICES=1
model_name=timer
token_num=7
token_len=96
seq_len=$((token_num * token_len))
# training one model with a context length
for test_len in 96 192 336 720
do
for lr in  1e-5
do
python -u run.py \
  --task_name forecast \
  --is_training 0 \
  --root_path ../../TS-Library/dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id Solar_rel \
  --model $model_name \
  --data MultivariateDatasetBenchmark  \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len 720 \
  --test_seq_len $seq_len \
  --test_pred_len $test_len \
  --e_layers 6 \
  --d_model 512 \
  --d_ff 1024 \
  --batch_size 16 \
  --learning_rate $lr \
  --lradj type1 \
  --train_epochs 10 \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --use_norm \
  --valid_last \
  --test_dir forecast_Solar_rel_timer_MultivariateDatasetBenchmark_sl672_it96_ot720_lr1e-05_bt16_wd0_el6_dm512_dff1024_nh8_cosTrue_test_0
done
done