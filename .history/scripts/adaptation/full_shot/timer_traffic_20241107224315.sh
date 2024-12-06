export CUDA_VISIBLE_DEVICES=0,1
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
  --model_id Traffic \
  --model $model_name \
  --data MultivariateDatasetBenchmark  \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len $test_pred_len \
  --e_layers 8 \
  --d_model 1024 \
  --d_ff 2048 \
  --batch_size 4 \
  --learning_rate 0.00002 \
  --train_epochs 10 \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --use_norm \
  --valid_last \
  --adaptation \
  --pretrain_model_path ../../Large-Time-Series-Model/checkpoints/Timer_forecast_1.0.ckpt \
  --dp \
  --devices 0,1,2
done