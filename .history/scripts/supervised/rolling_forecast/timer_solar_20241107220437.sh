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
  --model_id Solar_rel \
  --model $model_name \
  --data MultivariateDatasetBenchmark  \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len 96 \
  --e_layers 6 \
  --d_model 512 \
  --d_ff 1024 \
  --batch_size 16 \
  --learning_rate 0.00001 \
  --lradj type1 \
  --train_epochs 10 \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --use_norm \
  --valid_last