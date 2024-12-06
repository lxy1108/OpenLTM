export CUDA_VISIBLE_DEVICES=2
model_name=timer
token_num=7
token_len=96
seq_len=$((token_num * token_len))
# training one model with a context length
for test_pred_len in 720
do
for lr in 1e-4
do
python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ../../TS-Library/dataset/weather/ \
  --data_path weather.csv \
  --model_id WTH \
  --model $model_name \
  --data MultivariateDatasetBenchmark  \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $test_pred_len \
  --test_seq_len $seq_len \
  --test_pred_len $test_pred_len \
  --e_layers 4 \
  --d_model 512 \
  --d_ff 1024 \
  --batch_size 32 \
  --learning_rate $lr \
  --lradj type1 \
  --train_epochs 10 \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --use_norm \
  --valid_last
done
done