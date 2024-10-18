export CUDA_VISIBLE_DEVICES=2
model_name=timer_lite
token_num=7
token_len=96
seq_len=$[$token_num*$token_len]
# training one model with a context length
python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL \
  --model $model_name \
  --data Custom_Multi  \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len 96 \
  --e_layers 2 \
  --d_layers 2 \
  --d_model 512 \
  --d_ff 2048 \
  --batch_size 4 \
  --learning_rate 0.0005 \
  --train_epochs 10 \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --use_norm \
  --valid_last