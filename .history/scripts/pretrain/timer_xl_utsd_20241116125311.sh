export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_ENDPOINT=https://hf-mirror.com
model_name=timer
token_num=7
token_len=96
seq_len=$((token_num * token_len))

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./dataset/UTSD-full-npy \
  --model_id utsd \
  --model $model_name \
  --data Utsd \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len 720 \
  --test_seq_len $seq_len \
  --test_pred_len 720 \
  --e_layers 8 \
  --d_model 1024 \
  --d_ff 2048 \
  --batch_size 4096 \
  --learning_rate 0.00005 \
  --train_epochs 10 \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --use_norm \
  --valid_last \
  --dp \
  --devices 0,1,2,3