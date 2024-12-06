export CUDA_VISIBLE_DEVICES=3
model_name=timer
token_num=8
token_len=96
seq_len=$((token_num * token_len))

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./datasets/ERA5/ERA5-Large \
  --data_path pretrain.npy \
  --model_id era5_pretrain_test \
  --model $model_name \
  --data Era5_Pretrain  \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len 96 \
  --test_seq_len $seq_len \
  --test_pred_len 96 \
  --e_layers 4 \
  --d_model 512 \
  --d_ff 2048 \
  --batch_size 64 \
  --learning_rate 0.00005 \
  --train_epochs 10 \
  --gpu 0 \
  --cosine \
  --tmax 10