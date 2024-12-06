export CUDA_VISIBLE_DEVICES=0,1,2,3
model_name=timer
token_num=8
token_len=96
seq_len=$((token_num * token_len))

python -u run.py \
  --task_name forecast \
  --is_training 0 \
  --root_path ./datasets/ERA5/ERA5-Large \
  --data_path pretrain.npy \
  --model_id era5_pretrain \
  --model $model_name \
  --data Era5_Pretrain_Test  \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len 96 \
  --test_seq_len $seq_len \
  --test_pred_len 96 \
  --e_layers 4 \
  --d_model 512 \
  --d_ff 2048 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --test_dir forecast_era5_pretrain_timer_Era5_Pretrain_sl768_it96_ot96_lr0.0001_bt24_wd0_el4_dm512_dff2048_nh8_cosTrue_test_0

  # --test_dir forecast_era5_pretrain_timer_Era5_Pretrain_sl768_it96_ot96_lr5e-05_bt64_wd0_el4_dm512_dff2048_nh8_cosTrue_test_0