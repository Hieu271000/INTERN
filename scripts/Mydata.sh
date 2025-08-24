export CUDA_VISIBLE_DEVICES=0

# --- Giai đoạn Huấn luyện ---
python main.py --num_epochs 25    --batch_size 32  --mode train --dataset MyData  --data_path MyData --input_c 1  --output_c 1  --loss_fuc MSE  --win_size 28  --patch_size 7

# --- Giai đoạn Kiểm thử ---
python main.py --anormly_ratio 3  --num_epochs 5  --batch_size 32  --mode test    --dataset MyData   --data_path MyData --input_c 1  --output_c 1  --loss_fuc MSE  --win_size 28  --patch_size 7


