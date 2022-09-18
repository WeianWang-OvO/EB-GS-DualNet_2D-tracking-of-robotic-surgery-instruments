import os
os.system("CUDA_VISIBLE_DEVICES=0 python train_G.py --dataset GaE_Gray --vit_name R50-ViT-B_16")

os.system("python test_G.py --dataset GaE_Gray --is_savenii --vit_name R50-ViT-B_16")