import os
os.system("CUDA_VISIBLE_DEVICES=0 python train_E.py --dataset GaE_EvB --vit_name R50-ViT-B_16")

os.system("python test_E.py --dataset GaE_EvB --is_savenii --vit_name R50-ViT-B_16")