import numpy as np
import os
import cv2

# TransUNet - grayscale channel
#os.system("CUDA_VISIBLE_DEVICES=0 python train_G.py --dataset GaE_final_Gray --vit_name R50-ViT-B_16")
#os.system("python test_G.py --dataset GaE_final_Gray --is_savenii --vit_name R50-ViT-B_16")

# TransUNet - event_based channel
#os.system("CUDA_VISIBLE_DEVICES=0 python train_E.py --dataset GaE_final_EvB --vit_name R50-ViT-B_16")
#os.system("python test_E.py --dataset GaE_final_EvB --is_savenii --vit_name R50-ViT-B_16")


# data concat
#loss_event = np.load("./loss_record/loss_record_test_GaE_event.npy")
#print(loss_event[1][:].shape)
#print(loss_event)


#loss_record_test_GaE_gray.npy


# Xception



