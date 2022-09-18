import numpy as np
import os
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
import pickle

# Paths to traning images
train_data_path = "./GaE_Dataset/Training/Grayscale/New_folder"
train_label_path = "./GaE_Dataset/Training/Masks/New_folder"

# Path to save generated .npz files
npz_saving_path = "./GaE/train_npz/"

# This is used to filter bad images
# preserving_ratio = 0.25


filenames = os.listdir(train_data_path)

png_train_list = []
for filename in filenames: # Get all images name
    if filename.endswith('png'):
        png_train_list.append(filename)
print('Number of images found : ', len(png_train_list))

png_train_list.sort()

filenames2 = os.listdir(train_label_path)

png_label_list = []
for filename in filenames2: # Get all labels name
    if filename.endswith('png'):
        png_label_list.append(filename)
print('Number of labels found : ', len(png_label_list))

png_label_list.sort()

list_file = open('./train.txt','a')
 
for i, f in enumerate(png_train_list):
    # Read traning image
    data_path = os.path.join(train_data_path, f)
    img_o = cv2.imread(data_path)
    img = cv2.resize(img_o, (320,320), interpolation=cv2.INTER_AREA)  
    img = img / np.amax(img)  # Normalisation
    img_avg = (img[:,:,0]+img[:,:,1]+img[:,:,2])/3
    img_arr = np.asarray(img_avg) 

    # Read traning label
    label_path = os.path.join(train_label_path, png_label_list[i])
    img_label_o = cv2.imread(label_path)
    img_label = cv2.resize(img_label_o, (320,320), interpolation=cv2.INTER_AREA)
    img_label_avg = (img_label[:,:,0]+img_label[:,:,1]+img_label[:,:,2])/3
    above_1 = np.nonzero(img_label_avg)
    img_label_avg[above_1] = 1
    img_label_arr = np.asarray(img_label_avg) 

    # Store data as .npz format
    if not os.path.exists(npz_saving_path):
        os.makedirs(npz_saving_path)
    np.savez(os.path.join(npz_saving_path, f[0:-4]+'.npz'), image=img_arr, label=img_label_arr)

    # Make a record in list
    list_file.write(f[0:-4])
    list_file.write('\n')
    
    print(f+'.npz finished.')

list_file.close()
print('All .npz transform complete!')


# Show dataset with npz format
train = np.load(os.path.join(npz_saving_path,png_train_list[0][0:-4]+'.npz'))  
cv2.imshow('train_data',train['image'])
cv2.waitKey(5000)
cv2.imshow('train_label',train['label'])
cv2.waitKey(5000)
