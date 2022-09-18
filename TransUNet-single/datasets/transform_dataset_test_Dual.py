import numpy as np
import os
import cv2
import pickle

# Paths to traning images
test_gray_path = "./GaE_Dataset/Testing/Grayscale/New_folder"
test_event_path = "./GaE_Dataset/Testing/Events/New_folder"
test_label_path = "./GaE_Dataset/Testing/Masks/New_folder"

# Path to save generated .npz files
npz_saving_path = "./GaE_Dual/test_npz/"

# This is used to filter bad images
# preserving_ratio = 0.25

# import grayscale image 
filenames = os.listdir(test_gray_path)

gray_test_list = []
for filename in filenames: # Get all images name
    if filename.endswith('png'):
        gray_test_list.append(filename)
print('Number of grayscale images found : ', len(gray_test_list))

gray_test_list.sort()

# import event-based image
filenames1 = os.listdir(test_event_path)

event_test_list = []
for filename in filenames1: # Get all images name
    if filename.endswith('png'):
        event_test_list.append(filename)
print('Number of grayscale images found : ', len(event_test_list))

event_test_list.sort()

# import labels
filenames2 = os.listdir(test_label_path)

png_label_list = []
for filename in filenames2: # Get all labels name
    if filename.endswith('png'):
        png_label_list.append(filename)
print('Number of labels found : ', len(png_label_list))

png_label_list.sort()


list_file = open('../lists/lists_GaE_Dual/test.txt','a')
 
for i, f in enumerate(gray_test_list):
    # Read grayscale image
    gray_path = os.path.join(test_gray_path, f)
    img_g_o = cv2.imread(gray_path)
    img_g = cv2.resize(img_g_o, (320,320), interpolation=cv2.INTER_AREA)  
    img_g = img_g / np.amax(img_g)  # Normalisation
    img_g_avg = (img_g[:,:,0]+img_g[:,:,1]+img_g[:,:,2])/3
    img_g_arr = np.asarray(img_g_avg) 
    img_g_arr = np.expand_dims(img_g_arr,axis=-1)

    # Read event-based image
    evb_path = os.path.join(test_event_path, event_test_list[i])
    img_e_o = cv2.imread(evb_path)
    img_e = cv2.resize(img_e_o, (320,320), interpolation=cv2.INTER_AREA)  
    img_e = img_e / np.amax(img_e)  # Normalisation
    img_e_avg = (img_e[:,:,0]+img_e[:,:,1]+img_e[:,:,2])/3
    img_e_arr = np.asarray(img_e_avg) 
    img_e_arr = np.expand_dims(img_e_arr,axis=-1)

    # Read traning label
    label_path = os.path.join(test_label_path, png_label_list[i])
    img_label_o = cv2.imread(label_path)
    img_label = cv2.resize(img_label_o, (320,320), interpolation=cv2.INTER_AREA)
    img_label_avg = (img_label[:,:,0]+img_label[:,:,1]+img_label[:,:,2])/3
    above_1 = np.nonzero(img_label_avg)
    img_label_avg[above_1] = 1
    img_label_arr = np.asarray(img_label_avg) 

    # Store data as .npz format
    img_arr = np.concatenate((img_g_arr, img_e_arr), axis=-1) # merge images together

    if not os.path.exists(npz_saving_path):
        os.makedirs(npz_saving_path)
    np.savez(os.path.join(npz_saving_path, 'GaE Dual '+f[0:-4]+'.npz'), image=img_arr, label=img_label_arr)

    # Make a record in list
    list_file.write('GaE Dual '+f[0:-4])
    list_file.write('\n')
    
    print('GaE Dual '+f[0:-4]+'.npz finished.')

list_file.close()
print('All .npz transform complete!')


# Show dataset with npz format
test = np.load(os.path.join(npz_saving_path, 'GaE Dual '+ gray_test_list[0][0:-4]+'.npz'))  
cv2.imshow('test_data',test['image'][:,:,0])
cv2.waitKey(5000)
cv2.imshow('test_label',test['label'])
cv2.waitKey(5000)
