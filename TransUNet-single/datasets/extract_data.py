import numpy as np
import os
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
import pickle

# Paths to .nii.gz files
#train_data_path = "./train_data/"
train_label_path = "./train_label/"
train_data_path = "./test_data/"



# 生成的 pickle 文件存放路径
pickle_data_saving_path = "./dataset_pickle/"

# 保留非 0 数据比重超过 preserving_ratio 的切片
# preserving_ratio = 0.25


filenames = os.listdir(train_data_path)

nii_train_list = []
for filename in filenames:  # 获取所有.nii.gz的文件名
    if filename.endswith('nii.gz'):
        nii_train_list.append(filename)
print('.nii.gz文件数为：', len(nii_train_list))


 
nii_train_dataset = []
for i, f in enumerate(nii_train_list):
    data_path = os.path.join(train_data_path, f)
    data = nib.load(data_path).get_data()  # 获取.nii.gz文件中的数据
    print('.nii.gz文件的维度为：',data.shape)   #宽*高*切片数

    data = data / np.amax(data)  # Normalisation
    for i in range(data.shape[2]):  # 对切片进行循环，选出满足要求的切片
        img = data[:, :, i]  # 每一个切片是一个灰色图像
        #if float(np.count_nonzero(img) / img.size) >= preserving_ratio:
        #img = np.transpose(img, (1, 0))  # 将图片顺时针旋转90度（摆正了）
        nii_train_dataset.append(img)
print('选出符合preserving_ratio的图片有：', len(nii_train_dataset))
pickle_train_dataset = np.asarray(nii_train_dataset)  # list类型转为数组  切片数*宽*高
 

# Store dataset as .npz format
if not os.path.exists(pickle_data_saving_path):
    os.makedirs(pickle_data_saving_path)
np.savez(os.path.join(pickle_data_saving_path, 'test.npz'), pickle_train_dataset[0])

print('.nii.gz transform complete. npz finished !')


# Show dataset with npz format
train = np.load(os.path.join(pickle_data_saving_path,'test.npz'))    #train 与原来的 pickle_train_dataset 一模一样
#print(train['arr_0'])
cv2.imshow('img',train['arr_0'])
cv2.waitKey(100)
#for i in range(len(nii_train_dataset)):
#    cv2.imshow('img',train['arr_'+str(i)])
#    cv2.waitKey(100)
#cv2.destroyAllWindows()



# 将数据保存为pickle类型，方便读取
#if not os.path.exists(pickle_data_saving_path):
#    os.makedirs(pickle_data_saving_path)
#with open(os.path.join(pickle_data_saving_path, 'training.pickle'), 'wb') as f:
#    pickle.dump(pickle_train_dataset, f, protocol=4)
 
#print('.nii.gz transform complete. pickle finished !')

 
#######################################################
#需要用pickle数据的话，可以用下面方法读取，eg如下
#f = open(os.path.join(pickle_data_saving_path,'training.pickle'),'rb')
#train = pickle.load(f)    #train 与原来的 pickle_train_dataset 一模一样
#for i in range(train.shape[0]):
#    cv2.imshow('img',train[i])
#    cv2.waitKey(100)
#cv2.destroyAllWindows()
