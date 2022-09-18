import os
import tensorflow as tf
import numpy as np
from h5py._hl import dataset
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from segmentation_models.metrics import iou_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
import datetime


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

'''
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[1], enable=True)

from tensorflow.python.framework.ops import disable_eager_execution
tf.compat.v1.experimental.output_all_intermediates(True)
disable_eager_execution()
'''

train_aug = ImageDataGenerator(horizontal_flip=True, rescale=1. / 255,
                                     vertical_flip=True, dtype='float32',  
                                     fill_mode='constant', cval=125, zoom_range=[0.9, 1.1],
                                     rotation_range=180)

val_aug = ImageDataGenerator(dtype='float32', rescale=1. / 255)

test_aug = ImageDataGenerator(dtype='float32', rescale=1. / 255)


list_train_val = open('./lists/lists_GaE_EvB/train.txt').readlines()
list_train = list_train_val[0:len(list_train_val)*6//7]
list_val = list_train_val[len(list_train_val)*6//7:]

list_test = open('./lists/lists_GaE_EvB/test.txt').readlines()

data_dir = '../data/GaE_EvB'

def generator_train(bs, aug=None):
    images = []
    labels = []
    i_count = 0
    while True:
        while len(images) < bs:
            if i_count >= len(list_train):
                i_count = 0
            slice_name = list_train[i_count].strip('\n')
            data_path = os.path.join(data_dir+'/train_npz'+'/', slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            image = np.expand_dims(image,axis=-1)
            images.append(image)
            labels.append(label)
            i_count = i_count+1
		
        if aug is not None:
            (images, labels) = next(aug.flow(np.array(images), labels, batch_size=bs))
		
        yield np.array(images), labels

def generator_val(bs, aug=None):
    images = []
    labels = []
    i_count = 0
    while True:
        while len(images) < bs:
            if i_count >= len(list_val):
                i_count = 0
            slice_name = list_val[i_count].strip('\n')
            data_path = os.path.join(data_dir+'/train_npz'+'/', slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            image = np.expand_dims(image,axis=-1)
            images.append(image)
            labels.append(label)
            i_count = i_count+1
		
        if aug is not None:
            (images, labels) = next(aug.flow(np.array(images), labels, batch_size=bs))
		
        yield (np.array(images),labels)

def generator_test(bs, aug=None):
    images = []
    labels = []
    i_count = 0
    while True:
        while len(images) < bs:
            if i_count >= len(list_test):
                break
            slice_name = list_test[i_count].strip('\n')
            data_path = os.path.join(data_dir+'/test_npz'+'/', slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            image = np.expand_dims(image,axis=-1)
            images.append(image)
            labels.append(label)
            i_count = i_count+1

        if aug is not None:
            (images, labels) = next(aug.flow(np.array(images), labels, batch_size=bs))

        yield (np.array(images),labels)


Image_Size = [320, 320, 1]

# hyperparameters

BS = 8
NUM_EPOCHS = 150

Event_Input = tf.keras.layers.Input(shape=Image_Size)
Event_Input = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(Event_Input)


Event_Model = tf.keras.applications.xception.Xception(
    include_top=False, weights=None, input_tensor=Event_Input,
    input_shape=(320, 320, 1), pooling=max,
)
for layer in Event_Model.layers:
    layer._name = layer._name + str('_B')

output_1 = Event_Model.get_layer('block2_sepconv2_bn_B').output
output_2 = Event_Model.get_layer('block3_sepconv2_bn_B').output
output_3 = Event_Model.get_layer('block4_sepconv2_bn_B').output
output_4 = Event_Model.get_layer('block13_sepconv2_bn_B').output
output_5 = Event_Model.get_layer('block14_sepconv2_bn_B').output

decoder_0 = tf.keras.layers.Conv2DTranspose(filters=2048, kernel_size=3, strides=2, activation='relu',
                                            padding='same')(output_5)
decoder_0 = layers.BatchNormalization()(decoder_0)
decoder_0 = tf.image.resize(decoder_0, (tf.shape(output_5)[1], tf.shape(output_5)[2]))
decoder_0 = tf.concat([decoder_0, output_5], 3)

decoder_1 = tf.keras.layers.Conv2D(filters=2048, kernel_size=3, activation='relu', padding='same')(decoder_0)
decoder_1 = tf.keras.layers.Conv2DTranspose(filters=1024, kernel_size=3, strides=2, activation='relu',
                                            padding='same')(decoder_1)
decoder_1 = layers.BatchNormalization()(decoder_1)
decoder_1 = tf.image.resize(decoder_1, (tf.shape(output_4)[1], tf.shape(output_4)[2]))
decoder_1 = tf.concat([decoder_1, output_4], 3)
#
decoder_2 = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, activation='relu', padding='same')(decoder_1)
decoder_2 = tf.keras.layers.Conv2DTranspose(filters=728, kernel_size=3, strides=2, activation='relu',
                                            padding='same')(decoder_2)
decoder_2 = layers.BatchNormalization()(decoder_2)
decoder_2 = tf.image.resize(decoder_2, (tf.shape(output_3)[1], tf.shape(output_3)[2]))
decoder_2 = tf.concat([decoder_2, output_3], 3)
#
decoder_3 = tf.keras.layers.Conv2D(filters=728, kernel_size=3, activation='relu', padding='same')(decoder_2)
decoder_3 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2, activation='relu',
                                            padding='same')(decoder_3)
decoder_3 = layers.BatchNormalization()(decoder_3)
decoder_3 = tf.image.resize(decoder_3, (tf.shape(output_2)[1], tf.shape(output_2)[2]))
decoder_3 = tf.concat([decoder_3, output_2], 3)
#
decoder_4 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(decoder_3)
decoder_4 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, activation='relu',
                                            padding='same')( decoder_4)
decoder_4 = layers.BatchNormalization()(decoder_4)
decoder_4 = tf.image.resize(decoder_4, (tf.shape(output_1)[1], tf.shape(output_1)[2]))
decoder_4 = tf.concat([decoder_4, output_1], 3)
decoder_4 = tf.image.resize(decoder_4, [320, 320])
#
Event_Output = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(decoder_4)

Event_Model = Model(inputs=Event_Model.input, outputs=Event_Output)
Event_Model.compile(optimizer='adam', loss="binary_crossentropy", metrics=[iou_score])
#Event_Model.summary()


checkpoint_path = "../Weights Event Xception Test/cp.ckpt"
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)


#checkPoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1,
#                             save_best_only=True, save_weights_only=True,
#                             mode='min', period=1)
#callbacks_list = [checkPoint]

H = Event_Model.fit_generator(generator_train(BS, train_aug),
                    steps_per_epoch=len(list_train) // BS, epochs=NUM_EPOCHS,
                    verbose=1,  #callbacks=[callbacks_list],  # , lr_callback],
                    validation_data=generator_val(BS, val_aug),
                    validation_steps=len(list_val) // BS, validation_freq=1, class_weight=None, max_queue_size=10, workers=1,
                    use_multiprocessing=False, initial_epoch=0)

with open('train_Xception_Event_H.txt','wb') as file_H:
    pickle.dump(H.history, file_H)

#Event_Pred = Event_Model.predict_generator(generator_test(BS, test_aug), steps=len(list_test) // BS)

scores = Event_Model.evaluate_generator(generator_test(BS, test_aug), workers=10, verbose=0)

with open('test_Xception_Event_S.txt','wb') as file_S:
    pickle.dump(scores, file_S)

print(Event_Model.metrics_names)
print('\n')
print(scores)

                    