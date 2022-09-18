import os
# os.environ['SM_FRAMEWORK'] = 'tf.keras'
import tensorflow as tf
from h5py._hl import dataset
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from segmentation_models.metrics import iou_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from segmentation_models.losses import bce_jaccard_loss
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[1], enable=True)
from tensorflow.python.framework.ops import disable_eager_execution
tf.compat.v1.experimental.output_all_intermediates(True)
disable_eager_execution()


train_image_gen = ImageDataGenerator(horizontal_flip=True, rescale=1. / 255,
                                     vertical_flip=True, dtype='float32',  
                                     fill_mode='constant', cval=125, zoom_range=[0.9, 1.1],
                                     rotation_range=180)
train_gt_gen = ImageDataGenerator(horizontal_flip=True, rescale=1. / 255,
                                  vertical_flip=True, dtype='float32', 
                                  fill_mode='constant', cval=125, zoom_range=[0.9, 1.1],
                                  rotation_range=180)
train_event_gen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, dtype='float32', rescale=1. / 255,
                                     brightness_range=None, fill_mode='constant', cval=125, zoom_range=[0.9, 1.1],
                                     rotation_range=180)                                  

val_image_gen = ImageDataGenerator(dtype='float32', rescale=1. / 255)#
val_event_gen = ImageDataGenerator(dtype='float32', rescale=1. / 255)
val_gt_gen = ImageDataGenerator(dtype='float32', rescale=1. / 255)


test_image_gen = ImageDataGenerator(dtype='float32', rescale=1. / 255)
test_event_gen = ImageDataGenerator(dtype='float32', rescale=1. / 255)
test_gt_gen = ImageDataGenerator(dtype='float32', rescale=1. / 255)

scale = 2

def generate_generator_multiple_train(train_image_gen, train_event_gen, train_gt_gen):

    train_img_generator = train_image_gen.flow_from_directory(
        '/home/fahd_weian/my_workspace/DGX_Dataset/Training/Grayscale/'.format(dataset),
        target_size=(320, 470),
        color_mode='grayscale', class_mode=None, batch_size=8, shuffle=True,
        seed=1,
        interpolation='bilinear')

    train_event_generator = train_event_gen.flow_from_directory(
        '/home/fahd_weian/my_workspace/DGX_Dataset/Training/Events/'.format(dataset),
        target_size=(320, 470),
        color_mode='grayscale', class_mode=None, batch_size=8, shuffle=True, seed=1,
        interpolation='bilinear')

    train_gt_generator = train_gt_gen.flow_from_directory(
        '/home/fahd_weian/my_workspace/DGX_Dataset/Training/Masks/'.format(dataset),
        target_size=(320, 470),
        color_mode='grayscale', class_mode=None, batch_size=8,
        shuffle=True, seed=1,
        interpolation='nearest')
    
    while True:
        img_gen = train_img_generator.next()
        event_gen = train_event_generator.next()
        gt_gen = train_gt_generator.next()
        yield [img_gen, event_gen],  gt_gen

def generate_generator_multiple_val(val_image_gen, val_event_gen, val_gt_gen):
    val_img_generator = val_image_gen.flow_from_directory(
        '/home/fahd_weian/my_workspace/DGX_Dataset/Validation/Grayscale/'.format(dataset),
        target_size=(320, 470),
        color_mode='grayscale', class_mode=None, batch_size=4, shuffle=False,
        seed=1,
        interpolation='bilinear')

    val_event_generator = val_event_gen.flow_from_directory(
        '/home/fahd_weian/my_workspace/DGX_Dataset/Validation/Events/'.format(dataset),
        target_size=(320, 470),
        color_mode='grayscale', class_mode=None, batch_size=4, shuffle=True, seed=1,
        interpolation='bilinear')

    val_gt_generator = val_gt_gen.flow_from_directory(
        '/home/fahd_weian/my_workspace/DGX_Dataset/Validation/Masks/'.format(dataset),
        target_size=(320, 470),
        color_mode='grayscale', class_mode=None, batch_size=4,
        shuffle=False, seed=1,
        interpolation='nearest')
    
    while True:
        img_gen = val_img_generator.next()
        gt_gen = val_gt_generator.next()
        event_gen = val_event_generator.next()
        yield [img_gen, event_gen], gt_gen

def generate_generator_multiple_test(test_image_gen, test_event_gen, test_gt_gen):  

    test_img_generator = test_image_gen.flow_from_directory(
        '/home/fahd_weian/my_workspace/DGX_Dataset/Testing/Grayscale/'.format(dataset),
        target_size=(320, 470),
        color_mode='grayscale', class_mode=None, batch_size=1, shuffle=False,
        seed=1,
        interpolation='bilinear')

    test_event_generator = test_event_gen.flow_from_directory(
        'C:/Users/fbond/Desktop/New Folder/Dataset/Testing/Events/'.format(dataset),
        target_size=(320, 470),
        color_mode='grayscale', class_mode=None, batch_size=1, shuffle=True, seed=1,
        interpolation='bilinear')

    test_gt_generator = test_gt_gen.flow_from_directory(
        '/home/fahd_weian/my_workspace/DGX_Dataset/Testing/Masks/'.format(dataset),
        target_size=(320, 470),
        color_mode='grayscale', class_mode=None, batch_size=1,
        shuffle=False, seed=1,
        interpolation='nearest')

    while True:
        img_gen = test_img_generator.next()
        event_gen = test_event_generator.next()
        gt_gen = test_gt_generator.next()
        yield [img_gen, event_gen], gt_gen

Image_Size = [320, 470, 1]

Grayscale_Input = tf.keras.layers.Input(shape=Image_Size)
Grayscale_Input = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(Grayscale_Input)


Grayscale_Model = tf.keras.applications.xception.Xception(
    include_top=False, weights=None, input_tensor=Grayscale_Input,
    input_shape=(320, 470, 1), pooling=max,
)
for layer in Grayscale_Model.layers:
    layer._name = layer._name + str('_A')

output_1 = Grayscale_Model.get_layer('block2_sepconv2_bn_A').output
output_2 = Grayscale_Model.get_layer('block3_sepconv2_bn_A').output
output_3 = Grayscale_Model.get_layer('block4_sepconv2_bn_A').output
output_4 = Grayscale_Model.get_layer('block13_sepconv2_bn_A').output
output_5 = Grayscale_Model.get_layer('block14_sepconv2_bn_A').output

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
decoder_4 = tf.image.resize(decoder_4, [320, 470])
#
Grayscale_Output = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(decoder_4)

Grayscale_Model = Model(inputs=Grayscale_Model.input, outputs=Grayscale_Output)
Grayscale_Model.compile(optimizer='adam', loss=bce_jaccard_loss, metrics=[iou_score])

# Grayscale_Pred = Grayscale_Model.predict_generator(generate_generator_multiple_train(train_image_gen, train_event_gen,
#                                                                                      train_gt_gen), steps=90)

Event_Input = tf.keras.layers.Input(shape=Image_Size)
Event_Input = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(Event_Input)

Event_Model = tf.keras.applications.xception.Xception(
    include_top=False, weights=None, input_tensor=Event_Input,
    input_shape=(320, 470, 1), pooling=max,
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
decoder_4 = tf.image.resize(decoder_4, [320, 470])
#
Event_Output = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(decoder_4)


Event_Model = Model(inputs=Event_Model.input, outputs=Event_Output)
Event_Model.compile(optimizer='adam', loss=bce_jaccard_loss, metrics=[iou_score])



four_layers = tf.keras.layers.Concatenate(axis=-1)([Event_Model.input, Grayscale_Model.input, Event_Model.output, Grayscale_Model.output])

model = Model(inputs=[Event_Model.input, Grayscale_Model.input], outputs=[Event_Model.output, Grayscale_Model.output, four_layers])

final_input = model.get_layer('concatenate').input

Final_model = tf.keras.applications.xception.Xception(
    include_top=False, weights=None, input_tensor=four_layers,
    input_shape=(320, 470, 4), pooling=max,
)
for layer in Final_model.layers:
    layer._name = layer._name + str('_C')
output_1 = Final_model.get_layer('block2_sepconv2_bn_C').output
output_2 = Final_model.get_layer('block3_sepconv2_bn_C').output
output_3 = Final_model.get_layer('block4_sepconv2_bn_C').output
output_4 = Final_model.get_layer('block13_sepconv2_bn_C').output
output_5 = Final_model.get_layer('block14_sepconv2_bn_C').output

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

decoder_2 = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, activation='relu', padding='same')(decoder_1)
decoder_2 = tf.keras.layers.Conv2DTranspose(filters=728, kernel_size=3, strides=2, activation='relu',
                                            padding='same')(decoder_2)
decoder_2 = layers.BatchNormalization()(decoder_2)
decoder_2 = tf.image.resize(decoder_2, (tf.shape(output_3)[1], tf.shape(output_3)[2]))
decoder_2 = tf.concat([decoder_2, output_3], 3)

decoder_3 = tf.keras.layers.Conv2D(filters=728, kernel_size=3, activation='relu', padding='same')(decoder_2)
decoder_3 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2, activation='relu',
                                            padding='same')(decoder_3)
decoder_3 = layers.BatchNormalization()(decoder_3)
decoder_3 = tf.image.resize(decoder_3, (tf.shape(output_2)[1], tf.shape(output_2)[2]))
decoder_3 = tf.concat([decoder_3, output_2], 3)

decoder_4 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(decoder_3)
decoder_4 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, activation='relu',
                                            padding='same')( decoder_4)
decoder_4 = layers.BatchNormalization()(decoder_4)
decoder_4 = tf.image.resize(decoder_4, (tf.shape(output_1)[1], tf.shape(output_1)[2]))
decoder_4 = tf.concat([decoder_4, output_1], 3)
decoder_4 = tf.image.resize(decoder_4, [320, 470])

Final_output = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(decoder_4)

final_model = Model(inputs=Final_model.input, outputs=Final_output)
final_model.compile(optimizer='adam', loss=bce_jaccard_loss, metrics=[iou_score])
# final_model.summary()

# tf.keras.utils.plot_model(model, "multi_input_single_4_channel_output.png", show_shapes=True)



model = Model(inputs=[Event_Model.input, Grayscale_Model.input], outputs=Final_output)

final_model.compile(optimizer='adam', loss=bce_jaccard_loss, metrics=[iou_score])
final_model.summary()

checkpoint_path = "/home/fahd_weian/my_workspace/Weights Triple Xception Test/cp.ckpt"
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)



checkPoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1,
                             save_best_only=True, save_weights_only=True,
                             mode='min', period=1)
callbacks_list = [checkPoint]

final_model.fit_generator(generate_generator_multiple_train(train_image_gen, train_event_gen, train_gt_gen),
                    steps_per_epoch=90, epochs=150,
                    verbose=1,  callbacks=[callbacks_list],  # , lr_callback],
                    validation_data=generate_generator_multiple_val(val_image_gen, val_event_gen, val_gt_gen),
                    validation_steps=30, validation_freq=1, class_weight=None, max_queue_size=10, workers=1,
                    use_multiprocessing=False, shuffle=False, initial_epoch=0)
                    