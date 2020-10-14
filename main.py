import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, Concatenate
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, BatchNormalization, Layer, RepeatVector, Permute
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave

import numpy as np
import os
import random

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Get training images
X = []
for filename in os.listdir("Train/"):
    X.append(img_to_array(load_img("Train/"+filename)))
X = np.array(X, dtype=float)
Xtrain = X/255.0

# get test images
Xt = []
for filename in os.listdir("Test/"):
    Xt.append(img_to_array(load_img("Test/"+filename)))
Xt = np.array(Xt, dtype=float)
Xtest = Xt/255.0

inception = InceptionResNetV2(weights="imagenet", include_top=True)
# inception.graph = tf.Graph()
# print(inception.summary())

# build model

embed_input = Input(shape=(1000,)) # this is the output of the inception resnet model
# Encoder
encoder_input = Input(shape=(256,256,1,))
encoder_output = Conv2D(64,(3,3), activation="relu", padding="same", strides=2)(encoder_input)
encoder_output = Conv2D(128,(3,3), activation="relu", padding="same")(encoder_output)
encoder_output = Conv2D(128,(3,3), activation="relu", padding="same", strides=2)(encoder_output)
encoder_output = Conv2D(256,(3,3), activation="relu", padding="same")(encoder_output)
encoder_output = Conv2D(256,(3,3), activation="relu", padding="same", strides=2)(encoder_output)
encoder_output = Conv2D(512,(3,3), activation="relu", padding="same")(encoder_output)
encoder_output = Conv2D(512,(3,3), activation="relu", padding="same", strides=2)(encoder_output)
encoder_output = Conv2D(256,(3,3), activation="relu", padding="same", strides=2)(encoder_output)
# Fusion
fusion_output = RepeatVector(32 * 32)(embed_input)
fusion_output = Reshape(([32,32,1000]))(fusion_output)
fusion_output = Concatenate(axis=3)([encoder_output, fusion_output])
fusion_output = Conv2D(256, (1,1), activation="relu", padding="same")(fusion_output)

# Decoder
decoder_output = Conv2D(128, (3,3), activation="relu", padding="same")(fusion_output)
decoder_output = UpSampling2D((2,2))(decoder_output)
decoder_output = Conv2D(64, (3,3), activation="relu", padding="same")(decoder_output)
decoder_output = UpSampling2D((2,2))(decoder_output)
decoder_output = Conv2D(32, (3,3), activation="relu", padding="same")(decoder_output)
decoder_output = Conv2D(16, (3,3), activation="relu", padding="same")(decoder_output)
decoder_output = Conv2D(2, (3,3), activation="relu", padding="same")(decoder_output)
decoder_output = UpSampling2D((2,2))(decoder_output)

model = Model([encoder_input, embed_input], decoder_output)

# @tf.function
def create_inception_embedding(grayscaled_rgb):
    grayscaled_rgb_resized = []
    for i in grayscaled_rgb:
        i = resize(i, (299,299,3), mode="constant")
        grayscaled_rgb_resized.append(i)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    # with inception.graph.as_default():
    #     embed = inception.predict(grayscaled_rgb_resized)
    embed = inception.predict(grayscaled_rgb_resized)
    return embed

# image transformer
datagen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, rotation_range=20, horizontal_flip=True)

# generate training data
batch_size = 10

def image_a_b_gen(batch_size, img_data):
    for batch in datagen.flow(img_data, batch_size=batch_size):
        grayscaled_rgb = gray2rgb(rgb2gray(batch))
        embed = create_inception_embedding(grayscaled_rgb)
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        X_batch = X_batch.reshape(X_batch.shape + (1,))
        Y_batch = lab_batch[:,:,:,1:] / 128.0
        yield ([X_batch, embed], Y_batch)

# train model
model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])
# tf.keras.utils.plot_model(model, "full_model.png", show_shapes=True)
model_filepath = "models/image-colouriser-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath=model_filepath, monitor="val_accuracy", verbose=1, save_best_only=True, save_weights_only=False, mode="auto")
# early = EarlyStopping(monitor="val_accuracy", min_delta=0, patience=40, verbose=1, mode="auto")

# model.fit(image_a_b_gen(batch_size, Xtrain), epochs=3, steps_per_epoch=1, validation_data=image_a_b_gen(batch_size, Xtest),
#           validation_steps=1, callbacks=[checkpoint])

model.fit(image_a_b_gen(batch_size, Xtrain), epochs=3, steps_per_epoch=930, callbacks=[checkpoint], verbose=2)


color_me = []
for filename in os.listdir("Convert/"):
    color_me.append(img_to_array(load_img("Convert/" + filename)))
color_me = np.array(color_me, dtype=float)
gray_me = gray2rgb(rgb2gray(color_me/255.0))
color_me_embed = create_inception_embedding(gray_me)
color_me = rgb2lab(color_me/255.0)[:,:,:,0]
color_me = color_me.reshape(color_me.shape+(1,))

# test model
output = model.predict([color_me, color_me_embed])
output = output * 128

# output colourisations
for i in range(len(output)):
    cur = np.zeros((256,256,3))
    cur[:,:,0] = color_me[i][:,:,0]
    cur[:,:,1:] = output[i]
    imsave("result/img_"+str(i)+".png", (lab2rgb(cur)*255).astype(np.uint8))


# path = os.listdir("Train")
# train_datagen = ImageDataGenerator(rescale=1/255.0)
# traindata = train_datagen.flow_from_directory(path, target_size=(224,244), batch_size=32, class_mode=None)
# vggmodel = VGG16(weights="imagenet", include_top=True)
# # print(vggmodel.summary())
#
# for layers in (vggmodel.layers)[:19]:
#     # print(layers)
#     layers.trainable = False
#
# X = vggmodel.layers[-2].output
# predictions = Dense(2, activation="softmax")(X)
# model_final = Model(vggmodel.input, predictions)
#
# model_final.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
# print(model_final.summary())
#
# checkpoint = ModelCheckpoint("vgg16_1.h5", monitor="val_accuracy", verbose=1, save_best_only=True, save_weights_only=False, mode="auto", period=1)
# early = EarlyStopping(monitor="val_accuracy", min_delta=0, patience=40, verbose=1, mode="auto")
#
# model_final.fit(traindata, steps_per_epoch=2, epochs=100, validation_data=testdata, validation_steps=1, callbacks=[checkpoint, early])
# model_final.save_weights("vgg16_1.h5")