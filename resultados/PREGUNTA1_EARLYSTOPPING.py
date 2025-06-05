
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from glob import glob

image_size = 128
input_shape = (image_size, image_size, 1)
latent_dim = 1024
kernel_size = 3
layer_filters = [128, 256, 512, 1024]

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def load_dataset(directory):
    images_color, images_gray = [], []
    for filename in sorted(glob(os.path.join(directory, "*.jpg"))):
        img_color = load_img(filename, target_size=(image_size, image_size))
        img_color = img_to_array(img_color) / 255.0
        img_gray = rgb2gray(img_color)
        img_gray = np.expand_dims(img_gray, axis=-1)
        images_color.append(img_color)
        images_gray.append(img_gray)
    return np.array(images_gray), np.array(images_color)

ruta_base = "/home/cursos/ima543_2025_1/ima543_pmonsalve/dataset_flores"
x_train_gray, x_train_color = load_dataset(os.path.join(ruta_base, "train"))
x_test_gray, x_test_color = load_dataset(os.path.join(ruta_base, "test"))

inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
for filters in layer_filters:
    x = Conv2D(filters, kernel_size, strides=2, activation='relu', padding='same')(x)
shape = tf.keras.backend.int_shape(x)
x = Flatten()(x)
latent = Dense(latent_dim, name='latent_vector')(x)
encoder = Model(inputs, latent, name='encoder')

latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)
for filters in layer_filters[::-1]:
    x = Conv2DTranspose(filters, kernel_size, strides=2, activation='relu', padding='same')(x)
outputs = Conv2DTranspose(3, kernel_size, activation='sigmoid', padding='same', name='decoder_output')(x)
decoder = Model(latent_inputs, outputs, name='decoder')

autoencoder_input = Input(shape=input_shape, name='autoencoder_input')
autoencoder = Model(autoencoder_input, decoder(encoder(autoencoder_input)), name='autoencoder')

def custom_loss(alpha=0.5, beta=0.5):
    def loss(y_true, y_pred):
        mse_rgb = tf.reduce_mean(tf.square(y_true - y_pred))
        y_pred_gray = tf.image.rgb_to_grayscale(y_pred)
        y_true_gray = tf.image.rgb_to_grayscale(y_true)
        mse_gray = tf.reduce_mean(tf.square(y_true_gray - y_pred_gray))
        return alpha * mse_rgb + beta * mse_gray
    return loss

autoencoder.compile(optimizer=Adam(), loss=custom_loss())

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

autoencoder.fit(x_train_gray, x_train_color,
                validation_data=(x_test_gray, x_test_color),
                epochs=100,
                batch_size=16,
                callbacks=[early_stop])

autoencoder.save("/home/cursos/ima543_2025_1/ima543_pmonsalve/saved_models/autoencoder_flores_early.keras")
