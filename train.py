import tensorflow as tf
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from data import *

from sklearn.preprocessing import OneHotEncoder


threshold, upper, lower = 0.2, 1, 0
x_train = np.where(X_train>threshold, upper, lower)
x_test = np.where(X_test>threshold, upper, lower)

x_train = x_train.reshape(x_train.shape[0], 1, 16,16,16)
x_test = x_test.reshape(x_test.shape[0], 1, 16,16,16)
onehot_encoder = OneHotEncoder(sparse=False)
y_train_onehot = onehot_encoder.fit_transform(y_train.reshape(len(y_train), 1))
y_test_onehot = onehot_encoder.transform(y_test.reshape(len(y_test), 1))

print("x_train", X_train.shape)
print("y_train", y_train.shape)
print("y_train_onehot", y_train_onehot.shape)

print("x_test", X_test.shape)
print("y_test", y_test.shape)
print("y_test_onehot", y_test_onehot.shape)

model = Sequential([
    layers.Conv3D(8, (3, 3, 3), activation='relu', padding='same', input_shape=(1, 16, 16, 16)),
    layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same'),
    layers.MaxPooling3D((2, 2, 2), padding='same'),

    layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
    layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
    layers.MaxPooling3D((2, 2, 2), padding='same'),

    layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling3D(),
    layers.Flatten(),

    layers.Dense(units=1024, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(units=256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(units=10, activation='softmax'),
])

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['acc'])
model.summary()


model.fit(x=x_train, y=y_train_onehot, batch_size=128, epochs=50, validation_split=0.2)

print(model.evaluate(x=x_test, y=y_test_onehot))