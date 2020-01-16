import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

Dig_MNIST = pd.read_csv("../input/Kannada-MNIST/Dig-MNIST.csv")
sample_submission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")
test = pd.read_csv("../input/Kannada-MNIST/test.csv")
train = pd.read_csv("../input/Kannada-MNIST/train.csv")

test_data = test.iloc[:, 1:].values #MAIN TEST FOR SUBMISSION

X_train = train.iloc[:, 1:].values
y_train = train.iloc[:, 0].values
sns_y_train = y_train

test_data = test_data.astype("float32")/255

X_train = X_train.astype("float32")/255
y_train = to_categorical(y_train, num_classes=10)

test_data = np.array(test_data).reshape(-1, 28, 28, 1)
X_train = np.array(X_train).reshape(-1, 28, 28 ,1)

test_data = test_data.astype("float32")

print("test data: ", test_data.shape)
print("train data: ", X_train.shape)
print("train labels: ", y_train.shape)

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train, test_size = 0.15)

nets = 4
model = [0] * nets
    
for i in range(nets):
    
    model[i] = Sequential()

    model[i].add(Conv2D(64, kernel_size=3, padding="same", activation="relu", input_shape=(28,28,1)))
    model[i].add(BatchNormalization())
    model[i].add(Conv2D(64, kernel_size=3, padding="same", activation="relu"))
    model[i].add(MaxPooling2D(pool_size=(2,2)))
    model[i].add(BatchNormalization())
    model[i].add(Dropout(.3))

    model[i].add(Conv2D(128, kernel_size=3, padding="same", activation="relu"))
    model[i].add(BatchNormalization())
    model[i].add(Conv2D(128, kernel_size=3, padding="same", activation="relu"))
    model[i].add(BatchNormalization())
    model[i].add(Conv2D(128, kernel_size=3, padding="same", activation="relu"))
    model[i].add(MaxPooling2D(pool_size=(2,2)))
    model[i].add(BatchNormalization())
    model[i].add(Dropout(.3))

    model[i].add(Conv2D(256, kernel_size=3, padding="same", activation="relu"))
    model[i].add(BatchNormalization())
    model[i].add(Conv2D(256, kernel_size=3, padding="same", activation="relu"))
    model[i].add(BatchNormalization())
    model[i].add(Conv2D(256, kernel_size=3, padding="same", activation="relu"))
    model[i].add(MaxPooling2D(pool_size=(2,2)))
    model[i].add(BatchNormalization())
    model[i].add(Dropout(.3))

    model[i].add(Conv2D(512, kernel_size=3, padding="same", activation="relu"))
    model[i].add(BatchNormalization())
    model[i].add(Conv2D(512, kernel_size=3, padding="same", activation="relu"))
    model[i].add(BatchNormalization())
    model[i].add(Conv2D(512, kernel_size=3, padding="same", activation="relu"))
    model[i].add(MaxPooling2D(pool_size=(2,2)))
    model[i].add(BatchNormalization())
    model[i].add(Dropout(.4))

    model[i].add(Flatten())
    model[i].add(Dense(512))
    model[i].add(BatchNormalization())
    model[i].add(Dense(10, activation="softmax"))

    model[i].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

datagen = ImageDataGenerator(
        rotation_range=12,
        zoom_range=.25,
        width_shift_range=0.25,
        height_shift_range=0.25,
        data_format="channels_last",
        shear_range=15)

datagen.fit(X_train1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25,
                              patience=5, min_lr=0.00001)

batch_size = 32
epochs = 70

for i in range(nets):
    model[i].fit_generator(datagen.flow(X_train1, y_train1, batch_size=batch_size), steps_per_epoch=test_data.shape[0]//batch_size, 
                           epochs = epochs, validation_data=(X_test1, y_test1), callbacks=[reduce_lr])


results = np.zeros((test_data.shape[0], 10))

for i in range(nets):
    results += model[i].predict(test_data)
    
results = np.argmax(results, axis=1)

submission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")
submission["label"] = results
submission.to_csv("submission.csv", index=False)
