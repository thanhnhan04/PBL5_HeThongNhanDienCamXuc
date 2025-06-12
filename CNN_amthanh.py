import os
import numpy as np
import tensorflow as tf
import librosa
from sklearn.utils import shuffle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dropout,
                                     BatchNormalization, GlobalAveragePooling2D, Dense)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ============================ CONFIG
SAMPLE_RATE = 16000
DURATION = 3  # giây
NUM_MFCC = 40
AUDIO_LENGTH = SAMPLE_RATE * DURATION  # 3s * 16000 = 48000 samples
NUM_CLASSES = 7
BATCH_SIZE = 32
EPOCHS = 150
MODEL_PATH = "E:/Python/PBL5/models/audio_emotion_audio5.keras"
DATA_DIR = "E:/Python/PBL5/data_audio_limited_split"

def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    if len(y) < AUDIO_LENGTH:
        y = np.pad(y, (0, AUDIO_LENGTH - len(y)))
    else:
        y = y[:AUDIO_LENGTH]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=NUM_MFCC)
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)  
    mfcc = mfcc[..., np.newaxis] 
    return mfcc


def load_dataset(split):
    X, y = [], []
    labels = sorted(os.listdir(os.path.join(DATA_DIR, split)))
    label2idx = {label: i for i, label in enumerate(labels)}

    for label in labels:
        folder = os.path.join(DATA_DIR, split, label)
        for file in os.listdir(folder):
            if file.endswith(".wav"):
                mfcc = preprocess_audio(os.path.join(folder, file))
                X.append(mfcc)
                y.append(label2idx[label])
    
    X, y = np.array(X), tf.keras.utils.to_categorical(y, num_classes=NUM_CLASSES)
    return shuffle(X, y)

X_train, y_train = load_dataset("train")
X_val, y_val = load_dataset("val")
X_test, y_test = load_dataset("test")

from tensorflow.keras.layers import LeakyReLU, Flatten

def build_audio_cnn_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.4)(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)
    x = Dense(512)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)


input_shape = X_train.shape[1:]  # (40, time_steps, 1)
model = build_audio_cnn_model(input_shape, NUM_CLASSES)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)


model.summary()


checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)


history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stopping, lr_scheduler]
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"🎯 Test Accuracy: {test_acc * 100:.2f}%")

plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
