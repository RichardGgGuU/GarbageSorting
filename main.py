import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def processing_data(data_path):
    train_data = ImageDataGenerator(
        rescale=1. / 225,
        shear_range=0.1,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.1
    )

    validation_data = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.1
    )

    train_generator = train_data.flow_from_directory(
        data_path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='categorical',
        subset='training',
        seed=0
    )
    validation_generator = validation_data.flow_from_directory(
        data_path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='categorical',
        subset='validation',
        seed=0
    )

    return train_generator, validation_generator


def train_model(train_generator, validation_generator, save_model_path):
    vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(6, activation='softmax'))

    model = Sequential()
    model.add(vgg16_model)
    model.add(top_model)

    model.compile(
        optimizer=SGD(learning_rate=1e-3, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        x=train_generator,
        epochs=100,
        steps_per_epoch=2259 // 16,
        validation_data=validation_generator,
        validation_steps=248 // 16
    )

    model.save(save_model_path)

    return model, history


def evaluate_model(validation_generator, save_model_path):
    model = load_model(save_model_path)
    loss, accuracy = model.evaluate_generator(validation_generator)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))


def plot_history(history):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


def main():
    data_path = "RubbishData_full/data_full"
    save_model_path = 'model3.h5'
    train_generator, validation_generator = processing_data(data_path)
    model, history = train_model(train_generator, validation_generator, save_model_path)
    evaluate_model(validation_generator, save_model_path)
    plot_history(history)


if __name__ == '__main__':
    main()
