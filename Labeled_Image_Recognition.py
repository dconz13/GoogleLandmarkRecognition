# All imports for ease here
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import numpy as np
import pandas as pd
import os, os.path

# init local path constants
raid_dir = '/mnt/raid0/Projects/Kaggle/GoogleLandmarkRecognition/'
raid_train_dir = '/mnt/raid0/Projects/Kaggle/GoogleLandmarkRecognition/train/'
raid_valid_dir = '/mnt/raid0/Projects/Kaggle/GoogleLandmarkRecognition/valid/'
raid_test_dir = '/mnt/raid0/Projects/Kaggle/GoogleLandmarkRetrieval/test/'
train_csv = '~/Documents/Kaggle/GoogleLandmarkRecognition/train.csv'
test_csv = '~/Documents/Kaggle/GoogleLandmarkRecognition/test.csv'

# use this function to load the train and test data
def load_dataset(path):
    data = load_files(path)
    image_files = np.array(data['filenames'])
    train_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return image_files, train_targets

# point this to the train.csv file
def load_variable_names(path):
    # use dtype=None because we have strings and ints
    data = pd.read_csv(path, quotechar='"')
    return data

# use this to count the images available to learn from
def get_total_files(path):
    return len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])

# Need to test and configure this code!!
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        raid_train_dir,
        target_size=(224, 224),
        batch_size=500,
        class_mode='categorical')

valid_generator = test_datagen.flow_from_directory(
        raid_valid_dir,
        target_size=(224,224),
        batch_size=500,
        class_mode='categorical')

from keras.layers import Dropout, Flatten, Dense, Activation
from keras.models import Sequential

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions


def create_model():
    model = Sequential()
    model.add(Flatten(input_shape=model.output_shape[1:]))
    model.add(Dense(14951, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='softmax'))
    return model

def run_resnet(): # Only run this when you need bottleneck features generated. Takes about 5 hours.
    bottleneck_features = resnet.predict_generator(train_generator, verbose=1)
    # save the output as a Numpy array
    np.save(open('bottleneck_features/bottleneck_features_train.npy', 'wb'), bottleneck_features)

    bottleneck_features = resnet.predict_generator(valid_generator, verbose=1)
    # save the output as a Numpy array
    np.save(open('bottleneck_features/bottleneck_features_valid.npy', 'wb'), bottleneck_features)
    


#from keras.callbacks import ModelCheckpoint 

#resnet = ResNet50(include_top=False, weights='imagenet')
model = create_model()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])



