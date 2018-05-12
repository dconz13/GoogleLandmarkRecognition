# All imports for ease here
#from sklearn.datasets import load_files
#from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import numpy as np
import pandas as pd
import os, os.path
import matplotlib.pyplot as plt
import cv2
import pickle
import datetime

# init local path constants
raid_dir = '/mnt/raid0/Projects/Kaggle/GoogleLandmarkRecognition/'
raid_train_dir = '/mnt/raid0/Projects/Kaggle/GoogleLandmarkRecognition/train/'
raid_valid_dir = '/mnt/raid0/Projects/Kaggle/GoogleLandmarkRecognition/valid/'
raid_test_dir = '/mnt/raid0/Projects/Kaggle/GoogleLandmarkRetrieval/test/'
train_csv = '~/Documents/Kaggle/GoogleLandmarkRecognition/train.csv'
test_csv = '~/Documents/Kaggle/GoogleLandmarkRecognition/test.csv'

# Constants
#num_classes = 14951
batch_size = 500
epochs = 2000


# use this function to load the train and test data
def load_dataset(path):
    targets = os.listdir(path)
    labels = np_utils.to_categorical(targets, len(targets))
    return labels
    
# point this to the train.csv file
def load_variable_names(path):
    # use dtype=None because we have strings and ints
    data = pd.read_csv(path, quotechar='"')
    return data

# use this to count the images available to learn from
def get_total_files(path):
    return len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])

# use to clone the missing dir names from train into valid
def create_valid_dirs():
    dirs = os.listdir(raid_train_dir)
    #print(len(dirs))
    #test = []
    for dir in dirs:
        if not os.path.exists(os.path.join(raid_valid_dir, dir)):
            #test.append(dir)
            os.makedirs(os.path.join(raid_valid_dir, dir))
    #print(test)
    #print(len(test))
    dirs2 = os.listdir(raid_valid_dir)
    print(len(dirs2))

# Need to test and configure this code!!
from keras.preprocessing.image import ImageDataGenerator


from keras.layers import Dropout, Flatten, Dense, Activation
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions


def run_resnet(): # Only run this when you need bottleneck features generated. Takes about 5 hours.
    datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_directory(
        raid_train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False)

    valid_generator = datagen.flow_from_directory(
        raid_valid_dir,
        target_size=(224,224),
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False)

    resnet = ResNet50(include_top=False, weights='imagenet')
    bottleneck_features = resnet.predict_generator(train_generator, verbose=1)
    # save the output as a Numpy array
    np.save(open('bottleneck_features/bottleneck_features_train.npy', 'wb'), bottleneck_features)
    print("Saved training bottleneck!")
    
    bottleneck_features = resnet.predict_generator(valid_generator, verbose=1)
    # save the output as a Numpy array
    np.save(open('bottleneck_features/bottleneck_features_valid.npy', 'wb'), bottleneck_features)
    print("Saved valid bottleneck!")

def run_top_model():
    # Build generator to get class labels and number of classes
    datagen_top = ImageDataGenerator(rescale=1./255)
    generator_top = datagen_top.flow_from_directory(
        raid_train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False)

    #train_samples = len(generator_top.filenames)
    num_classes = len(generator_top.class_indices)
    
    # Load training data from resnet-50
    train_data = np.load(open('bottleneck_features/bottleneck_features_train.npy', 'rb'))
    # Load training class labels
    train_labels = generator_top.classes

    generator_top = datagen_top.flow_from_directory(
        raid_valid_dir,
        target_size=(224,224),
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False)

    valid_data = np.load(open('bottleneck_features/bottleneck_features_valid.npy', 'rb'))

    valid_labels = generator_top.classes

    print("Creating model..")
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
#    model.add(Dense(num_classes, activation='sigmoid'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
   
    model.summary()

    print("creating checkpointer...")
    checkpointer = ModelCheckpoint(filepath='weights/checkpointer.hdf5', verbose=1, save_best_only=True)

    print("Fitting model...")

    history = model.fit(train_data, train_labels,
          epochs=epochs,
          batch_size=batch_size,
          verbose=1,
          validation_data=(valid_data, valid_labels),
          callbacks=[checkpointer])
    #model.save_weights('weights/first_try.h5')
    (eval_loss, eval_accuracy) = model.evaluate(
             valid_data, valid_labels, batch_size=batch_size, verbose=1)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))
    
    # uncomment when running x11 server
    #plt_history(history)
    with open('history/fit_history.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


def predict_test_set():
    datagen_top = ImageDataGenerator(rescale=1./255)
    generator_top = datagen_top.flow_from_directory(
        raid_test_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    #resnet = ResNet50(include_top=False, weights='imagenet')
    #bottleneck_features = resnet.predict_generator(generator_top, verbose=1)
    #np.save(open('bottleneck_features/bottleneck_features_test.npy', 'wb'), bottleneck_features)

    bottleneck_features = np.load(open('bottleneck_features/bottleneck_features_test.npy', 'rb'))
    
    #orig_model = load_model('weights/first_try.hdf5')
    #orig_model.save_weights('weights/extracted_weights.h5')
    #del orig_model

    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_features.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(14951, activation='softmax'))
    model.load_weights('weights/extracted_weights.h5')
    #model.compile(loss='sparse_categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    #model.summary()
    
    predict_array = model.predict(bottleneck_features, batch_size=batch_size,  verbose=1)
    print(predict_array)
    np.save('predict_array.npy', predict_array)
    
def plt_history(history):
    plt.figure(1)

     # summarize history for accuracy

    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

     # summarize history for loss

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()  

start_time = datetime.datetime.now()
    
#create_valid_dirs()
#run_resnet()
#run_top_model()
predict_test_set()

end_time = datetime.datetime.now()
delta = end_time - start_time
print("[INFO] runtime: {}".format(delta))

# Record of runs
# 4.42% accuracy on 20 epochs
