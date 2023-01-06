#Here we import all our dependencies

import time
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from sklearn.model_selection import train_test_split
import datetime
import matplotlib.pyplot as plt

SOURCE_FOLDER = 'all genres 30 sec/'
LABELS = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'qawwali', 'reggae', 'rock']
#A list of the categories we will be training our model on
NUM_FILES = 100 #Number of source files per genre
LEN_FILES = 30 #duration in seconds of each file
LEN_DEST = 30# Length of the destination files in seconds
TEST_PROPORTION = 0.25 #The proportion of the total dataset our test set will have
VAL_PROPORTION = 0.25 #The proportion of the dataset except for the test set our validation set will consist of
NUM_EPOCHS = 2 #The number of epochs we will be training our model
N_BATCH_SIZE = 32 #The batch size
MODEL_NAME = 'Yamnet Reduced Randomization Classifier' #This is the name of the file to which the model will be saved
BASE = 0 #The starting numeral part of the file name
RAND_RUNS = 1 # The amount of times we train the model
num_genres = len(LABELS)

start = time.time()
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)


def load_wav_16k_mono(filename):
    """ It loads a .wav file, converts it to a float tensor, resamples it to a 16 kHz single-channel audio and then returns it. This must be done
    because it is the only format supported by the YAMNet model"""
    
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents,desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

def features_extractor_iter(names,count):
    '''
    This function extracts features from all of the given files and returns them in the form of the list X, while it also returns all of the labels in
    the list y. Here the parameter names is a list of all the categories one wants, while count is the amount of files for each category from which one
    wishes to extract features. If none is given, it is assumed that it
    is in the same directory as the code.
    '''
    X = []
    y = []
    for name in names:
        for j in range(count):
            if BASE+j < 10:
                zeros = '0000'
            elif BASE+j < 100:
                zeros = '000'
            else:
                zeros = '00'
            file_name = SOURCE_FOLDER + name + '.' + zeros + str(j) + '.wav'
            data = load_wav_16k_mono(file_name)
            X.append(data)
            y.append(name + '.' + str(j))
    return X,y


def extract_embedding(wav_data):
  '''It runs YAMNet to extract embeddings from the wav data, which it then returns''' 
  scores, embeddings, spectrogram = yamnet_model(wav_data)
  return embeddings

def predict(data):
        '''
            It takes as an arguement a file path, and then returns its classification with the help of model.
        '''
        predicted_label = model.predict(data.reshape(1,step_size),verbose = 0)
        classes_x = int(np.argmax(predicted_label,axis=1))
        prediction_class = LABELS[classes_x]
        return prediction_class
    
def plot_history(history):
        fig, axs = plt.subplots(2)

        axs[0].plot(history.history["accuracy"], label='train accuracy')
        axs[0].plot(history.history["val_accuracy"], label = 'validation accuracy')
        axs[0].set_ylabel("Accuracy")
        axs[0].legend(loc="lower right")
        axs[0].set_title("Accuracy eval")

        axs[1].plot(history.history["loss"], label='train error')
        axs[1].plot(history.history["val_loss"], label = 'validation error')
        axs[1].set_ylabel("Error")
        axs[1].set_xlabel("Epoch")
        axs[1].legend(loc="upper right")
        axs[1].set_title("Error eval")

        plt.show()
        
accuracy = []
results = []
for runs in range(RAND_RUNS):
    X,y = features_extractor_iter(LABELS,NUM_FILES)
    x = []
    #This for loop makes sure that all of the features extracted for each audio file are of the same length
    for char in X:
      e = extract_embedding(char)
      e = e[0:2 * LEN_FILES]
      x.append(np.array(e))
      
    #This reshapes x into an acceptable shape.
    x = np.array(x)
    x = x.reshape(num_genres * NUM_FILES,1024 * LEN_FILES * 2)
    #This line splits up x and y to get the train and test sets
    X_tr,X_te,y_tr,y_te = train_test_split(x,y,test_size = TEST_PROPORTION)
    X_tr,X_v,y_tr,y_v = train_test_split(X_tr,y_tr,test_size = VAL_PROPORTION)

    X_train = []
    step_size = (1024 * LEN_FILES * 2) // (LEN_FILES // LEN_DEST)
    #These loops seperate the original data taken from the source files to that of the destination files.
    for char in X_tr:
        start = 0
        for num in range(LEN_FILES // LEN_DEST):
            X_train.append(char[start * step_size:(start + 1) * step_size])
            start = start + 1

    X_val = []
    for char in X_v:
        start = 0
        for num in range(LEN_FILES // LEN_DEST):
            X_val.append(char[start * step_size:(start + 1) * step_size])
            start = start + 1

    X_test = []        
    for char in X_te:
        start = 0
        for num in range(LEN_FILES // LEN_DEST):
            X_test.append(char[start * step_size:(start + 1) * step_size])
            start = start + 1

    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)
    #These loops create the arrays y_train, y_val and y_test, which contain all the names.

    y_train = []
    for char in y_tr:
        y_train.append([char] * (LEN_FILES // LEN_DEST))
    y_train = np.array(y_train).reshape(-1)

    y_val = []
    for char in y_v:
        y_val.append([char] * (LEN_FILES // LEN_DEST))
    y_val = np.array(y_val).reshape(-1)

    y_test = []
    for char in y_te:
        y_test.append([char] * (LEN_FILES // LEN_DEST))
    y_test = np.array(y_test).reshape(-1)

    #The classification model. It is very small because YAMNet is doing all of the work.
    model = Sequential()
    model.add(Dense(num_genres, input_shape=(step_size,)))
    model.add(Activation('softmax'))



    model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
    from tensorflow.keras.callbacks import ModelCheckpoint

    checkpointer = ModelCheckpoint(filepath='model.hdf5', save_weights_only=True,verbose=1)
    from tensorflow.keras.callbacks import ModelCheckpoint

    #These loops create the final labels by taking out the labels from the filenames, e.g 'blues' from 'blues.34'.
    y_model_train = []
    for char in y_train:
        c = char.split('.')[0]
        y_model_train.append(c)
    y_model_val = []
    for char in y_val:
        c = char.split('.')[0]
    y_model_val.append(c)

    y_model_test = []
    for char in y_test:
        c = char.split('.')[0]
        y_model_test.append(c)

    #This converts them to binary arrays
    labelencoder = LabelEncoder()
    y_model_train = to_categorical(labelencoder.fit_transform(y_model_train))
    y_model_val = to_categorical(labelencoder.fit_transform(y_model_val))
    y_model_test = to_categorical(labelencoder.fit_transform(y_model_test))

    history = model.fit(X_train, y_model_train, batch_size = NUM_BATCH_SIZE, epochs = NUM_EPOCHS, validation_data=(X_val, y_model_val), callbacks=checkpointer, verbose=0)
    model.save(MODEL_NAME)

    wrong = [] #wrong is a list of all the files it got wrong.


    for index in range(len(X_test)):
        c = y_test[index].split('.')[0]
        if predict(X_test[index]) != c:
            wrong.append(y_test[index])
            
    print('These are all the files it got wrong for this run:')
    print('\n')
    print(wrong)

    train_accuracy = model.evaluate(X_train,y_model_train,verbose=0)
    print("The training accuracy for this run is ", train_accuracy[1])

    validation_accuracy = model.evaluate(X_val,y_model_val,verbose=0)
    print("The validation accuracy for this run is ", validation_accuracy[1])

    test_accuracy = model.evaluate(X_test, y_model_test, verbose=0)
    print("The test accuracy for this run is ", test_accuracy[1])
    
    end = time.time()
    print('It took us in total ' + str(end - start) + ' seconds to complete this run.')
    accuracy = accuracy + [[train_accuracy[1], validation_accuracy[1], test_accuracy[1]]]
    
    y_prediction = model.predict(X_test)
    y_prediction = to_categorical(np.argmax(y_prediction, axis=1), num_genres)
    #Creates confusion matrix and normalizes it over predicted (columns)
    result = confusion_matrix(
        y_test.argmax(axis=1), y_prediction.argmax(axis=1))
    results.append(result)
plot_history(history)

total = results[0]
for i in range(1,RAND_RUNS):
    total = total + results[i]
result = total/RAND_RUNS
print(result.sum(axis=0))
df_cm = pd.DataFrame(result, index = [i for i in "BLCDHJMPQRO"],
                  columns = [i for i in "BLCDHJMPQRO"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.show()

