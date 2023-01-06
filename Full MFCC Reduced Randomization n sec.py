import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPool2D,BatchNormalization,Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


from datetime import datetime


MODEL_NAME = 'Full MFCC Reduced Randomization Classifier'#This is the name of the file to which the model will be saved
N_FILES = 100 #The amount of files per each genre
N_MFCC = 30 #The amount of MFCCs we will extract
LIST_CATEGORIES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz','metal', 'pop','qawwali', 'reggae', 'rock']
#A list of the categories we will be training our model on
N_EPOCHS = 2 #The number of epochs we will be training our model
N_BATCH_SIZE = 32 #The batch size
TEST_PROPORTION = 0.15 #The proportion of the total dataset our test set will have
VAL_PROPORTION = 0.15 #The proportion of the dataset except for the test set our validation set will consist of
DROP_OUT = 0.5 #The dropout regulation
SOURCE_FOLDER = 'all genres 30 sec/' #The filepath to the folder where the source files are located
BASE = 0 #The starting numeral part of the file name
KERNEL_REGULARIZER = 0.0001
RAND_RUNS = 1 #The amount of times we train the model
LEN_FILES = 30 # length of source files in seconds
LEN_DEST = 30 # Length of destination files in seconds
FRAME_EDGE = 5 # number of frames in each 1 sec seem to vary by at max. 3. Conservatively we set it to 5. 
n_frames = 0 # The number of frames by which we cut down the number of frames to account for variations in file length
#It starts off as zero so that we can reset it later on
n_genres = len(LIST_CATEGORIES) #The number of genres


def features_extractor(file):
    '''
        It extracts MFCC features from a file, given the filename, a string. It returns an array of arrays, with each
        array consisting of the MFCC features of a LEN_DEST long section of the file
    '''
    global n_frames
    total_mfccs = []
    whole_audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
    start = 0
    for num in range(LEN_FILES // LEN_DEST):
        audio = whole_audio[start * sample_rate:(start + LEN_DEST) * sample_rate]
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
        if (n_frames == 0): #to ensure that n_frames is only set once
            n_frames = mfccs_features.shape[1] - FRAME_EDGE
        total_mfccs.append(mfccs_features[:,:n_frames])#cut down data to n_frames each
        start = start + LEN_DEST
    return np.array(total_mfccs)

def features_extractor_iter(y):
    '''
        It iterates over all of the given files, returning an array X consisting of the features extracted from each file
    '''
    X = []
    global n_frames
    for char in y:
        file_name = SOURCE_FOLDER + char + '.wav'
        data = features_extractor(file_name)
        X.append(data)
    return np.array(X).reshape(len(y) * (LEN_FILES // LEN_DEST),N_MFCC * n_frames)


def plot_history(history):
    '''
        Plots the history of a model. The first graph plots the training and validation accuracies while the second
        plots the training and validation errors as a function of the epochs.
    '''
    fig, axs = plt.subplots(2)

    axs[0].plot(history.history["accuracy"], label='train accuracy')
    axs[0].plot(history.history["val_accuracy"], label = 'validation accuracy')
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")
    plt.subplots_adjust(hspace=0.4)
    axs[1].plot(history.history["loss"], label='train error')
    axs[1].plot(history.history["val_loss"], label = 'validation error')
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()
def make_model():
    '''
        It makes and returns a deep learning model.
    '''
    model = Sequential()
    #First layer
    model.add(Dense(2048,input_shape=(N_MFCC * n_frames,), kernel_regularizer=regularizers.l2(KERNEL_REGULARIZER)))
    model.add(Activation('relu'))
    model.add(Dropout(DROP_OUT))
    #Second layer
    model.add(Dense(1024, kernel_regularizer=regularizers.l2(KERNEL_REGULARIZER)))    
    model.add(Activation('relu'))
    model.add(Dropout(DROP_OUT))
    # Third layer
    model.add(Dense(256, kernel_regularizer=regularizers.l2(KERNEL_REGULARIZER)))    
    model.add(Activation('relu'))
    model.add(Dropout(DROP_OUT))
    #Fourth layer
    model.add(Dense(64, kernel_regularizer=regularizers.l2(KERNEL_REGULARIZER)))
    model.add(Activation('relu'))
    model.add(Dropout(DROP_OUT))
    #Final layer
    model.add(Dense(n_genres, kernel_regularizer=regularizers.l2(KERNEL_REGULARIZER)))
    model.add(Activation('softmax'))
    return model
    
# This creates the list label, which contains all the filenames used. The variable zeros is used to accomodate
# the GTZAN naming convention
label = []
for name in LIST_CATEGORIES:
    for j in range(N_FILES-1,0,-1):
        if BASE+j < 10:
            zeros = '0000'
        elif BASE+j < 100:
            zeros = '000'
        else:
            zeros = '00'
        label.append(name + '.' + zeros + str(BASE + j))
    

accuracy = []
results = []
for runs in range(RAND_RUNS):
    train_label, test_label = train_test_split(label,test_size = TEST_PROPORTION)
    train_label, validation_label = train_test_split(train_label,test_size = VAL_PROPORTION)

    X_train = features_extractor_iter(train_label)
    X_validation = features_extractor_iter(validation_label)
    X_test = features_extractor_iter(test_label)
    #These create y_train, y_validation and y_test from labels, which are arrays of the classes of each file used
    #in the train, validation and test sets respectively
    y_train = []
    for char in train_label:
        c = char.split('.')[0]
        y_train.append([c] * (LEN_FILES // LEN_DEST))
    y_train = np.array(y_train).reshape(-1)

    y_validation = []
    for char in validation_label:
        c = char.split('.')[0]
        y_validation.append([c] * (LEN_FILES // LEN_DEST))
    y_validation = np.array(y_validation).reshape(-1)

    y_test = []
    for char in test_label:
        c = char.split('.')[0]
        y_test.append([c] * (LEN_FILES // LEN_DEST))
    y_test = np.array(y_test).reshape(-1)

    labelencoder = LabelEncoder()
    y_train = to_categorical(labelencoder.fit_transform(y_train))
    y_validation = to_categorical(labelencoder.fit_transform(y_validation))
    y_test = to_categorical(labelencoder.fit_transform(y_test))
    model = make_model()
    _optimizer = Adam(learning_rate = 0.001)
    model.compile(loss = 'categorical_crossentropy',metrics = ['accuracy'],optimizer=_optimizer)

    checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.hdf5', 
                               verbose=0, save_best_only=True)
    start = datetime.now()

    history = model.fit(X_train, y_train, batch_size = N_BATCH_SIZE, epochs = N_EPOCHS, validation_data = (X_validation, y_validation), callbacks=[checkpointer], verbose=2)
    model.save(MODEL_NAME + '.hdf5')
    duration = datetime.now() - start
    print('Training completed in time: ', duration)
    train_accuracy=model.evaluate(X_train,y_train,verbose=0)
    print("The training accuracy for this run is ", train_accuracy[1])
    validation_accuracy=model.evaluate(X_validation,y_validation,verbose=0)
    print("The validation accuracy for this run is ", validation_accuracy[1])
    test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("The test accuracy for this run is ", test_accuracy[1])
    print(len(X_train), len(X_validation), len(X_test))
    plot_history(history)
    accuracy = accuracy + [[train_accuracy[1], validation_accuracy[1], test_accuracy[1]]]
    y_prediction = model.predict(X_test)
    y_prediction = to_categorical(np.argmax(y_prediction, axis=1), len(LIST_CATEGORIES))
    #Create confusion matrix and normalizes it over predicted (columns)
    result = confusion_matrix(
        y_test.argmax(axis=1), y_prediction.argmax(axis=1))
    results.append(result)
#Prints the overall average accuracies 
print('test accuracy', round(sum([x[-1] for x in accuracy])/RAND_RUNS, 4))
print('validation accuracy', round(sum([x[-2] for x in accuracy])/RAND_RUNS, 4))
print('training accuracy', round(sum([x[-3] for x in accuracy])/RAND_RUNS, 4))

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

