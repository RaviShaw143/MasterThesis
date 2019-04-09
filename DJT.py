import os
import supervisedModels
import argparse
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import to_categorical
import numpy as np

import keras.backend as K



def LSTModel(trainX, trainY, valX, valY, testX, testY):
    model = Sequential()
    inputX = int(len(trainX[0]))
    model.add(LSTM(inputX, dropout=0.2, return_sequences=True, recurrent_dropout=0.2, input_shape = (inputX,1)))
    model.add(LSTM(int(inputX/4), dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='sigmoid'))    
    print ("activation sigmoid" +str(inputX))
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy' ])    
        
    model.fit(trainX, trainY,
            batch_size=17,
            epochs=10,
            verbose=2,
            validation_data=(valX, valY))
    
    score, acc = model.evaluate(testX, testY,
                                batch_size=25,
                                verbose=2)
    
    print('Test score:', score)
    print('Test accuracy:', acc)

#divides the traning data further in to train and validation set
def getDataForLSTM(X, testX, Y, testY):
    
    #dividing the Text features training data into further train and validation set
    trainX, valX, trainY, valY = train_test_split(X, Y, test_size=0.20, random_state=0)
    
    #expands the dimension of data as LSTM expects a 3 dimension input
    trainX = np.expand_dims(trainX, 2)
    valX = np.expand_dims(valX, 2)
    trainY = to_categorical(trainY, num_classes =3)
    valY = to_categorical(valY, num_classes =3)
    
    #testing set to evaluate the model for Audio Features only
    testX = np.expand_dims(testX, 2)
    testY = to_categorical(testY, num_classes =3)
    
    return trainX, trainY, valX, valY, testX, testY


def prepareDataLSTM (textFeat, output, dataset):
    print ("LSTM model results using " + str(dataset) + " text features:")
    train_X_text, test_X_text, train_Y_text, test_Y_text = train_test_split(textFeat, output, test_size=0.20, random_state=0)
    #train_X_text, test_X_text, train_Y_text, test_Y_text = supervisedModels.splitDataAfterPCA(textFeat, output)
    #print (len (train_X_text[0]))
    print(train_Y_text)
    trainX_Text, trainY_Text,  valX_Text, valY_Text, testX_Text, testY_Text = getDataForLSTM(train_X_text, test_X_text, train_Y_text, test_Y_text)
    print(len(trainX_Text[0]))
    print(trainY_Text.shape )
    print(trainX_Text.shape)
    print(trainY_Text)
    
    trainLSTModel(trainX_Text, trainY_Text,  valX_Text, valY_Text, testX_Text, testY_Text)
    
def trainLSTModel(trainX, trainY, valX, valY, testX, testY):
    model = Sequential()
    inputX = int(len(trainX[0]))
    print(inputX)
    model.add(LSTM(inputX, dropout=0.2, return_sequences=True, recurrent_dropout=0.2, input_shape = (inputX,1)))
    model.add(LSTM(int(inputX/2), dropout=0.2, return_sequences=True, recurrent_dropout=0.2))
    model.add(LSTM(int(inputX/4), dropout=0.2, return_sequences=True, recurrent_dropout=0.2))
    model.add(LSTM(int(inputX/8), dropout=0.2, return_sequences=True, recurrent_dropout=0.2))
    model.add(LSTM(170, dropout=0.2,recurrent_dropout=0.2))
    model.add(Dense(3,activation= "softmax"))
    
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy' ])  
                  
    model.fit(trainX, trainY,
            batch_size = 10,
            epochs=10,
            verbose=2,
            validation_data=(valX, valY))
    
    score, acc = model.evaluate(testX, testY,
                                batch_size=10,
                                verbose=2)
    
    print('Test score:', score)
    print('Test accuracy:', acc)
    
    
  
 
config = K.tf.ConfigProto()
config.gpu_options.allow_growth = True
session = K.tf.Session(config=config)
  

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, default = "YouTube")
parser.add_argument("--useNN", type = str, default = "yes")
args = parser.parse_args()
print(args)

#retrieve the current working directory where feature files are located
featureFilesDirectory = os.path.join(os.getcwd(),"MasterThesis","datasets")
rootDirectoryDataset = os.path.join(featureFilesDirectory,args.dataset)
print(rootDirectoryDataset)

if args.dataset == "YouTube":
    # creates the dataframe object of youtube datasets
    inputYouTubeAudioFeat, inputYouTubeTextFeat, inputYouTubeAudioTextFeat, outputYouTube = supervisedModels.getYouTubeData(rootDirectoryDataset)
    print("LSTM model results for YouTube Datasets:")
    prepareDataLSTM(inputYouTubeTextFeat, outputYouTube, "YouTube")
    #getLSTModelResults(inputYouTubeAudioFeat, inputYouTubeTextFeat, inputYouTubeAudioTextFeat, outputYouTube, "YouTube")

if args.dataset == "TEAM":
    # creates the dataframe object of TEAM datasets 
    inputTeamAudioFeat, inputTeamTextFeat, inputTeamAudioTextFeat, outputTeam = supervisedModels.getTEAMData(rootDirectoryDataset)
    print("LSTM model results for TEAM Datasets:")
    #getLSTModelResults(inputTeamAudioFeat, inputTeamTextFeat, inputTeamAudioTextFeat, outputTeam, "TEAM")
