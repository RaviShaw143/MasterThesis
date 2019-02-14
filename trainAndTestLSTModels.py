import os
import supervisedModels
import argparse
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import to_categorical
import numpy as np


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
    trainY = to_categorical(trainY)
    valY = to_categorical(valY)
    
    #testing set to evaluate the model for Audio Features only
    testX = np.expand_dims(testX, 2)
    testY = to_categorical(testY)
    
    return trainX, trainY, valX, valY, testX, testY
    
def getLSTModelResults(audioFeat, textFeat, audioTextFeat, output, dataset):
    #splits the data of joined audioAndTextFeatures, AudioFeatures, and Text Features into training and testing data
    train_X_Aud, test_X_Aud, train_Y_Aud, test_Y_Aud = supervisedModels.splitDataAfterPCA(audioFeat, output)
    train_X_text, test_X_text, train_Y_text, test_Y_text = supervisedModels.splitDataAfterPCA(textFeat, output)
    train_X_Audtext, test_X_Audtext, train_Y_Audtext, test_Y_Audtext = supervisedModels.splitDataAfterPCA(audioTextFeat, output)
    
    print ("LSTM model results using " + str(dataset) + " audio features:" )
    trainX_Audio, trainY_Audio, valX_Audio, valY_Audio, testX_Audio, testY_Audio = getDataForLSTM(train_X_Aud, test_X_Aud, train_Y_Aud, test_Y_Aud)
    LSTModel(trainX_Audio, trainY_Audio, valX_Audio, valY_Audio, testX_Audio, testY_Audio)
    
    print ("LSTM model results using " + str(dataset) + " text features:")
    trainX_Text, trainY_Text,  valX_Text, valY_Text, testX_Text, testY_Text = getDataForLSTM(train_X_text, test_X_text, train_Y_text, test_Y_text)
    LSTModel(trainX_Text, trainY_Text,  valX_Text, valY_Text, testX_Text, testY_Text)
    
    print ("LSTM model results using " + str(dataset) + " audio and text features:")
    trainX_AT, trainY_AT, valX_AT,  valY_AT, testX_AT, testY_AT = getDataForLSTM(train_X_Audtext, test_X_Audtext, train_Y_Audtext, test_Y_Audtext)
    LSTModel(trainX_AT, trainY_AT, valX_AT,  valY_AT, testX_AT, testY_AT)



parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, default = "YouTube")
parser.add_argument("--useNN", type = str, default = "yes")
args = parser.parse_args()
print(args)

#retrieve the current working directory where feature files are located
featureFilesDirectory = os.path.join(os.getcwd(),"datasets")
rootDirectoryDataset = os.path.join(featureFilesDirectory,args.dataset)
print(rootDirectoryDataset)

if args.dataset == "YouTube":
    # creates the dataframe object of youtube datasets
    inputYouTubeAudioFeat, inputYouTubeTextFeat, inputYouTubeAudioTextFeat, outputYouTube = supervisedModels.getYouTubeData(rootDirectoryDataset)
    print("LSTM model results for YouTube Datasets:")
    getLSTModelResults(inputYouTubeAudioFeat, inputYouTubeTextFeat, inputYouTubeAudioTextFeat, outputYouTube, "YouTube")

if args.dataset == "TEAM":
    # creates the dataframe object of TEAM datasets 
    inputTeamAudioFeat, inputTeamTextFeat, inputTeamAudioTextFeat, outputTeam = supervisedModels.getTEAMData(rootDirectoryDataset)
    print("LSTM model results for TEAM Datasets:")
    supervisedModels.getResults(inputTeamAudioFeat, inputTeamTextFeat, inputTeamAudioTextFeat, outputTeam, "TEAM")
