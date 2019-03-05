import os
import supervisedModels
import argparse
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import warnings
warnings.filterwarnings('ignore')    

    
    

"""
Model to learn feature from the data
"""    
def modelToLearnFeat(trainX, trainY, valX, valY, testX, testY):
    model = Sequential()
    inputX = len(trainX[0])
    print (inputX)
    model.add(LSTM(inputX, dropout=0.2, return_sequences=True, recurrent_dropout=0.2, input_shape = (int(len(trainX[0])),1)))
    model.add(LSTM(int(inputX/2), dropout=0.2, return_sequences=True, recurrent_dropout=0.2))
    model.add(LSTM(int(inputX/4), dropout=0.2, return_sequences=True, recurrent_dropout=0.2))
    model.add(LSTM(int(inputX/8), dropout=0.2, return_sequences=True, recurrent_dropout=0.2))
    model.add(LSTM(170, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3, activation='softmax'))    
    
    
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy' ])
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
                
    model.fit(trainX, trainY,
            epochs=30,
            batch_size = 10,
            verbose =2,
            callbacks=[early_stopping],
            validation_data=(valX, valY))
    

    # collects the output of the layer before the penultimate layer
    get_3rd_layer_output = K.function([model.layers[0].input],
                                    [model.layers[4].output])
                                    
    layer_output = get_3rd_layer_output([trainX])[0]
    print(layer_output.shape)
    print(layer_output)

    
    score, acc = model.evaluate(testX, testY)
   
    print('Test score:', score)
    print('Test accuracy:', acc)
            
                
"""
Divides the traning data further in to training and validation set
expands the dimension of input data as LSTM expects input data to be one dimension higher than the actual dimension
output class are encoded in one hot format
"""
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

  
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, default = "YouTube")
parser.add_argument("--useNN", type = str, default = "yes")
args = parser.parse_args()
print(args)

#retrieve the current working directory where feature files are located
featureFilesDirectory = os.path.join(os.getcwd(),"MasterThesis","datasets")
rootDirectoryDataset = os.path.join(featureFilesDirectory,args.dataset)
print(rootDirectoryDataset)


"""
The below section is executed if dataset to be use is specified as YouTube
By default it will be executed unless the argument "--dataset" is assigned "TEAM"
"""

if args.dataset == "YouTube":
    
    # creates the dataframe object of youtube datasets
    inputYouTubeAudioFeat, inputYouTubeTextFeat, inputYouTubeAudioTextFeat, outputYouTube = supervisedModels.getYouTubeData(rootDirectoryDataset)
    
    
    print ("LSTM model results for YouTube Datasets Text Features :")
    trainX, testX, trainY, testY = train_test_split(inputYouTubeTextFeat, outputYouTube, test_size=0.20, random_state=0)
    train_X_Text, train_Y_Text,  val_X_Text, val_Y_Text, test_X_Text, test_Y_Text = getDataForLSTM(trainX, testX, trainY, testY)
    modelToLearnFeat(train_X_Text, train_Y_Text,  val_X_Text, val_Y_Text, test_X_Text, test_Y_Text)


    