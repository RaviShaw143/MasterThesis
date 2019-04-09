import os
import supervisedModels
import argparse
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten, GlobalAveragePooling1D, LSTM
from tensorflow.keras.layers import Add
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
#from tekeras.models import load_model
import warnings
warnings.filterwarnings('ignore')    


"""
* Trains a LSTM model using given input data
* Extract output from layer before the penultimate layer to use it in feed forward network
"""
def LSTModel(trainX, trainY, valX, valY, testX, testY):
    model = Sequential()
    inputX = len(trainX[0])
    print (inputX)
    model.add(LSTM(inputX, dropout=0.2, return_sequences=True, recurrent_dropout=0.2, input_shape = (int(len(trainX[0])),1)))
    model.add(LSTM(130, dropout=0.2, return_sequences=True, recurrent_dropout=0.2))
    model.add(LSTM(95, dropout=0.2, recurrent_dropout=0.2))  
    model.add(Dense(3, activation='softmax'))    
    
    
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy' ])
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
                
    model.fit(trainX, trainY,
            epochs=30,
            verbose =2,
            callbacks=[early_stopping],
            validation_data=(valX, valY))
    

    # collects the output of the layer before the penultimate layer
    get_3rd_layer_output = K.function([model.layers[0].input],
                                    [model.layers[2].output])
                                    
    layer_output = get_3rd_layer_output([trainX])[0]
    print(layer_output.shape)
    print(layer_output)

    
    score, acc = model.evaluate(testX, testY)
   
    print('Test score:', score)
    print('Test accuracy:', acc)
    
    atPath = os.path.join(os.getcwd(),"MasterThesis","model_LSTM(T).h5")
    model.save(atPath)
    
    model_LSTM = load_model(atPath)
    
    scoreL, accL = model_LSTM.evaluate(testX, testY)
    print('Test score:', scoreL)
    print('Test accuracy:', accL)
   
        
    
def getDataForLSTM(X, testX, Y, testY):
    
                        
    """
    Divides the traning data further in to training and validation set
    expands the dimension of input data as LSTM expects input data to be one dimension higher than the actual dimension
    output class are encoded in one hot format
    """

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

def CNNModel(trainX, trainY, valX, valY, testX, testY):
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    valX = np.reshape(valX, (valX.shape[0],valX.shape[1],1))
    
    print(trainX.shape)
    num_features = len (trainX[0])
    model_cnn = Sequential()
    model_cnn.add(Conv1D(32, kernel_size = 3, activation = "tanh", input_shape = (num_features,1)))
    model_cnn.add(MaxPooling1D(pool_size=2))
    model_cnn.add(Dropout(0.25))
    model_cnn.add(Conv1D(64, kernel_size = 2, activation = "tanh", input_shape = (num_features,1)))
    model_cnn.add(MaxPooling1D(pool_size=2))
    model_cnn.add(Dropout(0.25))
    model_cnn.add(Flatten())
    model_cnn.add(Dense(95, activation= "softmax"))
    model_cnn.add(Dense(3, activation= "softmax"))
    
    
    model_cnn.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy' ])
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)            
    
    model_cnn.fit(trainX, trainY,
            epochs=10,
            verbose =2,
            callbacks = [early_stopping],
            validation_data=(valX, valY))
            
    score, acc = model_cnn.evaluate(testX, testY)
   
    print('Test score:', score)
    print('Test accuracy:', acc)
    
    atPath = os.path.join(os.getcwd(),"MasterThesis","model_cnn(T).h5")
    model_cnn.save(atPath)
    
    model_L_cnn = load_model(atPath)
    
    scoreL, accL = model_L_cnn.evaluate(testX, testY)
    print('Test score:', scoreL)
    print('Test accuracy:', accL)
   
def jointModel(trainX_Text, trainY_Text,  valX_Text, valY_Text, testX_Text, testY_Text, trainX_Aud, trainY_Aud,  valX_Aud, valY_Aud, testX_Aud, testY_Aud):
    
    
    with tf.Session() as session:
        K.set_session(session)
        session.run(tf.global_variables_initializer())  
        session.run(tf.tables_initializer())
        #model_elmo = build_model() 
        #model_elmo.load_weights(os.path.join(os.getcwd(),"MasterThesis",'model_elmo_weights(YT0.5).h5'))
        #model_elmo.save(os.path.join(os.getcwd(),"MasterThesis",'model_elmo_DD(YT0.5).h5'))
        model_lstm = load_model(os.path.join(os.getcwd(),"MasterThesis",'model_elmo_lstm.h5'))
        #score, acc = model_elmo.evaluate(testX, testY_oh)
        #print('Test score:', score)
        #print('Test accuracy:', acc)
        
        
        #loads the trained audio model
        atPath1 = os.path.join(os.getcwd(),"MasterThesis","model_elmo_lstm.h5")        
        #model_lstm = load_model(atPath1)
        print(model_lstm.summary())
        model_lstm.pop()
        
        for layer in model_lstm.layers:
            layer.trainable= False
    
    
        
        #loads the trained text model
        atPath2 = os.path.join(os.getcwd(),"MasterThesis","model_cnn(0.46).h5")
        model_cnn = load_model(atPath2)
        print(model_cnn.summary())
        model_cnn.pop()
        
        for layer in model_cnn.layers:
            layer.trainable= False
    
            
        
        concat_layer =  Add()([model_lstm.output,model_cnn.output])
        concat_layer = Dense(190, name = 'fully_connected_layer')(concat_layer)
        output_layer = Dense(3, activation = "softmax", name = 'final_Dense_layer')(concat_layer)
        
        jm = Model ([model_lstm.input,model_cnn.input],output_layer)
        
        jm.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy' ])
        jm.summary()
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)            
            
        jm.fit([trainX_Text,trainX_Aud], trainY_Text, epochs=20, callbacks = [early_stopping], verbose =2,validation_data=([valX_Text,valX_Aud], valY_Text))
        
                
        score, acc = jm.evaluate([testX_Text,testX_Aud], testY_Text)
    
        print('Test score:', score)
        print('Test accuracy:', acc)
        
        atPath = os.path.join(os.getcwd(),"MasterThesis","model_jnn_elmo(D.).h5")
        jm.save(atPath)
        
        
    #print("Load the saved joint neural network model")
    #atPath1 = os.path.join(os.getcwd(),"MasterThesis","model_jnn(D.46).h5")

    #model_L_jm = load_model(atPath1)
    
    #scoreL, accL = model_L_jm.evaluate([testX_Text,testX_Aud], testY_Text)
    #print('Test score :', scoreL)
    #print('Test accuracy:', accL)
   
    
   
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
    
    #The below results is obtained using a method which divides the datasets into
    #training and testing set and uses PCA to reduce the number of features
    train_X_Audtext, test_X_Audtext, train_Y_Audtext, test_Y_Audtext = supervisedModels.splitDataAfterPCA(inputYouTubeAudioTextFeat, outputYouTube)
    print("Reduced audio and Text features: " + str(len(train_X_Audtext[0])))
   
    #extracts reduced audio features
    train_X_Aud, test_X_Aud, train_Y_Aud, test_Y_Aud = supervisedModels.splitDataAfterPCA(inputYouTubeAudioFeat, outputYouTube)
    print("Reduced audio features: " + str(len(train_X_Aud[0])))
    
    #extracts reduced text features
    train_X_text, test_X_text, train_Y_text, test_Y_text = supervisedModels.splitDataAfterPCA(inputYouTubeTextFeat, outputYouTube)
    print("Reduced Text features: " + str(len(train_X_text[0])))
    
    #The below uses a method to obatin a valid data for training and testing LSTM model
    trainX_Text, trainY_Text,  valX_Text, valY_Text, testX_Text, testY_Text = getDataForLSTM(train_X_text, test_X_text, train_Y_text, test_Y_text)
    trainX_Aud, trainY_Aud,  valX_Aud, valY_Aud, testX_Aud, testY_Aud = getDataForLSTM(train_X_Aud, test_X_Aud, train_Y_Aud, test_Y_Aud)

    #LSTModel(trainX_Text, trainY_Text,  valX_Text, valY_Text, testX_Text, testY_Text)
    jointModel(trainX_Text, trainY_Text,  valX_Text, valY_Text, testX_Text, testY_Text, trainX_Aud, trainY_Aud,  valX_Aud, valY_Aud, testX_Aud, testY_Aud)
    #print (valY_Text)
    #print (valY_Aud)
    #Trains and test the LSTM model
    #LSTModel(trainX_Text, trainY_Text,  valX_Text, valY_Text, testX_Text, testY_Text)
    
    

    