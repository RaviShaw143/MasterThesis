import os
import supervisedModels
import argparse
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten, LSTM, Add, Embedding
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from numpy import asarray
from numpy import zeros
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler

    


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
    model.add(LSTM(inputX, return_sequences=True, input_shape = (int(len(trainX[0])),1)))
    #model.add(LSTM(175, return_sequences=True))
    model.add(LSTM(150, return_sequences=True))    
    model.add(LSTM(95))
    model.add(Dropout(0.1, name = "dropout_l"))
    model.add(Dense(3, activation='softmax'))    
    
    
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy' ])
    model.summary()

    #early_stopping = EarlyStopping(monitor='val_loss', patience=5)
                
    history = model.fit(trainX, trainY,
            epochs=15,
            verbose =2,
            #callbacks=[early_stopping],
            validation_data=(valX, valY))
    

    # collects the output of the layer before the penultimate layer
    #get_3rd_layer_output = K.function([model.layers[0].input], [model.layers[2].output])
                                    
    #layer_output = get_3rd_layer_output([trainX])[0]
    #print(layer_output.shape)
    #print(layer_output)
    
    # list all data in history
    print (type(history))
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.show()
    import pickle
    saveHistATpath = os.path.join(os.getcwd(),"MasterThesis","PreTrainedModels","historyLSTM().pckl")

    # save:
    f = open(saveHistATpath, 'wb')
    pickle.dump(history.history, f)
    f.close()
    
    # retrieve:    
    # f = open('history.pckl', 'rb')
    # history = pickle.load(f)
    # f.close()

    score, acc = model.evaluate(testX, testY)
   
    print('Test score:', score)
    print('Test accuracy:', acc)
    
    #save the model at the following path 
    atPath = os.path.join(os.getcwd(),"MasterThesis","PreTrainedModels","model_LSTM().h5")
    model.save(atPath)
    
    #following loads the pretrained model to produce reproducilble resluts since models does not produce the same results on every run 
    loadModelAtPath = os.path.join(os.getcwd(),"MasterThesis","PreTrainedModels","model_LSTM().h5")
    model_LSTM = load_model(loadModelAtPath)
    
    scoreL, accL = model_LSTM.evaluate(testX, testY)
    print('Test score of pretrained Text model:', scoreL)
    print('Test accuracy of pretrained Text model:', accL)
   
        
    
def getDataForNeuralModel(X, Y, dataset):
    
                        
    """
    Divides the traning data further in to training and validation set
    expands the dimension of input data as LSTM expects input data to be one dimension higher than the actual dimension
    output class are encoded in one hot format
    """
    train_X, testX, train_Y, testY = train_test_split(X, Y, test_size=0.20, random_state=0)
    #Normalization technique
    scaler = StandardScaler()
    # Fit on training set only.
    scaler.fit(train_X)

    # Normalization of the dataset.    
    train_X = scaler.transform(train_X)
    testX = scaler.transform(testX)
    
    #dividing the Text features training data into further train and validation set
    trainX, valX, trainY, valY = train_test_split(train_X, train_Y, test_size=0.20, random_state=0)
      
    #expands the dimension of data as LSTM expects a 3 dimension input
    trainX = np.expand_dims(trainX, 2)
    valX = np.expand_dims(valX, 2)
    if dataset == "YouTube":
        trainY = to_categorical(trainY, num_classes =3)
        valY = to_categorical(valY, num_classes =3)
    if dataset == "TEAM":
        trainY = to_categorical(trainY, num_classes =2)
        valY = to_categorical(valY, num_classes =2)
    
    #testing set to evaluate the model for Audio Features only
   
    testX = np.expand_dims(testX, 2)
    if dataset == "YouTube":
        testY = to_categorical(testY, num_classes =3)

    if dataset == "TEAM":
        testY = to_categorical(testY, num_classes =2)
    
    return trainX, trainY, valX, valY, testX, testY

def CNNModel(trainX, trainY, valX, valY, testX, testY):
    
#    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
#    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
#    valX = np.reshape(valX, (valX.shape[0],valX.shape[1],1))
#    
    print(trainX.shape)
    
    num_features = len (trainX[0])
    model_cnn = Sequential()
    model_cnn.add(Conv1D(32, kernel_size = 3, activation = "relu", input_shape = (num_features,1)))
    model_cnn.add(Conv1D(64, kernel_size = 3, activation = "relu"))
    model_cnn.add(MaxPooling1D(pool_size=2))
    model_cnn.add(Dropout(0.10, name = 'dropout_cnn'))
    model_cnn.add(Flatten())
    model_cnn.add(Dense(95, activation= "relu", name = "dense_cnn"))
    model_cnn.add(Dense(3, activation= "softmax", name = "cnn_output_layer"))
    
    
    model_cnn.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy' ])
    
    model_cnn.summary()
    
    history = model_cnn.fit(trainX, trainY,
            epochs=15,
            verbose =2,
            #callbacks = [early_stopping],
            validation_data=(valX, valY))
    score, acc = model_cnn.evaluate(testX, testY)
   
    print('Test score:', score)
    print('Test accuracy:', acc)
    
    #saves the model at following path
    atPath = os.path.join(os.getcwd(),"PreTrainedModels","Amodel_cnn().h5")
    model_cnn.save(atPath)
    
#    #following loads the pretrained model to produce reproducilble resluts since models does not produce the same results on every run         
#    loadModelAtPath = os.path.join(os.getcwd(),"PreTrainedModels","Amodel_cnn().h5")
#    model_L_cnn = load_model(loadModelAtPath)
#    
#    scoreL, accL = model_L_cnn.evaluate(testX, testY)
#    print('Test score of pretrained Audio Model:', scoreL)
#    print('Test accuracy of pretrained Audio Model:', accL)
#   
            
    # list all data in history
    #print(history.history.keys())

    import pickle
    saveHistATpath = os.path.join(os.getcwd(),"PreTrainedModels","historyCNN().pckl")

    # save:
    f = open(saveHistATpath, 'wb')
    pickle.dump(history.history, f)
    f.close()
         
    
def jointModel(trainX_Text, trainY_Text,  valX_Text, valY_Text, testX_Text, testY_Text, trainX_Aud, trainY_Aud,  valX_Aud, valY_Aud, testX_Aud, testY_Aud):
    
    #loads the pretrained audio model
    atPath1 = os.path.join(os.getcwd(),"PreTrainedModels","embeddingGlove(YT0.389).h5")        
    model_lstm = load_model(atPath1)
    model_lstm.summary()
    model_lstm.pop()

    #following freezes the layers of the model to use the pretrained weights 
    for layer in model_lstm.layers:
        layer.trainable= False
        print(layer.name)            
    
    #loads the trained text model
    atPath2 = os.path.join(os.getcwd(),"PreTrainedModels","Amodel_cnn(0.48).h5")
    model_cnn = load_model(atPath2)
    model_cnn.summary()
    model_cnn.pop()
    
    #following freezes the layers of the model to use the pretrained weights 
    for layer in model_cnn.layers:
        print(layer.name)
        layer.trainable= False

            
    #following concatenates the output of the two models in to a fully connected layer
    concat_layer =  Add()([model_lstm.output,model_cnn.output])
    concat_layer = Dense(190, activation = "relu",name = 'fully_connected_layer')(concat_layer)
    concat_layer = Dense(95, activation = "relu",name = 'connected_layer')(concat_layer) 
    output_layer = Dense(3, activation = "softmax", name = 'final_Dense_layer')(concat_layer)
    
    jointModel = Model ([model_lstm.input,model_cnn.input],output_layer)
    
    #compiles the joint model
    jointModel.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy' ])
                
    #summary of the joint model
    jointModel.summary()
        
   # trains the joint model
#    history = jointModel.fit([trainX_Text,trainX_Aud], trainY_Text, epochs=15,
#   # callbacks = [early_stopping], 
#    verbose =2,validation_data=([valX_Text,valX_Aud], valY_Text))
#    
#    score, acc = jointModel.evaluate([testX_Text,testX_Aud], testY_Text)
#    
#    print('Test score:', score)
#    print('Test accuracy:', acc)
    
#    #save the model at the following path
#    atPath = os.path.join(os.getcwd(),"PreTrainedModels","model_jnn().h5")
#    jointModel.save(atPath)
#    
#    import pickle
#    saveHistATpath = os.path.join(os.getcwd(),"PreTrainedModels","historyJNN().pckl")
#
#    # save:
#    f = open(saveHistATpath, 'wb')
#    pickle.dump(history.history, f)
#    f.close()
    
    print("Load the saved joint neural network model")
    atPath1 = os.path.join(os.getcwd(),"PreTrainedModels","model_jnn(0.54).h5")

    model_L_jm = load_model(atPath1)
    print(model_L_jm.summary())
    
    scoreL, accL = model_L_jm.evaluate([testX_Text,testX_Aud], testY_Text)
    print('Test score of pretrained joint model:', scoreL)
    print('Test accuracy of pretrained joint model:', accL)
    
    
    # retrieve:    
    # f = open('history.pckl', 'rb')
    # history = pickle.load(f)
    # f.close()

    # list all data in history
    #print(history.history.keys())
    
            
    
def LSTMModelUsingEmbedding(trainX, trainY, valX, valY, testX, testY, vocab_size, max_words, embedding_matrix):

    #defining the model
    model = Sequential()
    model.add(Embedding(vocab_size, 300, input_length = max_words, trainable = False, weights=[embedding_matrix]))
    model.add(LSTM(300, return_sequences=True))
    model.add(LSTM(200, return_sequences=True))
    model.add(LSTM(95))
    model.add(Dense(95, activation = "tanh", name = "penultimate_layer"))
    model.add(Dropout(0.2, name = "dropout_lstm"))
    model.add(Dense(3, activation = "softmax"))
    
    #compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    #summary of the model
    print(model.summary())
    
    #early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    
    #fitting the model
    history = model.fit(trainX, trainY,
                epochs=40,
                verbose =2,
              #  callbacks=[early_stopping],
                validation_data=(valX, valY))
    
    atPathh = os.path.join(os.getcwd(),"MasterThesis","glove300T","embeddingGlove(T).h5")
    model.save(atPathh)
    import pickle
    saveHistATpath = os.path.join(os.getcwd(),"MasterThesis","glove300T","historyGloveT().pckl")
    
    # save:
    f = open(saveHistATpath, 'wb')
    pickle.dump(history.history, f)
    f.close()
        
    # retrieve:    
    #f = open(saveHistATpath, 'rb')
    #history = pickle.load(f)
    #f.close()
    #
    
    #test the model
    score, acc = model.evaluate(testX, testY)
    print('Test score:', score)
    print('Test accuracy:', acc)
    
    #loads and test the model 
    atPath = os.path.join(os.getcwd(),"MasterThesis","glove300T","embeddingGlove(T).h5")
    model_l  = load_model(atPath)  
      
    score1, acc1 = model_l.evaluate(testX, testY)
    print('Test score_l:', score1)
    print('Test accuracy_l:', acc1)
    

    
def LSTMModelUsingEmbeddingTEAM(trainX, trainY, valX, valY, testX, testY, vocab_size, max_words, embedding_matrix):


    print(trainX.shape)
    print(trainY.shape)
    print(valX.shape)
    print(valY.shape)
    print(testX.shape)
    print(testY.shape)
    
    #defining the model
    model = Sequential()
    model.add(Embedding(vocab_size, 300, input_length = max_words, trainable = False, weights=[embedding_matrix]))
    #model.add(LSTM(800, return_sequences=True))
    #model.add(Dropout(0.2, name = "dropout_lstm_1"))
#    model.add(LSTM(600, return_sequences=True))
#    #model.add(LSTM(600))
#    model.add(Dropout(0.2, name = "dropout_lstm_2"))
    model.add(LSTM(800))
    model.add(Dense(400, activation = "tanh", name = "penultimate_layer"))
    model.add(Dense(2, activation = "softmax"))
    
    #compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("Compilation Successful")
    #summary of the model
    print(model.summary())
    
    #early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    
    #fitting the model
    history = model.fit(trainX, trainY,
                epochs=25,
                verbose =2,
              #  callbacks=[early_stopping],
                validation_data=(valX, valY))
    
    atPathh = os.path.join(os.getcwd(),"MasterThesis","glove300T","embeddingGlove(T).h5")
    model.save(atPathh)
    import pickle
    saveHistATpath = os.path.join(os.getcwd(),"MasterThesis","glove300T","historyGloveT().pckl")
    
    # save:
    f = open(saveHistATpath, 'wb')
    pickle.dump(history.history, f)
    f.close()
        
    # retrieve:    
    #f = open(saveHistATpath, 'rb')
    #history = pickle.load(f)
    #f.close()
    #
    
    #test the model
    score, acc = model.evaluate(testX, testY)
    print('Test score:', score)
    print('Test accuracy:', acc)
    
    #loads and test the model 
    atPath = os.path.join(os.getcwd(),"MasterThesis","glove300T","embeddingGlove(T).h5")
    model_l  = load_model(atPath)  
      
    score1, acc1 = model_l.evaluate(testX, testY)
    print('Test score_l:', score1)
    print('Test accuracy_l:', acc1)
    

def CNNModelTEAM(trainX, trainY, valX, valY, testX, testY):
    
#    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
#    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
#    valX = np.reshape(valX, (valX.shape[0],valX.shape[1],1))
#    
    print(trainX.shape)
    
    num_features = len (trainX[0])
    model_cnn = Sequential()
    model_cnn.add(Conv1D(32, kernel_size = 3, activation = "tanh", input_shape = (num_features,1)))
    model_cnn.add(MaxPooling1D(pool_size=2))
    model_cnn.add(Dropout(0.20, name = 'dropout_cnn'))
  
    model_cnn.add(Conv1D(64, kernel_size = 3, activation = "tanh"))
    #model_cnn.add(Conv1D(128, kernel_size = 3, activation = "relu"))  
    model_cnn.add(MaxPooling1D(pool_size=2))
    model_cnn.add(Dropout(0.20, name = 'dropout_cnn1'))
    model_cnn.add(Flatten())
    model_cnn.add(Dense(400, activation= "tanh", name = "dense_cnn"))
    model_cnn.add(Dense(2, activation= "softmax", name = "cnn_output_layer"))
    
    
    model_cnn.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy' ])
    
    model_cnn.summary()
    
    history = model_cnn.fit(trainX, trainY,
            epochs=5,
            verbose =2,
            #callbacks = [early_stopping],
            validation_data=(valX, valY))
    score, acc = model_cnn.evaluate(testX, testY)
   
    print('Test score:', score)
    print('Test accuracy:', acc)
    
    #saves the model at following path
    atPath = os.path.join(os.getcwd(),"MasterThesis","TEAMAudioModels","Amodel_cnnTEAM().h5")
    model_cnn.save(atPath)
    
#    #following loads the pretrained model to produce reproducilble resluts since models does not produce the same results on every run         
    loadModelAtPath = os.path.join(os.getcwd(),"MasterThesis","TEAMAudioModels","Amodel_cnnTEAM().h5")
    model_L_cnn = load_model(loadModelAtPath)
    
    scoreL, accL = model_L_cnn.evaluate(testX, testY)
    print('Test score of pretrained Audio Model:', scoreL)
    print('Test accuracy of pretrained Audio Model:', accL)
#   
            
    # list all data in history
    #print(history.history.keys())

    import pickle
    saveHistATpath = os.path.join(os.getcwd(),"MasterThesis","TEAMAudioModels","historyCNNTEAM().pckl")

    # save:
    f = open(saveHistATpath, 'wb')
    pickle.dump(history.history, f)
    f.close()
    

    
          
          

def findMaxNumberOfWords(textData):
    max_words = 0
    for sentence in textData:
        num_words= len(sentence.split())
        if num_words > max_words:
            max_words = num_words
    print("Maximum number of words in a sentence in the YouTube dataset: " + str(max_words))
    return max_words

def getDataForEmbedModel(textFilePath, dataset):
    
    """
    This method does the following task:
        * Applies normalization technique to the text
        * Converts texts to sequence using a tokenizer object and saves the tokenizer object
        * returns training, validation, testing data, vocab size, and embedding matrix 
    """
    textFilePath = ""
    if dataset == "YouTube":
        textFilePath = os.path.join(os.getcwd(),"textYT.csv")
    if dataset == "TEAM":
         textFilePath = os.path.join(os.getcwd(),"textDataTEAM.csv")
         
    df = pd.read_csv(textFilePath)
    print (df.head(5))
   
    #finds the maximum number of words 		
    print("Before normalization")
    max_words = findMaxNumberOfWords(df['Text'])
    
 
    # applies text normalization
    df['Text'] = df['Text'].map(lambda x: normalizeTextData(x, True, True))
   
    print (df.head(5))
   
    #finds the maximum number of words 
    print("After normalization")		
    max_words = findMaxNumberOfWords(df['Text'])
   
    #splits the data into training and testing
    trainX, testX, trainY, testY = train_test_split(df['Text'], df['sentiment'], test_size = 0.20, random_state = 0 )
    
#    tok = Tokenizer()
#    tok.fit_on_texts(df['Text'])
#   
    
    # saves the tokenizer object
    saveTokenizerAtPath = os.path.join(os.getcwd(),"tokenizerTEAM().pickle")
#    with open(saveTokenizerAtPath, 'wb') as handle:
#        pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)
#    
    
    
    # loading
    print("Using saved tokenizer")
    f = open(saveTokenizerAtPath, 'rb') 
    tok = pickle.load(f)
    trainX_testTok = tok.texts_to_sequences(trainX)
    print(trainX_testTok[0])
    
    vocab_size = len(tok.word_index) + 1
    print ("Vocabulary size: " + str(vocab_size))
    
    trainX_seq = tok.texts_to_sequences(trainX)
    testX_seq = tok.texts_to_sequences(testX)
    
    #print(trainX_seq[0])
    
    
        
    pad_trainX_seq  = pad_sequences(trainX_seq, maxlen = max_words, padding = 'post')
    pad_testX_seq = pad_sequences(testX_seq, maxlen = max_words, padding = 'post')
    
    if dataset == "YouTube":
    
        trainY_Text = to_categorical(trainY, num_classes =3)
        testY_Text = to_categorical(testY, num_classes =3)
    
        #splitting off the validation set
        trainX_Text, valX_Text, trainY_Text, valY_Text = train_test_split(pad_trainX_seq, trainY_Text, test_size = 0.20, random_state = 0)

    if dataset == "TEAM":
    
        trainY_Text = to_categorical(trainY, num_classes =2)
        testY_Text = to_categorical(testY, num_classes =2)
    
        #splitting off the validation set
        trainX_Text, valX_Text, trainY_Text, valY_Text = train_test_split(pad_trainX_seq, trainY_Text, test_size = 0.20, random_state = 0)

#    embedding_matrix = embeddingMatrix(vocab_size, tok, dataset)
    
    #gets the embedding matrix
    saveEmbeddingMatrix = os.path.join(os.getcwd(),"gloveEmbedTEAM.pickle")
    f = open(saveEmbeddingMatrix, 'rb') 
    embedding_matrix = pickle.load(f)
   
#    embedding_matrix  = "empty"
    
    return trainX_Text, trainY_Text, valX_Text, valY_Text, pad_testX_seq, testY_Text, vocab_size, max_words, embedding_matrix


def normalizeTextData(text, remove_stopwords = False, lemmatization = False):
    
    # splits the words
    text = text.split()
    
    #Optionally remove stop words
    if remove_stopwords:
        #print("Removing Stop words")
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)
    
    #Optionally, apply lemmatization    
    if lemmatization:
        #print("Applyig lemmatization")
        lemmatizer = WordNetLemmatizer()
        text = text.split()
        text = ' '.join([lemmatizer.lemmatize(w) for w in text])
    
    return text  


def embeddingMatrix(vocab_size, tok, dataset):
    
    """
    This method returns the embedding matrix
    """
    # load the pretrained embedding model into memory
    embeddings_index = dict()
    pathToEmbedModel = os.path.join(os.getcwd(),"glove.6B.300d.txt")
    f = open(pathToEmbedModel, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))

    #create a weight matrix for words in training docs
    embedding_matrix = zeros((vocab_size, 300))
    for word, i in tok.word_index.items():
    	embedding_vector = embeddings_index.get(word)
    	if embedding_vector is not None:
    		embedding_matrix[i] = embedding_vector
    
    saveEmbeddingMatrix = ""
    #save the embedding matrix
    if dataset == "YouTube":
        saveEmbeddingMatrix = os.path.join(os.getcwd(),"gloveEmbedYT.pickle")
        with open(saveEmbeddingMatrix, 'wb') as handle:
            pickle.dump(embedding_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #save the embedding matrix
    if dataset == "TEAM":
        saveEmbeddingMatrix = os.path.join(os.getcwd(),"gloveEmbedTEAM.pickle")
        with open(saveEmbeddingMatrix, 'wb') as handle:
            pickle.dump(embedding_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    return embedding_matrix

        
    
   
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, default = "TEAM")
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
    
#    atPath = os.path.join(os.getcwd(), "PreTrainedModels", "embeddingModel(YT0.44).h5")
#    
#    model_L_cnn = load_model(atPath)
#    print("Summary of gloveEmbedding LSTM model")
#    model_L_cnn.summary()
#    
#    for layer in model_L_cnn.layers:
#        if layer.name != 'embedding':
#            if layer.name != "dropout":
#                print(layer.activation)
#        if layer.name == "dropout":
#            print(layer.rate)
#    
    
    
    # creates the dataframe object of youtube datasets
    inputYouTubeAudioFeat, inputYouTubeTextFeat, inputYouTubeAudioTextFeat, outputYouTube = supervisedModels.getYouTubeData(rootDirectoryDataset)
    
    #tfidfFeatures for lstm
    #tfidfInput, outputYT = supervisedModels.extractTextFeatures(rootDirectoryDataset)
    #print(tfidfInput.shape)
    
    #extracts reduced text features
#    train_X_textTf, test_X_textTf, train_Y_textTf, test_Y_textTf = supervisedModels.splitDataAfterPCA(tfidfInput, outputYT,"text","YouTube")
#    print("Reduced Text features after PCA: " + str(train_X_textTf.shape))
#    
    
    #data for CNN model
    trainX_Aud, trainY_Aud,  valX_Aud, valY_Aud, testX_Aud, testY_Aud = getDataForNeuralModel(inputYouTubeAudioFeat, outputYouTube, "YouTube")
    
    # tfidf data for LSTM model
    #trainX_Text, trainY_Text,  valX_Text, valY_Text, testX_Text, testY_Text = getDataForNeuralModel(train_X_textTf, test_X_textTf, train_Y_textTf, test_Y_textTf)
    
    print("Retrieve data for embedding model")
    trainX_EmbedText, trainY_EmbedText,  valX_EmbedText, valY_EmbedText, testX_EmbedText, testY_EmbedText, vocab_size, max_words, embedding_matrix = getDataForEmbedModel(rootDirectoryDataset, "YouTube")
    print("Data for embedding model loaded")
   
    #print("Builds LSTM model using tfidf features")
    #LSTModel(trainX_Text, trainY_Text,  valX_Text, valY_Text, testX_Text, testY_Text)
   
#    print("Builds LSTM model using embedding vectors")
#    LSTMModelUsingEmbedding(trainX_EmbedText, trainY_EmbedText,  valX_EmbedText, valY_EmbedText, testX_EmbedText, testY_EmbedText, vocab_size, max_words, embedding_matrix )
   
#    print("Builds CNN model using audio features")
#    CNNModel(trainX_Aud, trainY_Aud,  valX_Aud, valY_Aud, testX_Aud, testY_Aud)
    
    print("Joint model result using text embedding features and audio features") 
    jointModel(trainX_EmbedText, trainY_EmbedText,  valX_EmbedText, valY_EmbedText, testX_EmbedText, testY_EmbedText, trainX_Aud, trainY_Aud,  valX_Aud, valY_Aud, testX_Aud, testY_Aud)


if args.dataset == "TEAM":
    print ("Using TEAM datasets")
    # creates the dataframe object of youtube datasets
    inputTEAMAudioFeat, inputTEAMTextFeat, inputTEAMAudioTextFeat, outputTEAM = supervisedModels.getTEAMData(rootDirectoryDataset)
    
    #tfidfFeatures for lstm
    #print(tfidfInput.shape)

    #tfidfInput, output = supervisedModels.extractTextFeatures(rootDirectoryDataset)
    #print(tfidfInput.shape)

    
    #extracts reduced text features
#    train_X_textTf, test_X_textTf, train_Y_textTf, test_Y_textTf = supervisedModels.splitDataAfterPCA(tfidfInput, output,"text", "TEAM")
#    print("Reduced Text features after PCA: " + str(train_X_textTf.shape))
#    
    #extracts reduced audio features
    #train_X_Aud, test_X_Aud, train_Y_Aud, test_Y_Aud = supervisedModels.splitDataAfterPCA(inputTEAMAudioFeat, outputTEAM)
    #print("Reduced audio features: " + str(len(train_X_Aud[0])))
    
    #data for CNN model
    trainX_Aud, trainY_Aud,  valX_Aud, valY_Aud, testX_Aud, testY_Aud = getDataForNeuralModel(inputTEAMAudioFeat, outputTEAM, "TEAM")
    
    #data for LSTM model
    #trainX_Text, trainY_Text,  valX_Text, valY_Text, testX_Text, testY_Text = getDataForNeuralModel(train_X_textTf, test_X_textTf, train_Y_textTf, test_Y_textTf, "TEAM")
    print("Retrieve data for embedding model")
    #trainX_EmbedText, trainY_EmbedText,  valX_EmbedText, valY_EmbedText, testX_EmbedText, testY_EmbedText, vocab_size, max_words, embedding_matrix = getDataForEmbedModel(rootDirectoryDataset, "TEAM")
    print("Data for embedding model loaded")
   
    #print("Builds LSTM model using tfidf features")
    #LSTModel(trainX_Text, trainY_Text,  valX_Text, valY_Text, testX_Text, testY_Text)
   
#    print("Builds LSTM model using embedding vectors")
#    LSTMModelUsingEmbeddingTEAM(trainX_EmbedText, trainY_EmbedText,  valX_EmbedText, valY_EmbedText, testX_EmbedText, testY_EmbedText, vocab_size, max_words, embedding_matrix )
   
    print("Builds CNN model using audio features")
    CNNModelTEAM(trainX_Aud, trainY_Aud,  valX_Aud, valY_Aud, testX_Aud, testY_Aud)
    
#    print("Joint model result using text embedding features and audio features") 
#    jointModel(trainX_EmbedText, trainY_EmbedText,  valX_EmbedText, valY_EmbedText, testX_EmbedText, testY_EmbedText, trainX_Aud, trainY_Aud,  valX_Aud, valY_Aud, testX_Aud, testY_Aud)
