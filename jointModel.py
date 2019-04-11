import os
import supervisedModels
import argparse
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten, LSTM, Add, Embedding
from tensorflow.keras.utils import to_categorical
import numpy as np
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
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')    


    
def getDataForNeuralModel(X, Y,dataset,jm = False):
    
                        
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
    pca = PCA(0.95)
    # Fit on training set only.
    pca.fit(train_X)    
    # Apply transform to both the training set and the test set.
    train_X = pca.transform(train_X)
    testX = pca.transform(testX)    
    
    #divides the training data into further train and validation set
    trainX, valX, trainY, valY = train_test_split(train_X, train_Y, test_size=0.20, random_state=0)
    #expands the dimension of data to 3 dimension input
    trainX = np.expand_dims(trainX, 2)
    valX = np.expand_dims(valX, 2)
    testX = np.expand_dims(testX, 2)

    lb = LabelBinarizer()    
    if jm==True:
        trainY = lb.fit_transform(trainY)
        valY= lb.fit_transform(valY)
        testY= lb.fit_transform(testY)
   # optionally, transform labels from int to one-hot vectors
    if jm == False:
        if dataset == "YouTube":
            trainY = to_categorical(trainY, num_classes =3)
            valY = to_categorical(valY, num_classes =3)
            testY = to_categorical(testY, num_classes =3)
        if dataset == "TEAM":
            trainY = to_categorical(trainY, num_classes =2)
            valY = to_categorical(valY, num_classes =2)
            testY = to_categorical(testY, num_classes =2)

    return trainX, trainY, valX, valY, testX, testY, lb

def jointModelYouTube(audioInput, textInputPath, output, dataset ):
    
    print("loading data...")
    trainX_Aud, trainY_Aud,  valX_Aud, valY_Aud, testX_Aud, testY_Aud , lb = getDataForNeuralModel(audioInput, output, dataset, True)
    trainX_Text, trainY_Text,  valX_Text, valY_Text, testX_Text, testY_Text, vocab_size, max_words, embedding_matrix = getDataForEmbedModel(textInputPath, dataset)
    print("Data loaded")

    #loads the pretrained audio model
    atPath1 = os.path.join(os.getcwd(),"MasterThesis","PreTrainedModels","YouTubeModels","embeddingGlove(YT0.389).h5")        
    model_lstm = load_model(atPath1)
    model_lstm.summary()
    model_lstm.pop()

    #following freezes the layers of the model to use the pretrained weights 
    for layer in model_lstm.layers:
        layer.trainable= False
        
    #loads the trained cnn audio model
    atPath2 = os.path.join(os.getcwd(),"MasterThesis","PreTrainedModels","YouTubeModels","Amodel_cnn(0.48).h5")
    model_cnn = load_model(atPath2)
    model_cnn.summary()
    model_cnn.pop()
    
    #following freezes the layers of the model to use the pretrained weights 
    for layer in model_cnn.layers:
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
#                              verbose =2,validation_data=([valX_Text,valX_Aud], valY_Text))
#    
#    score, acc = jointModel.evaluate([testX_Text,testX_Aud], testY_Text)
#    
#    print('Test score:', score)
#    print('Test accuracy:', acc)
    
    print("Load the saved joint neural network model")
    atPath1 = os.path.join(os.getcwd(),"MasterThesis","PreTrainedModels","YouTubeModels","model_jnn(0.54).h5")

    model_L_jm = load_model(atPath1)
    model_L_jm.summary()
    
    scoreL, accL = model_L_jm.evaluate([testX_Text,testX_Aud], testY_Text)
    print('Test accuracy of pretrained joint model:', accL)
    #pred = model_L_jm.predict([testX_Text,testX_Aud])
    print("Predictions:")
    trainPreds = model_L_jm.predict([testX_Text,testX_Aud])
    target_names = [str(x) for x in lb.classes_]
    print(classification_report(testY_Text.argmax(axis=1),
                            trainPreds.argmax(axis=1),
                            target_names=target_names))    


def CNNModelYouTube(inputAudioFeat, output, dataset):

    print("loading data...")
    trainX, trainY, valX, valY, testX, testY, lb = getDataForNeuralModel(inputAudioFeat, output, dataset)
    print("Data loaded")
    
    num_features = len (trainX[0])

    model_cnn = Sequential()
    model_cnn.add(Conv1D(32, kernel_size = 3, activation = "tanh", input_shape = (num_features,1)))
    model_cnn.add(Conv1D(64, kernel_size = 3, activation = "tanh"))
    model_cnn.add(MaxPooling1D(pool_size=2))
    model_cnn.add(Dropout(0.20, name = 'dropout_cnn'))
    model_cnn.add(Flatten())
    model_cnn.add(Dense(95, activation= "tanh", name = "dense_cnn"))
    model_cnn.add(Dense(3, activation= "softmax", name = "cnn_output_layer"))
    
    
    model_cnn.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy' ])
    
    model_cnn.summary()
    
#    history = model_cnn.fit(trainX, trainY,
#            epochs=15,
#            verbose =2,
#            validation_data=(valX, valY))
     
#    score, acc = model_cnn.evaluate(testX, testY)
#   
#    print('Test score:', score)
#    print('Test accuracy:', acc)
    
    
    #following loads the pretrained model to produce reproducilble resluts since models does not produce the same results on every run         
    loadModelAtPath = os.path.join(os.getcwd(),"MasterThesis","PreTrainedModels","YouTubeModels","Amodel_cnn(0.48).h5")
    model_L_cnn = load_model(loadModelAtPath)
    
    scoreL, accL = model_L_cnn.evaluate(testX, testY)
    print('Test accuracy of pretrained Audio Model:', accL)

def LSTMModelUsingEmbeddingYouTube(inputTextPath, dataset):

    print("loading data...")
    trainX, trainY, valX, valY, testX, testY, vocab_size, max_words, embedding_matrix = getDataForEmbedModel(inputTextPath, dataset)
    print("data loaded")
 
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
    
    
    #fitting the model
#    history = model.fit(trainX, trainY,
#                epochs=40,
#                verbose =2,
#                validation_data=(valX, valY))
        
    #test the model
#    score, acc = model.evaluate(testX, testY)
#    print('Test score:', score)
#    print('Test accuracy:', acc)
    
    #loads and test the model 
    atPath = os.path.join(os.getcwd(),"MasterThesis","PreTrainedModels","YouTubeModels","embeddingGlove(YT0.389).h5")
    model_l  = load_model(atPath)  
      
    score1, acc1 = model_l.evaluate(testX, testY)
    print('Test accuracy:', acc1)
    
def jointModelTEAM(inputAudioFeat, inputTextPath, output, dataset):

    print("loading data...")
    trainX_Aud, trainY_Aud,  valX_Aud, valY_Aud, testX_Aud, testY_Aud , lb = getDataForNeuralModel(inputAudioFeat, output, dataset, True)
    trainX_Text, trainY_Text,  valX_Text, valY_Text, testX_Text, testY_Text, vocab_size, max_words, embedding_matrix = getDataForEmbedModel(inputTextPath, dataset)
    print("Data loaded")


    #loads the pretrained audio model
#    atPath1 = os.path.join(os.getcwd(),"PreTrainedModels","embeddingGlove(YT0.389).h5")  
    atPath1 =  os.path.join(os.getcwd(),"MasterThesis","PreTrainedModels","TeamModels","embeddingGlove(T0.78).h5")          
    model_lstm = load_model(atPath1)
    model_lstm.summary()
    model_lstm.pop()

    #following freezes the layers of the model to use the pretrained weights 
    for layer in model_lstm.layers:
        layer.trainable= False       
        
    
    #loads the trained text model
    atPath2 = os.path.join(os.getcwd(),"MasterThesis","PreTrainedModels","TeamModels","Amodel_cnnTEAM(0.51).h5")
    model_cnn = load_model(atPath2)
    model_cnn.summary()
    model_cnn.pop()
    
    #following freezes the layers of the model to use the pretrained weights 
    for layer in model_cnn.layers:
        layer.trainable= False
 
    #following concatenates the output of the two models in to a fully connected layer
    concat_layer =  Add()([model_lstm.output,model_cnn.output])
    concat_layer = Dense(800, activation = "relu",name = 'fully_connected_layer')(concat_layer)
    concat_layer = Dense(200, activation = "relu",name = 'connected_layer')(concat_layer) 
    output_layer = Dense(2, activation = "softmax", name = 'final_Dense_layer')(concat_layer)
    
    jointModel = Model ([model_lstm.input,model_cnn.input],output_layer)
    
    #compiles the joint model
    jointModel.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy' ])
                
    #summary of the joint model
    jointModel.summary()
        
   # trains the joint model
#    history = jointModel.fit([trainX_Text,trainX_Aud], trainY_Text, epochs=15,
#                             verbose =2,validation_data=([valX_Text,valX_Aud], valY_Text))
    
#    score, acc = jointModel.evaluate([testX_Text,testX_Aud], testY_Text)
    
#    print('Test score:', score)
#    print('Test accuracy:', acc)

    print("Load the saved joint neural network model")
    jnnPath = os.path.join(os.getcwd(),"MasterThesis","PreTrainedModels","TeamModels","model_jnnYTEAM(0.79).h5")

    model_L_jm = load_model(jnnPath)
    model_L_jm.summary()
    
    scoreL, accL = model_L_jm.evaluate([testX_Text,testX_Aud], testY_Text)
    print('Test accuracy of YouTube joint model:', accL)
    trainPreds = model_L_jm.predict([testX_Text,testX_Aud])
    target_names = [str(x) for x in lb.classes_]
    print(classification_report(testY_Text.argmax(axis=1),
                            trainPreds.argmax(axis=1),
                            target_names=target_names))    
                  

    
def LSTMModelUsingEmbeddingTEAM(inputTextPath, dataset):

    print("loading data...")
    trainX, trainY, valX, valY, testX, testY, vocab_size, max_words, embedding_matrix = getDataForEmbedModel(inputTextPath, dataset)
    print("Data loaded")

    #defining the model
    model = Sequential()
    model.add(Embedding(vocab_size, 300, input_length = max_words, trainable = False, weights=[embedding_matrix]))
    model.add(LSTM(800, return_sequences=True))
    model.add(Dropout(0.2, name = "dropout_lstm_1"))
    model.add(LSTM(600, return_sequences=True))
    model.add(Dropout(0.2, name = "dropout_lstm_2"))
    model.add(Dense(400, activation = "tanh", name = "penultimate_layer"))
    model.add(Dense(2, activation = "sigmoid"))
    
    #compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    #summary of the model
    model.summary()
    
    #fitting the model
#    history = model.fit(trainX, trainY,
#                epochs=25,
#                verbose =2,
#                validation_data=(valX, valY))
    
    #test the model
#    score, acc = model.evaluate(testX, testY)
#    print('Test score:', score)
#    print('Test accuracy:', acc)
    
    #loads the pretrained model and evaluate it on the test set 
    atPath = os.path.join(os.getcwd(),"MasterThesis","PreTrainedModels","TeamModels","embeddingGlove(T0.78).h5")
    model_l  = load_model(atPath)  
      
    score1, acc1 = model_l.evaluate(testX, testY)
    print('Test accuracy:', acc1)
    

def CNNModelTEAM(inputAudioFeat, output, dataset):

    print("loading data...")
    trainX, trainY, valX, valY, testX, testY, lb = getDataForNeuralModel(inputAudioFeat, output, dataset)
    print("Data loaded")
    
    
    num_features = len (trainX[0])
    model_cnn = Sequential()
    model_cnn.add(Conv1D(64, kernel_size = 3, activation = "tanh", input_shape = (num_features,1)))
    model_cnn.add(Conv1D(128, kernel_size = 3, activation = "tanh"))
    model_cnn.add(MaxPooling1D(pool_size=2))   
    model_cnn.add(Dropout(0.20, name = 'dropout_cnn1'))
    model_cnn.add(Flatten())
    model_cnn.add(Dense(400, activation= "tanh", name = "dense_cnn"))
    model_cnn.add(Dense(2, activation= "softmax", name = "cnn_output_layer"))
    
    
    model_cnn.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy' ])
    
    model_cnn.summary()
    
#    history = model_cnn.fit(trainX, trainY,
#            epochs=10,
#            verbose =2,
#            validation_data=(valX, valY))


    score, acc = model_cnn.evaluate(testX, testY)
   
    print('Test score:', score)
    print('Test accuracy:', acc)
    
    #following loads the pretrained model to produce reproducilble resluts since models does not produce the same results on every run         
    loadModelAtPath = os.path.join(os.getcwd(),"MasterThesis","PreTrainedModels","TeamModels","Amodel_cnnTEAM(0.51).h5")
    model_L_cnn = load_model(loadModelAtPath)
    
    scoreL, accL = model_L_cnn.evaluate(testX, testY)
    print('Test accuracy of audio model using cnn:', accL)  
          
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
    #finds the maximum number of words 		
    print("Before normalization")
    max_words = findMaxNumberOfWords(df['Text'])    
    # applies text normalization
    df['Text'] = df['Text'].map(lambda x: normalizeTextData(x, True, True))      
    #finds the maximum number of words 
    print("After normalization")		
    max_words = findMaxNumberOfWords(df['Text'])   
    #splits the data into training and testing
    trainX, testX, trainY, testY = train_test_split(df['Text'], df['sentiment'], test_size = 0.20, random_state = 0 )    
#    tok = Tokenizer()
#    tok.fit_on_texts(df['Text'])
#       
    # saves the tokenizer object
    if dataset == "YouTube":  
        saveTokenizerAtPath = os.path.join(os.getcwd(),"MasterThesis","PreTrainedModels","YouTubeModels","tokenizerYT().pickle")
    if dataset == "TEAM":
        print("Loaded team tokenizer")
        saveTokenizerAtPath = os.path.join(os.getcwd(),"MasterThesis","PreTrainedModels","TeamModels","tokenizerTEAM().pickle")    
    
    # loads the saved tokenizer
    f = open(saveTokenizerAtPath, 'rb') 
    tok = pickle.load(f)
    trainX_testTok = tok.texts_to_sequences(trainX)
    print(trainX_testTok[0])
    
    vocab_size = len(tok.word_index) + 1
    print ("Vocabulary size: " + str(vocab_size))
    
    trainX_seq = tok.texts_to_sequences(trainX)
    testX_seq = tok.texts_to_sequences(testX)
        
    pad_trainX_seq  = pad_sequences(trainX_seq, maxlen = max_words, padding = 'post')
    pad_testX_seq = pad_sequences(testX_seq, maxlen = max_words, padding = 'post')

    if dataset == "YouTube":
        trainY_Text = to_categorical(trainY, num_classes =3)
        testY_Text = to_categorical(testY, num_classes =3)
        #splitting the training data into train and validation set
        trainX_Text, valX_Text, trainY_Text, valY_Text = train_test_split(pad_trainX_seq, trainY_Text, test_size = 0.20, random_state = 0)
    if dataset == "TEAM":
        trainY_Text = to_categorical(trainY, num_classes =2)
        testY_Text = to_categorical(testY, num_classes =2)    
        #splitting the training data into train and validation set
        trainX_Text, valX_Text, trainY_Text, valY_Text = train_test_split(pad_trainX_seq, trainY_Text, test_size = 0.20, random_state = 0)    
    #gets the embedding matrix
    if dataset == "YouTube":        
        print("Loaded YouTube embedding matrix")
        saveEmbeddingMatrix = os.path.join(os.getcwd(),"MasterThesis","PreTrainedModels","YouTubeModels","gloveEmbedYT.pickle")
    if dataset == "TEAM":
        print("Loaded TEAM embedding matrix")
        saveEmbeddingMatrix = os.path.join(os.getcwd(),"MasterThesis","PreTrainedModels","TeamModels","gloveEmbedTEAM.pickle")        
    
    f = open(saveEmbeddingMatrix, 'rb') 
    embedding_matrix = pickle.load(f)
    
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
    print('Loaded %s word vectors using 300 dimensional glove embeddings.' % len(embeddings_index))

    #create a weight matrix for words in training docs
    embedding_matrix = zeros((vocab_size, 300))
    for word, i in tok.word_index.items():
    	embedding_vector = embeddings_index.get(word)
    	if embedding_vector is not None:
    		embedding_matrix[i] = embedding_vector
    
#    saveEmbeddingMatrix = ""
    #save the embedding matrix
#    if dataset == "YouTube":
#        saveEmbeddingMatrix = os.path.join(os.getcwd(),"gloveEmbedYT.pickle")
#        with open(saveEmbeddingMatrix, 'wb') as handle:
#            pickle.dump(embedding_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #save the embedding matrix
#    if dataset == "TEAM":
#        saveEmbeddingMatrix = os.path.join(os.getcwd(),"gloveEmbedTEAM.pickle")
#        with open(saveEmbeddingMatrix, 'wb') as handle:
#            pickle.dump(embedding_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
    
    return embedding_matrix


def allModelResults(inputAudioFeat, inputTextPath, output, dataset):
    """
    * Loads pretrained models and evaluate the models on the test dataset
    """
    
    print("loading data...")
    trainX_Aud, trainY_Aud,  valX_Aud, valY_Aud, testX_Aud, testY_Aud , lb = getDataForNeuralModel(inputAudioFeat, output, dataset)
    trainX_Text, trainY_Text,  valX_Text, valY_Text, testX_Text, testY_Text, vocab_size, max_words, embedding_matrix = getDataForEmbedModel(inputTextPath, dataset)
    print("Data loaded")
    
    if dataset == "YouTube":     
        
        print("Joint model results on YouTube")
        youTubeModelsPath = os.path.join(os.getcwd(),"MasterThesis","PreTrainedModels","YouTubeModels")
        lstm_model = os.path.join(youTubeModelsPath,"embeddingGlove(YT0.389).h5")
        cnn_model =  os.path.join(youTubeModelsPath,"Amodel_cnn(0.48).h5")
        jnn_model = os.path.join (youTubeModelsPath, "model_jnn(0.54).h5")

        print ("YouTube LSTM model summary")        
        lstm = load_model(lstm_model)
        lstm.summary()
        scoreL, accL = lstm.evaluate(testX_Text, testY_Text)
        print('LSTM accuracy:', accL)
       
        print ("YouTube CNN model summary")
        cnn = load_model(cnn_model)
        cnn.summary()
        score_C, acc_C= cnn.evaluate(testX_Aud, testY_Aud)
        print('LSTM accuracy:', acc_C)
       
        jnn = load_model(jnn_model)
        print ("YoTube Joint neural network summary")
        jnn.summary()
        score_J, acc_J= jnn.evaluate([testX_Text,testX_Aud], testY_Text)
        print('LSTM accuracy:', acc_J)
        
    if dataset == "TEAM":

        print("Joint model results on TEAM")

        youTubeModelsPath = os.path.join(os.getcwd(),"MasterThesis","PreTrainedModels","TeamModels")
        lstm_model = os.path.join(youTubeModelsPath,"embeddingGlove(T0.78).h5")
        cnn_model =  os.path.join(youTubeModelsPath,"Amodel_cnnTEAM(0.51).h5")
        jnn_model = os.path.join (youTubeModelsPath, "model_jnnYTEAM(0.79).h5")
        
        print ("TEAM LSTM model summary")        
        lstm = load_model(lstm_model)
        lstm.summary()
        scoreL, accL = lstm.evaluate(testX_Text, testY_Text)
        print('TEAM accuracy:', accL)
       
        print ("TEAM CNN model summary")
        cnn = load_model(cnn_model)
        cnn.summary()
        score_C, acc_C= cnn.evaluate(testX_Aud, testY_Aud)
        print('TEAM accuracy:', acc_C)
       
        jnn = load_model(jnn_model)
        print ("TEAM Joint neural network summary")
        jnn.summary()
        score_J, acc_J= jnn.evaluate([testX_Text,testX_Aud], testY_Text)
        print('TEAM accuracy:', acc_J)
        

   
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

    print("Joint model result using text embedding features and audio features") 
    jointModelYouTube(inputYouTubeAudioFeat, rootDirectoryDataset, outputYouTube, "YouTube" )

#    allModelResults(inputYouTubeAudioFeat,rootDirectoryDataset, outputYouTube, "YouTube")

#    print("Builds LSTM model using embedding vectors")
#    LSTMModelUsingEmbeddingYouTube(rootDirectoryDataset, "YouTube")

#    print("Builds CNN model using audio features")
#    CNNModelYouTube(inputYouTubeAudioFeat, outputYouTube, "YouTube")
    

if args.dataset == "TEAM":
    # creates the dataframe object of youtube datasets
    inputTEAMAudioFeat, inputTEAMTextFeat, inputTEAMAudioTextFeat, outputTEAM = supervisedModels.getTEAMData(rootDirectoryDataset)

    print("Joint model result using text embedding features and audio features")     
    jointModelTEAM(inputTEAMAudioFeat, rootDirectoryDataset, outputTEAM, "TEAM" )

#    allModelResults(inputTEAMAudioFeat,rootDirectoryDataset, outputTEAM, "TEAM")

#    print("Builds LSTM model using embedding vectors")
#    LSTMModelUsingEmbeddingTEAM(rootDirectoryDataset, "TEAM")

#    print("Builds CNN model using audio features")
#    CNNModelTEAM(inputTEAMAudioFeat, outputTEAM, "TEAM")
