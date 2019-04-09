# -*- coding: utf-8 -*-
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
#from yellowbrick.classifier import ClassificationReport
from sklearn.feature_extraction.text import TfidfVectorizer



       
#splits the data into training and testing data and then reduce the features using PCA      
def splitDataAfterPCA(input, output):
    train_data, test_data, train_lbl, test_lbl = train_test_split(input, output, test_size=0.20, random_state=0)
    
    scaler = StandardScaler()
    # Fit on training set only.
    scaler.fit(train_data)

    # Apply transform to both the training set and the test set.    
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    
     
    pca = PCA(0.95) 
     # Fit on training set only.
    pca.fit(train_data)    
     # Apply transform to both the training set and the test set.
    train_data = pca.transform(train_data)
    test_data = pca.transform(test_data)
    
#    if feature == "text":
#        pca = PCA(215) 
#        # Fit on training set only.
#        pca.fit(train_data) 
#        # Apply transform to both the training set and the test set.
#        train_data = pca.transform(train_data)
#        test_data = pca.transform(test_data)
    
    return train_data, test_data, train_lbl, test_lbl 
    

def findBestEstimator(model, tuned_parameters, X_train, X_test, y_train, y_test, dataset):
    """
    * trains a given model with training data provided 
    * finds the best hyperparameters for the model
    * computers precision, recall, f1, and accuarcy scores of the final model
    """
    clf = GridSearchCV(model, tuned_parameters, cv=10)
#                        scoring='%s_micro' % score)
#        visualizer = ClassificationReport(clf,classes = classes )
#        visualizer.fit(X_train, y_train)  # Fit the visualizer and the model
#        visualizer.score(X_test, y_test)  # Evaluate the model on the test data
#        g = visualizer.poof()             # Draw/show/poof the dataprint("Using visualizer")
    
    
    #to avoid unnecessary warning while training the model using YouTube dataset to build the model
    if dataset == "YouTube":
        clf.fit(X_train, y_train.values.ravel())
    if dataset == "TEAM":
        clf.fit(X_train, y_train)
        
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Detailed classification report:")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
    print("Accuracy_score: ")
    print(accuracy_score(y_true, y_pred))
    
    

#Gives SVC models (SVC, LinearSVC and NuSVC) results 
def getSVCModelResults(train_X_Com, test_X_Com, train_Y_Com, test_Y_Com, features, dataset):    
    if dataset == "YouTube":
        # hyperparameter set to find best hyperparameter for the SVC model
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.01, 0.001, 0.0001],
                                'C': [0.001,0.01,0.1,1, 10, 100, 1000]},
                                {'kernel': ['linear'],'gamma': [0.01, 0.001, 0.0001], 'C': [0.001,0.01,0.1,1, 10, 100, 1000]}]
        
        # hyperparameter set to find best hyperparameter for the linearSVC model
        tuned_parameters_lin = [{'dual': [False],'C': [1, 10, 100, 1000],'multi_class' :['ovr']},
                                {'dual': [False],'C': [1, 10, 100, 1000],'multi_class' :['crammer_singer']},
                                {'dual': [True],'C': [1, 10, 100, 1000],'multi_class' :['crammer_singer']}]
         
        # hyperparameter set to find best hyperparameter for the nuSVC model
        tuned_parameters_nuSVC = [{'nu':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8], 'gamma': [0.01, 0.001, 0.0001],
                                   'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'decision_function_shape' : ['ovo', 'ovr']}]
            
        if features == "audio":
            #following is the best hyperparameter for SVC model
            svc_best_params = [{'C': [10], 'gamma':[0.001], 'kernel': ['rbf']}]
            #following is the best hyperparameter for linearSVC model
            linearSVC_best_params = [{'C' : [1], 'dual' : [False], 'multi_class' : ['crammer_singer']}]
            #following is the best hyperparameter for NuSVC model
            nuSVC_best_params = [{'gamma' : [0.01], 'nu' : [0.6], 'decision_function_shape' : ['ovo'], 'kernel' : ['poly'] }]
            
        if features == "text":
            #following is the best hyperparameter for SVC model
            svc_best_params = [{'C': [100], 'gamma':[0.01], 'kernel': ['rbf']}]
            #following is the best hyperparameter for linearSVC model
            linearSVC_best_params = [{'C' : [10], 'dual' : [False], 'multi_class' : ['crammer_singer']}]
            #following is the best hyperparameter for NuSVC model
            nuSVC_best_params = [{'decision_function_shape': ['ovo'], 'gamma': [0.01], 'kernel': ['linear'], 'nu': [0.7] }]
        
        if features == "audio_text":
            #following is the best hyperparameter for SVC model
            svc_best_params = [{'C': [0.01], 'gamma':[0.01], 'kernel': ['linear']}]
            #following is the best hyperparameter for linearSVC model
            linearSVC_best_params = [{'C' : [1], 'dual' : [False], 'multi_class' : ['crammer_singer']}]
            #following is the best hyperparameter for NuSVC model
            nuSVC_best_params = [{'decision_function_shape': ['ovo'], 'gamma': [0.01], 'kernel': ['linear'], 'nu': [0.6]}]
       
    if dataset == "TEAM":
        print("inside team")
        # hyperparameter set to find best hyperparameter for the SVC model
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.01, 0.001, 0.0001],
                                'C': [0.001,0.01,0.1,1, 10, 100, 1000]},
                                {'kernel': ['linear'],'gamma': [0.01, 0.001, 0.0001], 'C': [0.001,0.01,0.1,1, 10, 100, 1000]}]
        
        # hyperparameter set to find best hyperparameter for the linearSVC model
        tuned_parameters_lin = [{'dual': [False],'C': [1, 10, 100, 1000],'multi_class' :['ovr']},
                                {'dual': [False],'C': [1, 10, 100, 1000],'multi_class' :['crammer_singer']},
                                {'dual': [True],'C': [1, 10, 100, 1000],'multi_class' :['crammer_singer']}]
         
        # hyperparameter set to find best hyperparameter for the nuSVC model
        tuned_parameters_nuSVC = [{'nu':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8], 'gamma': [0.01, 0.001, 0.0001],
                                   'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'decision_function_shape' : ['ovo', 'ovr']}]
            
        if features == "audio":
            #following is the best hyperparameter for SVC model
            svc_best_params = [{'C': [1], 'gamma':[0.0001], 'kernel': ['rbf']}]
            #following is the best hyperparameter for linearSVC model
            linearSVC_best_params = [{'C' : [1], 'dual' : [False], 'multi_class' : ['crammer_singer']}]
            #following is the best hyperparameter for NuSVC model
            nuSVC_best_params = [{'gamma' : [0.0001], 'nu' : [0.8]}]
            
        if features == "text":
            #following is the best hyperparameter for SVC model
            svc_best_params = [{'C': [0.001], 'gamma':[0.01], 'kernel': ['linear']}]
            #following is the best hyperparameter for linearSVC model
            linearSVC_best_params = [{'C' : [10], 'dual' : [False], 'multi_class' : ['ovr']}]
            #following is the best hyperparameter for NuSVC model
            nuSVC_best_params = [{'gamma': [0.0001], 'nu': [0.2]}]
        
        if features == "audio_text":
            #following is the best hyperparameter for SVC model
            svc_best_params = [{'C': [10], 'gamma':[0.0001], 'kernel': ['rbf']}]
            #following is the best hyperparameter for linearSVC model
            linearSVC_best_params = [{'C' : [1], 'dual' : [False], 'multi_class' : ['crammer_singer']}]
            #following is the best hyperparameter for NuSVC model
            nuSVC_best_params = [{'gamma': [0.0001], 'nu': [0.1]}]
    
        
        
    print("Summary of SVC model results: ")
    
    #following finds the best hyperparameter for the model and report results
    #findBestEstimator(SVC(),tuned_parameters,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com, dataset)
    
    #following trains the model with the best hyperparameter for the model and report results
    findBestEstimator(SVC(),svc_best_params,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com, dataset)
    
    print("Summary of LinearSVC model results: ")
    
    #following finds the best hyperparameter for the model and report results
    #findBestEstimator(LinearSVC(),tuned_parameters_lin,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com, dataset)
    
    #following uses the best hyperparameter for the model and report results
    findBestEstimator(LinearSVC(),linearSVC_best_params,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com, dataset)
    
    print("Summary of NuSVC model results: ")    
    #following finds the best hyperparameter for the model and report results
    #findBestEstimator(NuSVC(),tuned_parameters_nuSVC,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com, dataset)
 
    #following uses the best hyperparameter for the model and report results
    findBestEstimator(NuSVC(),nuSVC_best_params,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com, dataset)
    
#gives RandomForestClassifier models (RandomForestClassifier, ExtraTreesClassifier, DecisionTreeClassifier) results
def getRandomForestClassifierResults(train_X_Com, test_X_Com, train_Y_Com, test_Y_Com, features, dataset):    
    # hyperparameter set to find best hyperparameter for the DecisionTree model
    tune_param_DClf = [{'max_depth':[3],'min_samples_leaf':[1]}]
    if dataset == "YouTube":
        if features == "audio":
            # hyperparameter set to find best hyperparameter for the RandomForest and ExtraTrees model
            tune_params_RF = [{'n_estimators': [10,20,30,40,50,60],'max_features': ['auto', 'log2'],'max_depth':[None],'min_samples_split':[2], 'random_state':[0],'n_jobs':[1]}]
        
            #following is the best hyperparameter for RandomForest model
            #RF_best_params = [{'max_depth':[None], 'max_features' : [None], 'min_samples_split':[2], 'n_estimators' : [10], 'n_jobs' : [1], 'random_state' : [0] }]
            RF_best_params = [{'max_depth': [None], 'max_features': ['log2'], 'min_samples_split': [2], 'n_estimators': [20], 'n_jobs': [1], 'random_state': [0]}]
            #following is the best hyperparameter for ExtraTrees model
            #ET_best_params = [{'max_depth':[None], 'max_features' : [None], 'min_samples_split':[2], 'n_estimators' : [40], 'n_jobs' : [1], 'random_state' : [0] }]
            ET_best_params = [{'max_depth': [None], 'max_features': ['auto'], 'min_samples_split': [2], 'n_estimators': [30], 'n_jobs': [1], 'random_state': [0]}]
            
        if features == "text":
            # hyperparameter set to find best hyperparameter for the RandomForest and ExtraTrees model
            tune_params_RF = [{'n_estimators': [10,20,30,40,50,60],'max_features': ['auto', 'log2'],'max_depth':[None],'min_samples_split':[2], 'random_state':[0],'n_jobs':[1]}]
        
            #following is the best hyperparameter for RandomForest model
            RF_best_params = [{'max_depth':[None], 'max_features' : ['auto'], 'min_samples_split':[2], 'n_estimators' : [50], 'n_jobs' : [1], 'random_state' : [0] }]
            #following is the best hyperparameter for ExtraTrees model
            ET_best_params = [{'max_depth':[None], 'max_features' : ['auto'], 'min_samples_split':[2],'n_estimators' : [30], 'n_jobs' : [1], 'random_state' : [0] }]
        
        if features == "audio_text":
            # hyperparameter set to find best hyperparameter for the RandomForest and ExtraTrees model
            tune_params_RF = [{'n_estimators': [10,20,30,40,50,60],'max_features': [None],'max_depth':[None],'min_samples_split':[2], 'random_state':[0],'n_jobs':[1]}]
        
            #following is the best hyperparameter for RandomForest model
            RF_best_params = [{'max_depth':[None], 'max_features' : [None], 'min_samples_split':[2],  'n_estimators' : [60], 'n_jobs' : [1], 'random_state' : [0] }]
            #following is the best hyperparameter for ExtraTrees model
            ET_best_params = [{'max_depth':[None], 'max_features' : [None], 'min_samples_split':[2], 'n_estimators' : [60], 'n_jobs' : [1], 'random_state' : [0] }]

    if dataset == "TEAM":
        if features == "audio":
            # hyperparameter set to find best hyperparameter for the RandomForest and ExtraTrees model
            tune_params_RF = [{'n_estimators': [10,20,30,40,50,60],'max_features': ['auto', 'log2'],'max_depth':[None],'min_samples_split':[2], 'random_state':[0],'n_jobs':[1]}]
        
            #following is the best hyperparameter for RandomForest model
            #RF_best_params = [{'max_depth':[None], 'max_features' : [None], 'min_samples_split':[2], 'n_estimators' : [10], 'n_jobs' : [1], 'random_state' : [0] }]
            RF_best_params = [{'max_depth': [None], 'max_features': [None], 'min_samples_split': [2], 'n_estimators': [60], 'n_jobs': [1], 'random_state': [0]}]
            #following is the best hyperparameter for ExtraTrees model

            ET_best_params = [{'max_depth': [None], 'max_features': [None], 'min_samples_split': [2], 'n_estimators': [60], 'n_jobs': [1], 'random_state': [0]}]
            
        if features == "text":
            # hyperparameter set to find best hyperparameter for the RandomForest and ExtraTrees model
            tune_params_RF = [{'n_estimators': [10,20,30,40,50,60],'max_features': ['auto', 'log2'],'max_depth':[None],'min_samples_split':[2], 'random_state':[0],'n_jobs':[1]}]
        
            #following is the best hyperparameter for RandomForest model
            RF_best_params = [{'max_depth':[None], 'max_features' : ['auto'], 'min_samples_split':[2], 'n_estimators' : [50], 'n_jobs' : [1], 'random_state' : [0] }]
            #following is the best hyperparameter for ExtraTrees model
            ET_best_params = [{'max_depth':[None], 'max_features' : ['auto'], 'min_samples_split':[2],'n_estimators' : [30], 'n_jobs' : [1], 'random_state' : [0] }]
        
        if features == "audio_text":
            # hyperparameter set to find best hyperparameter for the RandomForest and ExtraTrees model
            tune_params_RF = [{'n_estimators': [10,20,30,40,50,60],'max_features': [None],'max_depth':[None],'min_samples_split':[2], 'random_state':[0],'n_jobs':[1]}]
        
            #following is the best hyperparameter for RandomForest model
            RF_best_params = [{'max_depth':[None], 'max_features' : [None], 'min_samples_split':[2],  'n_estimators' : [60], 'n_jobs' : [1], 'random_state' : [0] }]
            #following is the best hyperparameter for ExtraTrees model
            ET_best_params = [{'max_depth':[None], 'max_features' : [None], 'min_samples_split':[2], 'n_estimators' : [40], 'n_jobs' : [1], 'random_state' : [0] }]
    
    print("Summary of RandomForestClassifier model results: ")    
    #following finds the best hyperparameter for the model and report results
    #findBestEstimator(RandomForestClassifier(),tune_params_RF,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com, dataset)
    
    # following uses the best hyperparameter set for the model and report results
    findBestEstimator(RandomForestClassifier(),RF_best_params,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com, dataset)
    
    print("Summary of ExtraTreesClassifier model results: ")    
    #following finds the best hyperparameter for the model and report results 
    #findBestEstimator(ExtraTreesClassifier(),tune_params_RF,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com, dataset)
    
    # following uses the best hyperparameter set for the model and report results
    findBestEstimator(ExtraTreesClassifier(),ET_best_params,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com, dataset)
   
    print("Summary of DecisionTreeClassifier model results: ")        
    # following uses the best hyperparameter set for the model and report results
    findBestEstimator(DecisionTreeClassifier(),tune_param_DClf,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com, dataset)

#gives Logistic Regression model results
def getLogisticRegressionResults(train_X_Com, test_X_Com, train_Y_Com, test_Y_Com, features, dataset):        
    tune_params_LR1 = [{'penalty':['l1'], 'solver':['liblinear','saga'],'max_iter':[10000],'multi_class':['ovr']}]
    tune_params_LR2 = [{'penalty':['l2'], 'solver':['newton-cg','lbfgs','sag'], 'max_iter':[10000],'multi_class':['multinomial','ovr']}]

    if dataset == "YouTube":    
        if features == "audio":
            best_params = [{'penalty':['l1'], 'solver':['saga'],'max_iter':[10000],'multi_class':['ovr']}]
        if features == "text":
            best_params = [{'penalty':['l2'], 'solver':['newton-cg'],'max_iter':[10000],'multi_class':['multinomial']}]
        if features == "audio_text":
            best_params = [{'penalty':['l2'], 'solver':['newton-cg'],'max_iter':[10000],'multi_class':['ovr']}]
                    
    if dataset == "TEAM":    
        if features == "audio":
            best_params = [{'penalty':['l2'], 'solver':['newton-cg'],'max_iter':[10000],'multi_class':['multinomial']}]
        if features == "text":
            best_params = [{'penalty':['l2'], 'solver':['saga'],'max_iter':[10000],'multi_class':['multinomial']}]
        if features == "audio_text":
            best_params = [{'penalty':['l2'], 'solver':['newton-cg'],'max_iter':[10000],'multi_class':['multinomial']}]

    print("Summary of LogisticRegression using model results: ")            
    findBestEstimator(LogisticRegression(),best_params,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com, dataset)

#gets the results of all models used at once 
def getSupervisedModelResults(train_X_Com, test_X_Com, train_Y_Com, test_Y_Com, features, dataset):
    print("SVM models results:")
    getSVCModelResults(train_X_Com, test_X_Com, train_Y_Com, test_Y_Com, features, dataset)
    print("RandomForest classifier models results:")
    getRandomForestClassifierResults(train_X_Com, test_X_Com, train_Y_Com, test_Y_Com, features, dataset)
    print("LogisticRegression models results:")
    getLogisticRegressionResults(train_X_Com, test_X_Com, train_Y_Com, test_Y_Com, features, dataset)


def extractTextFeatures (rootDirectory, dataset):
    print(rootDirectory)
    textualData = ""
    if dataset == "YouTube":
        print("Loading YouTube text dataset from path: ")
        textualData = os.path.join (rootDirectory, "textYT.csv")        
        print(textualData)
        print("YouTube text dataset loaded into memory")

    if dataset == "TEAM":
        print("Loading TEAM text dataset from path: ")
        textualData = os.path.join (rootDirectory, "textDataTEAM.csv")
        print(textualData)
        print("TEAM text dataset loaded into memory ")


    textDf = pd.read_csv(textualData)
    inputText = textDf['Text'].values.tolist()
    #print(inputText)
    #inputText = textDf.loc[:,textDf.columns!="sentiment"]
    #inp = inputText
    output = textDf.loc[:,textDf.columns == "sentiment"]
    
    print("Extracting 2 gram text features")
    tfidf = TfidfVectorizer(ngram_range= (2,2))
    features = tfidf.fit_transform(inputText)
    data = pd.DataFrame(
            features.todense(), 
            columns=tfidf.get_feature_names())
    
    #creating a csv file to store the the text features
    if dataset == "YouTube":    
        atPath = os.path.join(os.getcwd(),"MasterThesis","datasets","YouTube","tfidf2gramFeat.csv")
        
        #saves the data at the above path
        #data.to_csv(atPath, encoding='utf-8')
        
        df = pd.read_csv(atPath)
        input_data = df.loc[:, df.columns != 'sentiment'] 
        print ("TF-IDF text features shape of YouTube dataset: " + str(input_data.shape))
	
         
    if dataset == "TEAM":
        atPath = os.path.join(os.getcwd(),"MasterThesis","datasets","TEAM","tfidf2gramFeatTEAM.csv")
        
        #saves the data at the above path
        #data.to_csv(atPath, encoding='utf-8')
        
        df = pd.read_csv(atPath)
        input_data = df.loc[:, df.columns != 'sentiment'] 
        print ("TF-IDF text features shape of TEAM dataset: " + str(input_data.shape))
	
        
    return input_data, output


def getYouTubeData(rootDirectoryYouTubeDatasets):
    	
    # YouTube dataset features file path
    youTubeAudioFeatPath = os.path.join(rootDirectoryYouTubeDatasets,"YouTubeAudioFeat.csv")
    youTubeTextFeatPath = os.path.join(rootDirectoryYouTubeDatasets,"YouTubeTextFeat.csv")
    youTubeAudioTextFeatPath = os.path.join(rootDirectoryYouTubeDatasets,"YouTubeAudioTextFeat.csv")
   
    #creates the dataframe object of Audio Features
    youTubeAudioFeat = pd.read_csv(youTubeAudioFeatPath)  
    inputYouTubeAudioFeat = youTubeAudioFeat.loc[:, youTubeAudioFeat.columns != 'sentiment']  
    print ("YouTube audio features shape: " + str(inputYouTubeAudioFeat.shape))
    
    #creates the dataframe object of Text Features
    youTubeTextFeat = pd.read_csv(youTubeTextFeatPath)
    inputYouTubeTextFeat = youTubeTextFeat.loc[:, youTubeTextFeat.columns != 'sentiment']
    print ("YouTube text features shape: " + str(inputYouTubeTextFeat.shape))

    #creates the dataframe object of Audio and Text Features    
    youTubeAudioTextFeat = pd.read_csv(youTubeAudioTextFeatPath)   
    inputYouTubeAudioTextFeat = youTubeAudioTextFeat.loc[:, youTubeAudioTextFeat.columns != 'sentiment'] 
    print ("YouTube combine audio and text features shape: " + str(inputYouTubeAudioTextFeat.shape))
	
    outputYouTube = youTubeTextFeat.loc[:, youTubeTextFeat.columns == 'sentiment']
    print(str(outputYouTube.shape))
    return  inputYouTubeAudioFeat, inputYouTubeTextFeat, inputYouTubeAudioTextFeat, outputYouTube

def getTEAMData(rootDirectoryTEAMDatasets):    

    # TEAM dataset features file path
    teamAudioFeatPath = os.path.join(rootDirectoryTEAMDatasets,"TEAMAudioFeat.csv")
    teamTextFeatPath = os.path.join(rootDirectoryTEAMDatasets,"TEAMTextFeat.csv")
    teamAudioTextFeatPath = os.path.join(rootDirectoryTEAMDatasets,"TEAMAudioTextFeat.csv")
    
    #creates the dataframe object of Audio Features
    teamAudioFeat = pd.read_csv(teamAudioFeatPath)
    inputTeamAudioFeat = teamAudioFeat.loc[:, teamAudioFeat.columns != 'sentiment']
    print ("TEAM audio features shape: " + str(inputTeamAudioFeat.shape))

    #creates the dataframe object of Text Features
    teamTextFeat = pd.read_csv(teamTextFeatPath)
    inputTeamTextFeat = teamTextFeat.loc[:, teamTextFeat.columns != 'sentiment']
    print ("TEAM text features shape: " + str(inputTeamTextFeat.shape))

    #creates the dataframe object of Audio and Text Features    
    teamAudioTextFeat = pd.read_csv(teamAudioTextFeatPath)
    inputTeamAudioTextFeat = teamAudioTextFeat.loc[:, teamAudioTextFeat.columns != 'sentiment']
    print ("TEAM combined audio and text features shape: " + str(inputTeamAudioTextFeat.shape))
	
    outputTeam = teamAudioFeat["sentiment"][:].values 
    return inputTeamAudioFeat, inputTeamTextFeat, inputTeamAudioTextFeat, outputTeam   
	
def getResults (audioFeat, textFeat, audioTextFeat, output, dataset):
     
    print(dataset)
    #splits the data of joined audioAndTextFeatures, AudioFeatures, and Text Features into training and testing data
    train_X_Aud, test_X_Aud, train_Y_Aud, test_Y_Aud = splitDataAfterPCA(audioFeat, output)
    print ("Reduce Audio feat Team:" + str(len(train_X_Aud[0])))
    train_X_text, test_X_text, train_Y_text, test_Y_text = splitDataAfterPCA(textFeat, output)
    print ("Reduce Text feat Team:" + str(len(train_X_text[0])))
    train_X_Audtext, test_X_Audtext, train_Y_Audtext, test_Y_Audtext = splitDataAfterPCA(audioTextFeat, output)
    print ("Reduce Audio & Text feat Team:" + str(len(train_X_Audtext[0])))

    print ("Results using " + str(dataset) + " audio features:")
    getSupervisedModelResults(train_X_Aud, test_X_Aud, train_Y_Aud, test_Y_Aud, "audio", dataset)
    print ("Results using " + str(dataset) + " text features:")
    getSupervisedModelResults(train_X_text, test_X_text, train_Y_text, test_Y_text, "text", dataset)
    print ("Results using " + str(dataset) + " audio and text features:")
    getSupervisedModelResults(train_X_Audtext, test_X_Audtext, train_Y_Audtext, test_Y_Audtext, "audio_text", dataset)


