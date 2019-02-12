# -*- coding: utf-8 -*-
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

       
#splits the data into training and testing data and then reduce the features using PCA      
def splitDataAfterPCA(input, output):
    train_data, test_data, train_lbl, test_lbl = train_test_split(input, output, test_size=0.20, random_state=0)
    
    scaler = StandardScaler()
    # Fit on training set only.
    scaler.fit(train_data)

    # Apply transform to both the training set and the test set.    
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    
    pca = PCA(.95) 
    # Fit on training set only.
    pca.fit(train_data)
    
    # Apply transform to both the training set and the test set.
    train_data = pca.transform(train_data)
    test_data = pca.transform(test_data)
    return train_data, test_data, train_lbl, test_lbl 
    
#trains a given model with training data provided by finding the best hyperparameters for the model and  then returns the precision and recall score of the model
def findBestEstimator(model, tuned_parameters, X_train, X_test, y_train, y_test):
    scores = ['precision', 'recall']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(model, tuned_parameters, cv=10,
                        scoring='%s_micro' % score)
        clf.fit(X_train, y_train)
    
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
       # print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        
        print()
    
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
    

#Gives SVC models (SVC, LinearSVC and NuSVC) results 
def getSVCModelResults(train_X_Com, test_X_Com, train_Y_Com, test_Y_Com):
    
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.01, 0.001, 0.0001],
                        'C': [0.001,0.01,0.1,1, 10, 100, 1000]},
                        {'kernel': ['linear'],'gamma': [0.01, 0.001, 0.0001], 'C': [0.001,0.01,0.1,1, 10, 100, 1000]}]
    #findBestEstimator(SVC(),tuned_parameters,train_data_Com, test_data_Com, train_lbl_Com, test_lbl_Com)
    
    tuned_parameters_lin = [{'dual': [False],'C': [1, 10, 100, 1000],'multi_class' :['ovr']},
                        {'dual': [False],'C': [1, 10, 100, 1000],'multi_class' :['crammer_singer']},
                        {'dual': [True],'C': [1, 10, 100, 1000],'multi_class' :['crammer_singer']}]
    
    tuned_parameters_nuSVC = [{'nu':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8], 'gamma': [0.01, 0.001, 0.0001]}]
    
    print("Summary of SVC model results: ")
    findBestEstimator(SVC(),tuned_parameters,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com)
    print("Summary of LinearSVC model results: ")
    findBestEstimator(LinearSVC(),tuned_parameters_lin,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com)
    print("Summary of NuSVC model results: ")    
    findBestEstimator(NuSVC(),tuned_parameters_nuSVC,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com)
 
#gives RandomForestClassifier models (RandomForestClassifier, ExtraTreesClassifier, DecisionTreeClassifier) results
def getRandomForestClassifierResults(train_X_Com, test_X_Com, train_Y_Com, test_Y_Com):    
    tune_params_RF = [{'n_estimators': [10,20,30,40,50,60],'max_features': [None],'max_depth':[None],'min_samples_split':[2], 'random_state':[0],'n_jobs':[1]}]
    tune_param_DClf = [{'max_depth':[3],'min_samples_leaf':[1]}]
	
    print("Summary of RandomForestClassifier model results: ")    
    findBestEstimator(RandomForestClassifier(),tune_params_RF,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com)
    print("Summary of ExtraTreesClassifier model results: ")    
    findBestEstimator(ExtraTreesClassifier(),tune_params_RF,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com)
    print("Summary of DecisionTreeClassifier model results: ")        
    findBestEstimator(DecisionTreeClassifier(),tune_param_DClf,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com)

#gives Logistic Regression model results
def getLogisticRegressionResults(train_X_Com, test_X_Com, train_Y_Com, test_Y_Com):        
    tune_params_LR1 = [{'penalty':['l1'], 'solver':['liblinear','saga'],'max_iter':[2000],'multi_class':['ovr']}]
    tune_params_LR2 = [{'penalty':['l2'], 'solver':['newton-cg','lbfgs','sag'], 'max_iter':[2000],'multi_class':['multinomial','ovr']}]
    print("Summary of LogisticRegression using l1 model results: ")            
    findBestEstimator(LogisticRegression(),tune_params_LR1,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com )
    print("Summary of LogisticRegression using l2 model results: ")            
    findBestEstimator(LogisticRegression(),tune_params_LR2,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com )

#gets the results of all models used at once 
def getSupervisedModelResults(train_X_Com, test_X_Com, train_Y_Com, test_Y_Com):
    print("SVM models results:")
    getSVCModelResults(train_X_Com, test_X_Com, train_Y_Com, test_Y_Com)
    print("RandomForest classifier models results:")
    getRandomForestClassifierResults(train_X_Com, test_X_Com, train_Y_Com, test_Y_Com)
    print("LogisticRegression models results:")
    getLogisticRegressionResults(train_X_Com, test_X_Com, train_Y_Com, test_Y_Com)


def getYouTubeData(rootDirectoryYouTubeDatasets):
    	
    # YouTube dataset features file path
    youTubeAudioFeatPath = os.path.join(rootDirectoryYouTubeDatasets,"YouTubeAudioFeat.csv")
    youTubeTextFeatPath = os.path.join(rootDirectoryYouTubeDatasets,"YouTubeTextFeat.csv")
    youTubeAudioTextFeatPath = os.path.join(rootDirectoryYouTubeDatasets,"YouTubeAudioTextFeat.csv")

    #creates the dataframe object of Audio Features
    youTubeAudioFeat = pd.read_csv(youTubeAudioFeatPath)
    inputYouTubeAudioFeat = youTubeAudioFeat.loc[:, youTubeAudioFeat.columns != 'sentiment']
    print ("YouTube audio Features shape: " + str(inputYouTubeAudioFeat.shape))
    
    #creates the dataframe object of Text Features
    youTubeTextFeat = pd.read_csv(youTubeTextFeatPath)
    inputYouTubeTextFeat = youTubeTextFeat.loc[:, youTubeTextFeat.columns != 'sentiment']
    print ("YouTube text Features shape: " + str(inputYouTubeTextFeat.shape))

    #creates the dataframe object of Audio and Text Features    
    youTubeAudioTextFeat = pd.read_csv(youTubeAudioTextFeatPath)
    inputYouTubeAudioTextFeat = youTubeAudioTextFeat.loc[:, youTubeAudioTextFeat.columns != 'sentiment']
    print ("YouTube combined audio and text features shape: " + str(inputYouTubeAudioTextFeat.shape))
	
    outputYouTube = youTubeAudioFeat["sentiment"][:].values 
    return inputYouTubeAudioFeat, inputYouTubeTextFeat, inputYouTubeAudioTextFeat, outputYouTube

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
    	
    #splits the data of joined audioAndTextFeatures, AudioFeatures, and Text Features into training and testing data
    train_X_Aud, test_X_Aud, train_Y_Aud, test_Y_Aud = splitDataAfterPCA(audioFeat, output)
    train_X_text, test_X_text, train_Y_text, test_Y_text = splitDataAfterPCA(textFeat, output)
    train_X_Audtext, test_X_Audtext, train_Y_Audtext, test_Y_Audtext = splitDataAfterPCA(audioTextFeat, output)

    print ("Results using " + str(dataset) + " audio features:")
    getSupervisedModelResults(train_X_Aud, test_X_Aud, train_Y_Aud, test_Y_Aud)
    print ("Results using " + str(dataset) + " text features:")
    getSupervisedModelResults(train_X_text, test_X_text, train_Y_text, test_Y_text)
    print ("Results using " + str(dataset) + " audio and text features:")
    getSupervisedModelResults(train_X_Audtext, test_X_Audtext, train_Y_Audtext, test_Y_Audtext)

	
#retrieve the current working directory where feature files are located
featureFilesDirectory = os.path.join(os.getcwd(),"datasets")
rootDirectoryYouTubeDatasets = os.path.join(featureFilesDirectory,"YouTube")
print(rootDirectoryYouTubeDatasets)
rootDirectoryTEAMDatasets = os.path.join(featureFilesDirectory,"TEAM")

# creates the dataframe object of youtube datasets
inputYouTubeAudioFeat, inputYouTubeTextFeat, inputYouTubeAudioTextFeat, outputYouTube = getYouTubeData(rootDirectoryYouTubeDatasets)
getResults(inputYouTubeAudioFeat, inputYouTubeTextFeat, inputYouTubeAudioTextFeat, outputYouTube, "YouTube")

# creates the dataframe object of TEAM datasets 
#inputTeamAudioFeat, inputTeamTextFeat, inputTeamAudioTextFeat, outputTeam = getTEAMData(rootDirectoryTEAMDatasets)
#getResults(inputTeamAudioFeat, inputTeamTextFeat, inputTeamAudioTextFeat, outputTeam, "TEAM")

