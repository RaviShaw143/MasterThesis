import os
import supervisedModels
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, default = "YouTube")
args = parser.parse_args()
print(args)

#retrieve the current working directory where feature files are located
featureFilesDirectory = os.path.join(os.getcwd(), "MasterThesis","datasets")
rootDirectoryDataset = os.path.join(featureFilesDirectory,args.dataset)
print(rootDirectoryDataset)

if args.dataset == "YouTube": 
    # creates the dataframe object of youtube datasets
    inputYouTubeAudioFeat, inputYouTubeTextFeat, inputYouTubeAudioTextFeat, outputYouTube = supervisedModels.getYouTubeData(rootDirectoryDataset)
    train_X_Audtext, test_X_Audtext, train_Y_Audtext, test_Y_Audtext = supervisedModels.splitDataAfterPCA(inputYouTubeAudioTextFeat, outputYouTube)
    supervisedModels.getResults(inputYouTubeAudioFeat, inputYouTubeTextFeat, inputYouTubeAudioTextFeat, outputYouTube, "YouTube")

if args.dataset == "TEAM":
    # creates the dataframe object of TEAM datasets 
    inputTeamAudioFeat, inputTeamTextFeat, inputTeamAudioTextFeat, outputTeam = supervisedModels.getTEAMData(rootDirectoryDataset)
    supervisedModels.getResults(inputTeamAudioFeat, inputTeamTextFeat, inputTeamAudioTextFeat, outputTeam, "TEAM")
