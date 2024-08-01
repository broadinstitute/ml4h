import matplotlib.pyplot as plt 
import csv
import os
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from ml4h import plots
import scipy
import argparse

""""creates ROC and PR plots of a model's predictions of the patient's sex and the presence/absence of atrial fibrillation. 
In addition a scatterplot comparing the models predicted ages and actual ages."""""


#Prepares lists of prediction and truth values in the specified index and graphs them using the ml4h plot_rocs method. 
def ROCgraph(dataFile, predictionIndex,truthIndex,outputFilepath = "./ROC Graph/",graphtitle = "",label = "",modelName = "ECG2AF"):
    with open(dataFile, newline='') as csvfile:
        if(dataFile[-3:] == "tsv"):
            pddata = pd.read_csv(dataFile,delimiter='\t')
            reader = csv.reader(csvfile, delimiter='\t')
        else:
            pddata = pd.read_csv(dataFile)
            reader = csv.reader(csvfile, delimiter=',')
        readerData = list(reader)
        predictionValues = []
        truthValues = []
        #creating a list of all the prediction and values. 
        #Documentation says  "The arrays have shape (num_samples, num_classes)". Hence why the values are in a nested list.
        for row in readerData[1:]:
            if(row[predictionIndex] != "nan" and row[truthIndex] != "nan"):
                predictionValues.append([float(row[predictionIndex])])
                truthValues.append([row[truthIndex]])
        predictionValues = np.array(predictionValues)
        predictions = {label:predictionValues}
        #One hot encoding the truth valleys using pandas. 
        #Some of the rows include "nan" values. Those need to be removed. 
        #The method to add truth values may have already done this. 
        removedNans = pddata[pddata[readerData[0][predictionIndex]] != "nan"]
        #Isolating the truth values. 
        one_hot_truths = removedNans[[readerData[0][truthIndex]]]
        #One Hot Encoding.
        one_hot_truths = pd.get_dummies(one_hot_truths)
        truths = one_hot_truths.to_numpy()
        labels = {modelName:0}
        title = graphtitle
        #TODO: Fix the outputfilepath.
        plots.plot_rocs(predictions,truths,labels,title,outputFilepath)
#Creates a dictionary of predictions with a model name as a key and 2 lists of values, where one is a list of predictions returned by inference, and the other is a list equal to 1-the first list. 
def createOppositePredictionsDict(data,predictionIndex1,predictionIndex2 = None,modelName = "ECG2AF"):
    #param data: The filename of a tsv or similar file generated from running inference. 
    #param predictionIndex1: The index where the first column of predictions is located in the data.
    #param predictionIndex2: The index where the second column of predictions is located. These are assumed to be the 1 - the values of the data in predictionIndex1 if not provided. 
    predictions = []
    #For now we'll set this to the first index + 2. This requires the enter the first variable to be entered first.
    with open(data, newline='') as csvfile:
        if(data[-3:] == "tsv"):
            reader = csv.reader(csvfile, delimiter='\t')
        else:
            reader = csv.reader(csvfile, delimiter=',')
        readerData = list(reader)
        for row in readerData[1:]:
            #Ignores values where prediction ended up being nan. 
            #Truth should never be nan since they're either set in manually or returns 0. 
            #TODO: This is graphing it as 100% precision with everyone going in male. Fix that. 
            if(row[predictionIndex1] != "nan"):
                pred1 = float(row[predictionIndex1])
                if(predictionIndex2 == None):
                    pred2 = 1.0 - float(pred1)
                else:
                    pred2 = float(row[predictionIndex2])
                predictions.append([pred1,pred2])
        predictions = np.array(predictions)  
    return {modelName:predictions}
def createOppositeTruths(data,truthIndex,predictionIndex=None,modelName = "ECG2AF"):
    truths = []
    with open(data, newline='') as csvfile:
        if(data[-3:] == "tsv"):
            reader = csv.reader(csvfile, delimiter='\t')
        else:
            reader = csv.reader(csvfile, delimiter=',')
        readerData = list(reader)
        for row in readerData[1:]:
            #Doing it this way to avoid floating point errors.
            #If either the predictions or the truth values are "nan" it will just ignore them. 
            if(row[predictionIndex] != "nan" and row[truthIndex] != "nan"):
                if(float(row[truthIndex]) >= 1):
                    truth1 = 1
                    truth2 = 0
                else:
                    truth1 = 0
                    truth2 = 1
                truths.append([truth1,truth2])
        truths = np.array(truths)  
    return truths
def createPredictionsDict(data,PredictionIndex,modelName = "ECG2AF"):
    predictions = []
    with open(data, newline='') as csvfile:
        if(data[-3:] == "tsv"):
            reader = csv.reader(csvfile, delimiter='\t')
        else:
            reader = csv.reader(csvfile, delimiter=',')
        readerData = list(reader)
        for row in readerData[1:]:
            if(row[PredictionIndex] != "nan"):
                predictions.append(row[PredictionIndex])
        predictions = np.array(predictions)  
    return {modelName:predictions}
def createTruths(data,PredictionIndex,TruthIndex,modelName = "ECG2AF"):
    truths = []
    with open(data, newline='') as csvfile:
        if(data[-3:] == "tsv"):
            reader = csv.reader(csvfile, delimiter='\t')
        else:
            reader = csv.reader(csvfile, delimiter=',')
        readerData = list(reader)
        for row in readerData[1:]:
            #Doing it this way to avoid floating point errors.
            if(row[PredictionIndex] != "nan" and row[TruthIndex] != "nan"):
                truths.append(row[TruthIndex])
        truths = np.array(truths)  
    return truths
def PRgraph(dataFile, predictionIndex,truthIndex,labels=None,title="PR Graph",outputFilepath = "./PR Graph/"):
    #param datafile: The file with results of inference as an array. 
    #param predictionIndex: The index in the array where the models prediction of a certain variable is located. 
    #param truthIndex: The index in the array where the true value of that varaible is located
    #param labels: A list of length 2 with the name of the variable found in the index of predictionIndex followed by its opposite. 
    #For example, if entering the predictions a subject is female in the truth index, you would pass [Female', 'Male']
    #param title: The title of the graph.  
    if(labels==None):
        labels = {dataFile[0][predictionIndex] : 0, dataFile[0][predictionIndex + 2]:1}
    else:
        labels = {labels[0] : 0,labels[1] : 1}
    predictions = createOppositePredictionsDict(dataFile,predictionIndex)
    truth = createOppositeTruths(dataFile,truthIndex,predictionIndex)
    plots.plot_precision_recalls(predictions, truth, labels, title, prefix=outputFilepath, dpi=300, width=7, height=7)
def graphAge(data,graphname,outputDirectory,hasHeaders = True,ageThreshhold = 0):
    plt.cla()
    with open(data, newline='') as csvfile:
        if(data[-3:] == "tsv"):
            reader = csv.reader(csvfile, delimiter='\t')
        else:
            reader = csv.reader(csvfile, delimiter=',')
        data = list(reader)
    predictedAges = []
    trueAges =[]
    if(hasHeaders):
        #this excludes the first row, which is headers.
        data = data[1:]
    for row in data:
        if(float(row[9]) >= ageThreshhold and float(row[9]) != "NA"):
            predictedAges.append(float(row[8]))
            trueAges.append(float(row[9]))
    correlation_coefficient,p_value = scipy.stats.pearsonr(trueAges,predictedAges)
    correlation_coefficient = round(correlation_coefficient,3)
    p_value = round(p_value,3)
    plt.title(graphname+" Patient Age Predictions CC = " + str(correlation_coefficient) +" p-value = "+str(p_value),fontsize = 8) 
    plt.ylabel("Model Predicted Age")
    plt.xlabel("Patient Actual Age")
    #Setting it so that both axes have the same scale.
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.scatter(trueAges,predictedAges)
    plt.savefig(outputDirectory+"Ages >= "+ str(ageThreshhold) +".png")
def graphAll(inference_file,modelName,outputFilepath="./figures/",age_threshhold=0):
    #The plot_rocs seems to always have it's graph titles start with "ROC". 
    ROCgraph(inference_file,12,13,outputFilepath,"For "+modelName+ " AFIB Presence Predictions",label="AFIB Prediction")
    ROCgraph(inference_file,10,11,outputFilepath,"For "+modelName+ " AFIB Absence Predictions",label="AFIB Absence")
    ROCgraph(inference_file,4,5,outputFilepath,"For "+modelName+ " Female Predictions",label="Female")
    ROCgraph(inference_file,6,7,outputFilepath,"For "+modelName+ " Male Predictions",label="Male")
    PRgraph(inference_file,4,5,title="Precision Recall Graph of "+modelName+ " Gender Identification",labels=['Female', 'Male'],outputFilepath=outputFilepath)
    PRgraph(inference_file,10,11,title = "Precision Recall Graph of " + modelName + " AFIB Presence Identification", labels = ['Absence of AFIB', "Presence of AFIB"],outputFilepath=outputFilepath)
    graphAge(inference_file,graphname="Scatterplot of " + modelName +" Predicted vs Actual Ages",outputDirectory=outputFilepath,ageThreshhold=age_threshhold)
def main(inference_file=None,prediction_index=None,truth_index=None,labels = None,title=None,model = "Model",quick_graph = False,output_file_path="./figure/",age_threshhold = 0):
    if(inference_file == None):
        print("No inference file provided. Aborting.")
        return 0
    if(os.path.exists(os.path.dirname((output_file_path))) == False):
        os.mkdir(output_file_path)
    print("Creating graphs of gender and atrial fibrillation predictions.")
    graphAll(inference_file,modelName=model,outputFilepath=output_file_path,age_threshhold=age_threshhold)

    

if __name__ == "__main__":
    #Commented out variables are in case graphing individual variables are to be implemented in the future. For now this script simply graphs both gender and atrial fibrillation. 
    parser = argparse.ArgumentParser(description="Creates ROC and PR plots of a models predictions of the patient's sex and the presence/absense of atrial fibrulation. In addition a scatterplot comparing a models predicted ages and actual ages.")
    parser.add_argument("--inference_file", type=str,help="The path to the tsv file recieved by running inference on your data using recipes.py.")
    #parser.add_argument("--prediction_index",type = int,help = "The index of the column where the predicted values of the variable being graphed are located. For graph two opposite probabilities (EX: Likelihood of patient being female or male) use the index of the one that comes first.")
    #parser.add_argument("--truth_index",type=int,help = "The index in the array where the true value of the varaible being graphed is located")
    #parser.add_argument("--labels",type = str,nargs='+',help="A list of length 2 with the name of the variable found in the index of predictionIndex followed by its opposite. For example, if entering the predictions a subject is female in the truth index, you would pass [Female', 'Male']")
    #parser.add_argument("--title",type = str,help= "The graph's title.")
    parser.add_argument("--model",type=str,help="The name of the model used for inference.")
    parser.add_argument("--output_file_path",type=str,help="Filepath where the completed graphs should go. Defaults to './figures/'.")
    #parser.add_argument("--quick_graph",type = bool,help="A mode to make PR and ROC graphs of gender and AFIB presence predictions on the provided file. ")
    parser.add_argument("--age_threshhold",type = int, help="A variable where every entry with an age below this number will not be graphed.")
    args = parser.parse_args()
    #main(inference_file=args.inference_file,prediction_index=args.prediction_index,truth_index=args.truth_index,labels=args.labels,title=args.title,model=args.model,output_file_path=args.output_file_path,quick_graph=args.quick_graph)
    main(inference_file=args.inference_file,model=args.model,output_file_path=args.output_file_path,age_threshhold = args.age_threshhold)
