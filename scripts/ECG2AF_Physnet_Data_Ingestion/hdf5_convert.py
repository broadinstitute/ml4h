import csv
import os
import numpy as np
import h5py
import pandas as pd
import argparse


#A class so it's easier to quickly call up attributes about a patient. 
class patient:
    def __init__(self,id):
        #param id: The data which will be used to identify this patient and be used as key in a dictionary of patients. 
        self.id = id
        #A dictionary of what attribute names and values.
        self.attributeDictionary = {}
    def __str__(self):
        return self.id
    #Adds an attribute with the specified name and value to the dictionary of attributes.
    def addAttribute(self,att,value):
        #param att: The name of the attribute being added.
        #param value: The value of the added attribute for this patient. 
        self.attributeDictionary.update({att:value})
    #Takes a string representing the name of an attribute (att) and returns the value of that attribute. 
    def retrieveAttribute(self,att):
        return self.attributeDictionary.get(att)
    #Returns a list of the names of every attribute this patient object has. 
    def getAttributeNames(self):
        return self.attributeDictionary.keys()


#Creates dictionary object containing all patients listed in the diagnostics file and the their information. 
#The first column in the diagnostics file will be used as keys. This is expected to be the filename or some other ID.
def makePatientDictionary(data,headers,split=True):
    #param data: A spreadsheet file containing patient information. Rows should start with patient information. 
    #param headers: A list containing data labels for the rows in the diagnostic spreadsheet.
    #param split: If data containing spaces should be stored as a list.
    headerRow = headers
    allPatients = {}
    for row in data:
        p = patient(row[0])
        #Adding all attributes to patient represented by this row. 
        for i in range(0,len(row)):
            value = row[i]
            if(split):
                if(isinstance(value,str)):
                    if(len(value.split()) > 1):
                        #splits the data into a list if it is suppose to represent a space seperated list. 
                        value = value.split()
            p.addAttribute(headerRow[i],value)
        allPatients.update({p.id:p})
    return allPatients

#A function for converting the datatype of data within a CSV.
def convertData(data,datatype):
    #param data: All rows and columns of the data to be converted.
    #param datatype: The datatype to convert the data to.  
    index = 0
    while(index < len(data)):
        row = data[index]
        if(datatype == int):
            #ints get their own function to ensure they round correctly. 
            data[index] = [convertToInt(i) for i in row]
        else:
            data[index] = [datatype(i) for i in row]
        index = index + 1
    return data
#Converts a float to an integer by rounding to the nearest whole number instead of rounding down. 
def convertToInt(f):
    #param f: The float to be converted into an integer.
    remainder = f - float(int(f)) 
    if (abs(remainder) >= 0.5):
        if(f >= 0):
            return int(f) + 1
        else:
            return int(f) - 1
    else:
        return int(f)
    
#A function for determining if the patient is male. Will default to female if they are not. But give a warning if they are neither male or female.
def calculateIsMale(patientSex):
    #param patientSex: A string, either "male" or "female" (case insensitive), representing the sex of the patient.
    if(patientSex.upper() == "MALE"):
        return 1
    elif (patientSex.upper() == "FEMALE"):
        return 0
    else:
        print("Warning: Patient with sex " + patientSex + " was marked as neither male or female.")
        #Defaults to female value if sex not recognized. 
        return 0

#Converts a directory of CSV files into hd5 format. Additional information is added from a diagnostics file is avaliable. 
def convertCVStoHDF5(patientDictionary,sourceDirectoryName,datatype = float,outputDirectoryName = "ECGdatahdf5",hasHeaders = True,groupLabels = None,groupNamePrefix= "/ukb_ecg_rest/",age_label = "PatientAge"):        
    #Param: Datatype: The datatype stored in the files being converted to hdf5
    #Param: sourceDirectoryName: The name of the directory containing the CSV files to convert into hdf5 files files. This directory should contain CSV files that all have the same number of columns, and are a homogeneous shape.
    #Param: outputDirectoryName: The name of the directory where the created hdf5 files will be stored.
    #Param: hasHeaders: If the first row of the data files are data labels that should be removed before converting them (And possibly used to label the hdf5 groups).
    #Param: groupLabels: A list containing what the groups in the converted hdf5 file should be called. Will overwrite the headers of the data if provided. 
    #Param: groupNamePrefix: A string that goes in front of the name of each group in the hdf5 file in addition to what is provided in groupLabels. EX: "Strip_" if the data is an ECG. 
    for file in os.listdir(sourceDirectoryName):
        patient = os.path.splitext(os.path.basename(file))[0]
        with open(sourceDirectoryName+"/"+file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            data = list(reader)
            #removing the headers so it can be converted right if they're present. 
            if(hasHeaders):
                if(groupLabels == None):
                    #Will use the headers to name the groups if none are given. (This is so they can still be overwritten)
                    groupLabels = data.pop(0)
                else:
                    data.pop(0)
            else:
                if(groupLabels == None):
                    print("Error: Data does not have headers to use as labels and none were given. Using 12-lead ECG lead names instead.")
                    groupLabels = ['Strip_I', 'Strip_II', 'Strip_III', 'Strip_aVR', 'Strip_aVL', 'Strip_aVF', 'Strip_V1', 'Strip_V2', 'Strip_V3', 'Strip_V4', 'Strip_V5', 'Strip_V6']
            if(len(groupLabels) != len(data[0])):
                #If the number of labels is not in sinc with the data (EX: The there are 9 headers/labels and 12 columns) then it will not convert that file.
                #This is currently so that the program does not need to delete labels/groups with no data and does not apply misleading labels to groups.
                #The exception is if no labels are provided period. In which case it was deemed fit to simply label them numerically.
                print(patient + "Encountered an error: Number of labels does not match number of columns in data.")
                print("Number of labels provided: " + str(len(groupLabels))+ ". Number of columns in the data "+ str(len(data[0])) + ".")
                print(patient + "Cannot be converted without proper number of group labels. Please provide labels and try again.")
            else:
                try:
                    #This is currently hardcoded to assume the date's here. Need to figure out if that's true.
                    dateValue = int(patient[5:13])
                    #seems you need to convert to a float first, can't go from string to float.
                    npdata = np.array(data, dtype = float)
                    if(datatype != float):
                        try:
                            data = convertData(npdata,datatype)
                        except Exception as error:
                            print("The first datapoint of " + patient + " did not match the datatype.")
                            print("The following error was encounted while trying to convert it:" + str(error))
                            print("Cannot convert this file. Skipping it.")
                    os.makedirs(outputDirectoryName,  exist_ok = True)
                    root = h5py.File(outputDirectoryName+ "/"+ patient+".hd5", "w")
                    groupNumber = 0
                    for group in groupLabels:
                        #instance_0 appears to be the standard name for a dataset in this case. 
                        root.create_dataset(groupNamePrefix + group + "/"+"instance_0", data=npdata[:,groupNumber])
                        #attaching the date to each strip. 
                        root[groupNamePrefix + group].attrs.create("date",dateValue,dtype=datatype)
                        groupNumber = groupNumber + 1

                    #hd5 files need to have a new datatype declared for strings so it knows how to encode it.
                    stringDatatype = h5py.string_dtype()
                    #Adding in the diagnostic data. 
                    if(patientDictionary != None):
                        patientAttributeNames = patientDictionary.get(patient).getAttributeNames()
                        for diagnostic in patientAttributeNames:
                            diagnosticValue = patientDictionary.get(patient).retrieveAttribute(diagnostic)
                            #PatientAge was listed as a catagorical value in the original run. This is to mirror that fact. 
                            if(diagnostic == age_label):
                                root.create_dataset("categorical/"+diagnostic,data=diagnosticValue)
                            elif(isinstance(diagnosticValue,(str,list))):
                                #String values are put into the catagorical subgroup.
                                diagnosticValue = np.array(diagnosticValue, dtype=stringDatatype)
                                root.create_dataset("categorical/"+diagnostic,data=diagnosticValue)
                            else: 
                                #Numerical values are put in the same group as the leads.
                                diagnosticValue = np.array(diagnosticValue,dtype= type(diagnosticValue))
                                root.create_dataset(groupNamePrefix+diagnostic+"/instance_0",data=diagnosticValue)
                                #Date was originally stored as a the value of the "datatype' variable.
                                root[groupNamePrefix + diagnostic].attrs.create("date",dateValue)
                except Exception as error:
                    print(patient + " caused an error: " + str(error))
                    print("Cannot convert this file. Skipping it.")
def main(sourceDirectoryName,diagnosticFile = None,datatype = float,outputDirectoryName = None,hasHeaders = True,groupLabels = ['strip_I', 'strip_II', 'strip_III', 'strip_aVR', 'strip_aVL', 'strip_aVF', 'strip_V1', 'strip_V2', 'strip_V3', 'strip_V4', 'strip_V5', 'strip_V6'],groupNamePrefix= "ukb_ecg_rest/",age_label = "PatientAge"):
    #param: sourceDirectoryName: The path to the directory containing the CSV files to converted to HDF5 format. 
    #param: diagnosticFile: The path to an optional file containing additonal information about every file in the source directory.
    #The first row should act as labels for the data and first column should be patient's name/ID.
    #param: datatype: The format the CSV data in will be stored in the HDF5 files. Defaults to float.
    #param: outputDirectoryName: The directory the converted files will be stored in. Will generate a name using the source directory if none is provided.
    #param: hasHeaders: If the top row the CSV files are headers/labels for the data. 
    #param: groupLabels: A list containing the labels for the columns of the CSV files that will form the groups within the HDF5 files. 
    #If groupLabels is left empty, group labels will be generated from the file headers, or they will use the default value of the 12-lead ECG strip names if hasHeaders = False.
    #Will cause an error if the length of the list does not match the number of rows in the CSV files. 
    #param: groupNamePrefix: A string that is applied to the begining of every non-catagorical group. Useful if they are all to go in a certain subgroup or dataset. Defaults to "ukb_ecg_rest/"
    #param: 
    if(sourceDirectoryName == None):
        print("Error: Source directory not provided. Aborting.")
        return 0
    if(diagnosticFile != None):
        if(os.path.exists(os.path.dirname((diagnosticFile)))):
            print("Creating patient Dictionary from "+ diagnosticFile)
            raw_diagnostic_data = pd.read_excel(diagnosticFile)
            headers = list(raw_diagnostic_data.columns.values)
            raw_diagnostic_data = raw_diagnostic_data.to_numpy()
            patientDictionary = makePatientDictionary(raw_diagnostic_data,headers)
        else: 
            print("Error, filepath " + str(diagnosticFile) + " does not exist. Aborting.")
            return 0
    else:
        print("No diagnostics files detected. Converting only raw ECG data into HDF5.")
        patientDictionary = None
    if(outputDirectoryName == None):
        outputDirectoryName = sourceDirectoryName+" HDF5"
    print("Converting files from source directory into HDF5 Files.")
    print("Converted Files will be stored at " + outputDirectoryName)
    convertCVStoHDF5(patientDictionary,datatype= float,sourceDirectoryName=sourceDirectoryName,outputDirectoryName = outputDirectoryName,groupLabels=groupLabels,groupNamePrefix=groupNamePrefix,age_label=age_label)
    print("Done!")
    return 1
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts a CSV file, and accompanying diagnostic spreadsheet file into HDF5 Format.")
    parser.add_argument("--sourceDirectoryName",type = str,help="The path to the directory containing the CSV files to converted to HDF5 format.")
    parser.add_argument("--diagnosticFile",type = str,help="The path to an optional file containing additonal information about every file in the source directory. \nThe first row should act as labels for the data and first column should be patient's name/ID.")
    parser.add_argument("--datatype",type = type,help="The datatype the information in the CSV files should be stored in after conversion. Defaults to Float")
    parser.add_argument("--outputDirectoryName",type=str,help="The directory the converted files will be stored in. Will generate a name using the source directory if none is provided.")
    parser.add_argument("--hasHeaders",type=bool,help="If the top row the CSV files are headers/labels for the data.")
    parser.add_argument("--groupLabels",type = str,nargs='+',help="A list containing the labels for the columns of the CSV files. Will use the file headers if left blank and hasHeaders = true. Otherwise will attempt to use the default 12-lead ECG names. Fails to convert a file if the length of the list is !+ the number of columns in the CSV.")
    parser.add_argument("--groupNamePrefix",type = str,help="A string that is applied to the begining of every non-catagorical group. Useful if they are all to go in a certain subgroup or dataset. Defaults to ukb_ecg_rest/")
    parser.add_argument("--ageLabel",type = str,help="The label corresponding to patient's age in the diagnostic file. Defaults to 'PatientAge'.")
    args = parser.parse_args()
    main(sourceDirectoryName=args.sourceDirectoryName,diagnosticFile=args.diagnosticFile,datatype=args.datatype,outputDirectoryName=args.outputDirectoryName,hasHeaders=args.hasHeaders,groupLabels=args.groupLabels,groupNamePrefix=args.groupNamePrefix,age_label=args.ageLabel)
