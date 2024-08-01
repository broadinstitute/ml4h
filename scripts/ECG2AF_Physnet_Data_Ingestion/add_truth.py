import csv
import os
import numpy as np
import pandas as pd
import argparse
#This is a script to add missing truth values to the results of running inference using recipes.py.


#A class so it's easier to quickly call up attributes about a patient instead of referencing the indexes. 
class patient:
    def __init__(self,id):
        self.id = id
        #A dictionary of what attribute names and values.
        #All entries should be lowercase to avoid case-sensitivity. 
        self.attributeDictionary = {}
    def __str__(self):
        return self.id
    def addAttribute(self,att,value):
        #Adds an attribute with the specified name and value to the dictionary of attributes.
        self.attributeDictionary.update({att:value})
    #Takes a string representing the name of an attribute (att) and returns the value of that attribute. 
    def retrieveAttribute(self,att):
        return self.attributeDictionary.get(att)
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

def isFemale(patientDictionary,patient):
    if(patientDictionary.get(patient).retrieveAttribute("Gender") == "FEMALE"):
        return 1.0
        
    else: 
        return 0.0
def hasAFIB(patientDictionary,patient):
    if("AFIB" in patientDictionary.get(patient).retrieveAttribute("Rhythm")):
        return 1.0
    else: 
        return 0.0

def addTruthValues(patientDictionary, sourcefile,outputFile,removeNaN = True):
    with open(sourcefile) as tsvFile:
        reader = csv.reader(tsvFile, delimiter="\t", quotechar='"')
        rownumber = 0
        rows = []
        for row in reader:
            outputRow = row.copy()
            #ignore the header.
            if(rownumber != 0):
                patient = row[0]
                outputRow[5] = isFemale(patientDictionary,patient)
                outputRow[7] = 1.0 - isFemale(patientDictionary,patient)
                outputRow[9] = patientDictionary.get(patient).retrieveAttribute("PatientAge")
                outputRow[11] = 1 - hasAFIB(patientDictionary,patient)
                outputRow[13] = hasAFIB(patientDictionary,patient)
                if(outputRow[10] == "nan" and removeNaN == True):
                    print(str(patient) + " has 'NaN' values and has been removed from the results. ")
                else:
                    rows.append(outputRow)
            else:
                headers = outputRow
            rownumber = rownumber+1
    with open(outputFile, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)

def main(inference_file = None,diagnostic_file = None,output_file = None,removeNaN = True):
    if(diagnostic_file != None):
        if(os.path.exists(os.path.dirname((diagnostic_file)))):
            print("Creating patient Dictionary from "+ diagnostic_file)
            raw_diagnostic_data = pd.read_excel(diagnostic_file)
            headers = list(raw_diagnostic_data.columns.values)
            raw_diagnostic_data = raw_diagnostic_data.to_numpy()
            patientDictionary = makePatientDictionary(raw_diagnostic_data,headers)
        else: 
            print("Error, filepath " + str(diagnostic_file) + " does not exist. Aborting.")
            return 0
    else:
        print("Diagnostic file not provided. Cannot add truth values with no diagnostic file. Aborting.")
        return 0
    if(inference_file == None):
        print("Error: Inference file not provided. Aborting.")
        return 0
    if(output_file == None):
        output_file = inference_file + "_added_truth_values.csv"
    addTruthValues(patientDictionary,inference_file,output_file,removeNaN=removeNaN)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts a CSV file, and accompanying diagnostic spreadsheet file into HDF5 Format.")
    parser.add_argument("--inference_file", type=str,help="The path to the tsv file recieved by running inference on your data using recipes.py.")
    parser.add_argument("--diagnostic_file",type = str,help = "The path to the diagnostic file containing the truth values for your data.")
    parser.add_argument("--output_file_name",type = str,help = "The name you would like the file with the truth values added in to have.")
    parser.add_argument("--remove_nan",type=bool,help = "If the script should remove any entries in which the predicted confidence in the presence of atrial fibrillation is 'nan'. Defaults to true.")
    args = parser.parse_args()
    main(inference_file=args.inference_file,diagnostic_file=args.diagnostic_file,output_file=args.output_file_name,removeNaN=args.remove_nan)
