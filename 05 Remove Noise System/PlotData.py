import json
import matplotlib.pyplot as pyplot
import numpy as np

def getFileName(algoType, fileDate, input):
    fileName = "data/"
    fileName += ["Idle", "Sorting", "Graphs", "HeuristicSearch", "Searching"][algoType + 1] + "/"
    fileName += "turing2-"
    fileName += ["idle", "sorting", "graphs", "heuristic-search", "searching"][algoType + 1] + "-"
    if(algoType != -1):
        fileName += "i" + str(input) + "-"
    fileName += fileDate
    return fileName

def generateGraph():
    ### Start of User-Definable Parameters
    algoType = -1
    fileDate = "16-mar"
    input = 3
    ### End of User-Definable Parameters
    
    fileName = getFileName(algoType, fileDate, input)
    print(fileName)

    noise_data_file = "false"
    try:
        with open("../03 Record Power System/" + fileName + "-rec.json") as f:
            noise_data_file = json.load(f)
    except:
        print("ERROR: noise_data_file not found")
        
    removed_noise_data_file = "false"
    try:
        with open("./" + fileName + "-noise-removed.json") as f:
            removed_noise_data_file = json.load(f)
    except:
        print("ERROR: removed_noise_data_file not found")
        
    if(noise_data_file != "false" and removed_noise_data_file != "false"):              
        pyplot.plot(noise_data_file["power"], linewidth=1, color="#555555")
        pyplot.plot(removed_noise_data_file["power"], linewidth=3, color="#0044ff")
        pyplot.ylabel("Wattage (mW)")
        pyplot.xlabel("Time (s)")
        
        length = len(removed_noise_data_file["power"])
        steps = 60 * 30
        xSteps = np.arange(0, (length) + 1, steps * 2)
        xLabels = np.arange(0, (length / 2) + 1, (steps * 2) / 2) 
        pyplot.xticks(xSteps, np.asarray(xLabels.astype(int)))
        pyplot.xlim([0, length])
        
        pyplot.show()
        


generateGraph()