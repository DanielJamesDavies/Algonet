import os
import math
import json
import random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tabulate import tabulate
from architectures.NN_Deep import NN_Deep
from architectures.NN_Shallow import NN_Shallow
from architectures.NN_Single import NN_Single



### Start of User-Definable Parameters
architecture = "deep" # "deep", "shallow", or "single"
activation_function = "relu" # "relu" or "tanh"
determineBestModelBy = "correlation" # "correlation", "loss", or "overfitting"

datasetsCount = 120
repeatCount = 5
epochs = 120
batch_size = 10 # 0 for 100% Batch Size
learning_rate = 0.001
### End of User-Definable Parameters


np.seterr(divide='ignore', invalid='ignore')
results = {"bestTestCorrelation": 0, "bestTestLoss": 0, "bestTestOverfitting": 0, "testCorrelationMax": 0, "testCorrelationAvg": 0, "testCorrelationVar": 0, "testLossMin": 0, "testLossAvg": 0, "testLossVar": 0, "overfittingMin": 0, "overfittingAvg": 0, "overfittingVar": 0, "testLoss": [], "testCorrelation": [], "overfitting": []}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def getDatasets():
	datasets = []

	with open("../07 Create Dataset/data/dataset.json") as f:
		originalDataset = json.load(f)["data"]

	for i in range(datasetsCount):
		newDataset = originalDataset.copy()
		while(newDataset in datasets):
			random.shuffle(newDataset)
		datasets.append(newDataset)
		
	for i in range(len(datasets)):
		newDataset = []
		for datapoint in datasets[i]:
			for j in range(repeatCount):
				newDataset.append(datapoint)
		datasets[i] = newDataset

	return datasets


class TrainingDataset(Dataset):
	def __init__(self, x, y, test_size):
		self.x = torch.from_numpy(x[test_size:].astype('float32')).to(device)
		self.y = torch.from_numpy(y[test_size:].astype('float32')).to(device)
		self.n_samples = x[test_size:].shape[0]
		
	def __getitem__(self, index):
		return self.x[index], self.y[index]
		
	def __len__(self):
		return self.n_samples


class BestModel:
	def __init__(self):
		self.correlation = 0
		self.loss = 0
		self.overfitting = 0

	def update(self, model, correlation, loss, overfitting):
		if(not hasattr(self, "model")):
			self.model = model
			self.correlation = correlation
			self.loss = loss
			self.overfitting = overfitting
			return

		if (abs(correlation) < 0.9 or overfitting > 0.5):
			return

		if(determineBestModelBy == "correlation" and abs(correlation) > self.correlation):
			self.model = model
			self.correlation = correlation
			self.loss = loss
			self.overfitting = overfitting
		elif(determineBestModelBy == "loss" and loss < self.loss):
			self.model = model
			self.correlation = correlation
			self.loss = loss
			self.overfitting = overfitting
		elif(determineBestModelBy == "overfitting" and overfitting < self.overfitting):
			self.model = model
			self.correlation = correlation
			self.loss = loss
			self.overfitting = overfitting


def getProgressBarString(progress, total, length):
	progressBarString = "["

	progress = round((progress / total) * length)

	for j in range(progress):
		progressBarString += "="

	progressBarString += ">"

	for j in range(length - progress):
		progressBarString += "_"

	progressBarString += "]"

	return progressBarString


def printEpochProgress(epoch, epochs, datasetIndex, datasetsCount, datasets_start_time, epoch_start_time, loss, bestModel):
	print("\033[A" * 5)
 
	print("Dataset		" + str(datasetIndex + 1).zfill(len(list(str(datasetsCount)))).zfill(3) + "/" + str(datasetsCount).zfill(3) + "     ", end = '')
	print(getProgressBarString(datasetIndex + 1, datasetsCount, 48) + "	", end = '')
	secondsSinceStart = (datetime.now()-datasets_start_time).total_seconds()
	datasetsComplete = (datasetIndex + 1) + ((epoch + 1) / epochs)
	secondsTillCompletion = (secondsSinceStart / datasetsComplete) * (datasetsCount - datasetsComplete) * math.log(datasetsComplete / datasetsCount * 100)
	print("ETA: " + str(round(secondsTillCompletion)) + "s ", end = '')
	print("")

 
	print("  Epoch		" + str(epoch + 1).zfill(len(list(str(epochs)))).zfill(3) + "/" + str(epochs).zfill(3) + "     ", end = '')
	print(getProgressBarString(epoch + 1, epochs, 48) + "	", end = '')
	print("ETA: " + str(round(((datetime.now()-epoch_start_time).total_seconds() / (epoch + 1)) * (epochs - epoch + 1))) + "s ", end = '') 
	print("	Loss: " + "{:,.4f}".format(loss).rstrip('0').rstrip('.'), end = '')
	print("")
 
	print("Best Model:	", end = '')
	print("Correlation: " + "{:.4f}".format(float(bestModel.correlation)) + "	", end = '')
	print("Loss: " + "{:.4f}".format(float(bestModel.loss)) + "	", end = '')
	print("Overfitting: " + "{:.4f}".format(float(bestModel.overfitting)), end = '')
	print("")
	print("")
 
 
def isNotNaN(value):
	return value == value

def runOnDataset(datasets, i, NN, activation_function, bestModel, datasets_start_time):
	global epochs
	global batch_size
	global learning_rate
	global hidden_size

	# Dataset
	dataset = datasets[i]
 
	test_size = round(len(dataset) / 4.5)

	x = np.array(dataset)[:, 1:]
	y = np.array(dataset)[:, [0]]
	x_train = torch.from_numpy(x[test_size:].astype('float32')).to(device)
	x_test = torch.from_numpy(x[:test_size].astype('float32')).to(device)
	y_train = torch.from_numpy(y[test_size:].astype('float32')).to(device)
	y_test = torch.from_numpy(y[:test_size].astype('float32')).to(device)
 
	if(batch_size == 0):
		batchSize = x[test_size:].shape[0]
	else:
		batchSize = batch_size
  
	training_dataset = TrainingDataset(x, y, test_size)
	training_dataloader = DataLoader(dataset=training_dataset, batch_size=batchSize, shuffle=True, num_workers=0)

	# Parameters
	sample_count, input_size = x.shape
	hidden_size = len(dataset)
	output_size = 1
	
	# Model
	model = NN(input_size, hidden_size, output_size, activation_function, device)

	# Loss and Optimizer
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	# Train
	epoch_start_time = datetime.now()
	for epoch in range(epochs):
		for index, (inputs, labels) in enumerate(training_dataloader):
			# Forward
			predicted = model(inputs)
			predicted = predicted.to(device)
			labels = labels.to(device)
			loss = criterion(predicted, labels)
			
			# Backward
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
				
			# Print
			if(index == 0):
				printEpochProgress(epoch, epochs, i, len(datasets), datasets_start_time, epoch_start_time, loss.item(), bestModel)
			
	# Test
	test_predicted = model(x_test)
	test_predicted = test_predicted.to(device)
	test_loss = criterion(test_predicted, y_test)
 
	# Overfitting
	train_predicted = model(x_train)
	train_predicted = train_predicted.to(device)
	train_loss = criterion(train_predicted, y_train)
	overfitting = ((test_loss.item() / (train_loss.item() + test_loss.item())) - 0.5) * 2

	# Test Data
	test_predicted = np.asarray(test_predicted.cpu().detach().numpy())
	y_test = np.asarray(y_test.cpu().detach().numpy())
 
	# Correlation
	correlation = np.corrcoef(np.array(test_predicted[:,0]), np.array(y_test[:,0]))
  
	# Update Best Model 
	bestModel.update(model, correlation[0][1], loss.item(), overfitting)
 
	# Append Results
	if (isNotNaN(correlation[0][1])):
		results["testCorrelation"].append(correlation[0][1])
  
	if (isNotNaN(loss.item())):
		results["testLoss"].append(loss.item())
  
	if (isNotNaN(overfitting)):
		results["overfitting"].append(overfitting)
 
 
def printInitial(datasets):
	global architecture
	global activation_function
	global determineBestModelBy
	global datasetsCount
	global repeatCount
	global epochs
	global batch_size
	global learning_rate

	os.system('cls')

	print("\033[0;36;40m", end = '')
	print("")
	print("Training Neural Network ")
	print("")
 
	print("Parameters:")
	print("architecture=" + architecture + ", ", end = '')
	print("activation_function=" + activation_function + ", ", end = '')
	print("determineBestModelBy=" + determineBestModelBy + ", ", end = '')
	print("datasetsCount=" + str(datasetsCount) + ", ", end = '')
	print("repeatCount=" + str(repeatCount) + ", ", end = '')
	print("epochs=" + str(epochs) + ", ", end = '')
	print("batch_size=" + str(batch_size) + ", ", end = '')
	print("learning_rate=" + str(learning_rate), end = '')
	print("")
	print("")
 
	print("Number of Datapoints: " + str(len(datasets[0])) + "	")
	print("\n"*5)

 
def getArchitecture():
	global architecture
    
	if (architecture == "deep"):
		return NN_Deep
	elif (architecture == "shallow"):
		return NN_Shallow
	elif(architecture == "single"):
		return NN_Single
	else:
		print("ERROR: Invalid Architecture Parameter Provided")
		return "error"
    
def getActivationFunction():
	global activation_function
    
	if (activation_function == "relu"):
		return nn.ReLU()
	elif (architecture == "tanh"):
		return nn.Tanh()
	else:
		print("ERROR: Invalid Activation Function Parameter Provided")
		return "error"


def calculateFinalResults(bestModel): 
	results["testCorrelationMax"] = max(results["testCorrelation"])
	results["testCorrelationAvg"] = sum(results["testCorrelation"]) / len(results["testCorrelation"])
	results["testCorrelationVar"] = np.var(results["testCorrelation"])
 
	results["testLossMin"] = min(results["testLoss"])
	results["testLossAvg"] = sum(results["testLoss"]) / len(results["testLoss"])
	results["testLossVar"] = np.var(results["testLoss"])
 
	results["overfittingMin"] = min(results["overfitting"])
	results["overfittingAvg"] = sum(results["overfitting"]) / len(results["overfitting"])
	results["overfittingVar"] = np.var(results["overfitting"])
 
	results["bestTestCorrelation"] = bestModel.correlation
	results["bestTestLoss"] = bestModel.loss
	results["bestTestOverfitting"] = bestModel.overfitting


def printResults(bestModel):
	print("")
	print("Results:")

	table = []
	table.append(["- Best Model Correlation", bestModel.correlation])
	table.append(["- Best Model Loss", bestModel.loss])
	table.append(["- Best Model Overfitting", bestModel.overfitting])
	table.append(["- Minimum Test Loss", "{:,}".format(results["testLossMin"])])
	table.append(["- Average Test Loss", "{:,}".format(results["testLossAvg"])])
	table.append(["- Loss Variance", "{:,}".format(results["testLossVar"])])
	table.append(["- Maximum Test Correlation", str(results["testCorrelationMax"])])
	table.append(["- Average Test Correlation", str(results["testCorrelationAvg"])])
	table.append(["- Correlation Variance", str(results["testCorrelationVar"])])
	table.append(["- Minimum Overfitting", str(results["overfittingMin"])])
	table.append(["- Average Overfitting", str(results["overfittingAvg"])])
	table.append(["- Overfitting Variance", str(results["overfittingVar"])])

	print(tabulate(table, tablefmt='plain'))


def run():
	try:
		datasets = getDatasets()
	
		printInitial(datasets)
	
		neural_network = getArchitecture()
		if(neural_network == "error"):
			return "error"

		activation_function = getActivationFunction()
		if(activation_function == "error"):
			return "error"
	
		bestModel = BestModel()

		datasets_start_time = datetime.now()
	
		for i in range(len(datasets)):
			runOnDataset(datasets, i, neural_network, activation_function, bestModel, datasets_start_time)

		calculateFinalResults(bestModel)
		printResults(bestModel)

		with open("./results/results.json", 'w', encoding='utf-8') as f:
			json.dump(results, f, ensure_ascii=False, indent=4)
	
		torch.save(bestModel.model.state_dict(), "./models/model.pt")
	except KeyboardInterrupt:
		return

run()