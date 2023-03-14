import json
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import AffinityPropagation
from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter



### Start of User-Definable Parameters
datasetFileName = "dataset.json"
isBogoSortInDataset = "true"
### End of User-Definable Parameters



def run():
	# Get Data
    with open("../07 Create Dataset/data/" + datasetFileName) as f:
		dataset = json.load(f)["data"]

	code = []
	powers = []
	for dataPoint in dataset:
		dataPointCode = dataPoint
		dataPointPower = dataPointCode.pop(0)
		dataPointCode.pop(0)
		if(not dataPointCode in code):
			code.append(dataPointCode)
			powers.append(dataPointPower)

	print(sum(powers) / len(powers))

	# Cosine Distance
	distances = []
	for i in range(len(code)):
		distances.append([]);
		for j in range(len(code)): 
			distances[i].append(distance.cosine(code[i], code[j]))

	# Affinity Propagation
	distances = np.array(distances)
	simplefilter("ignore", category=ConvergenceWarning)
	clusters = AffinityPropagation(random_state=None).fit(np.array(code)).labels_

	if(-1 in clusters):
		print("Did not converge. Please try again.")
	else:
		# Display Data
		algorithmNames = [
				"bubbleSort",
				"selectionSort",
				"quickSort",
				"radixSort",
				"preOrderDepthFirstSearch",
				"breadthFirstSearch",
				"dijkstrasAlgorithm",
				"primsAlgorithm",
				"randomMutationHillClimbing",
				"randomRestartHillClimbing",
				"stochasticHillClimbing",
				"simulatedAnnealing",
				"binarySearchRecursive",
				"binarySearchIterative",
				"linearSearch",
				"jumpSearch",
				"interpolationSearch",
			]

		if (isBogoSortInDataset == "true"):
			algorithmNames.insert(0, "bogoSort")

		clustersAlgoNames = []
		for i in range(max(clusters) + 1):
			clustersAlgoNames.append([])

		for i in range(len(clusters)):
			clustersAlgoNames[clusters[i]].append(algorithmNames[i])
			
		print("")		

		sumClustersPowerVariance  = 0
		for cluster in clustersAlgoNames:
			clusterPowers = []
			for name in cluster:
				clusterPowers.append(powers[algorithmNames.index(name)])
			clusterPowersVar = np.var(clusterPowers)
			sumClustersPowerVariance += clusterPowersVar
			print(cluster, clusterPowersVar)

		print(sumClustersPowerVariance)

run()