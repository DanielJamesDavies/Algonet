import fs from "fs";
import prompt from "prompt";

function getIdleClustersData(fileDates) {
	let clusters = [];
	for (let i = 0; i < fileDates.length; i++) {
		let file = JSON.parse(fs.readFileSync("./data/Idle/turing2-idle-" + fileDates[i] + "-clusters.json", "utf8"));
		for (let j = 0; j < file.clusters.length; j++) {
			let clusterIndex = clusters.findIndex((e) => e.cluster === file.clusters[j].cluster);
			if (clusterIndex === -1) {
				clusters.push({ cluster: file.clusters[j].cluster, values: [file.clusters[j].average] });
			} else {
				clusters[clusterIndex].values.push(file.clusters[j].average);
			}
		}
	}
	for (let i = 0; i < clusters.length; i++) {
		let total = 0;
		for (let j = 0; j < clusters[i].values.length; j++) {
			total += clusters[i].values[j];
		}
		clusters[i].power = total / clusters[i].values.length;
		delete clusters[i].values;
	}
	clusters = clusters.sort((a, b) => b.cluster - a.cluster);
	return clusters;
}

async function selectClustersData(inputData, clusterArray) {
	let data = JSON.parse(JSON.stringify(inputData));
	data.cluster = [];

	let powers = [];
	for (let i = 0; i < data.power.length; i++) {
		if (!powers.includes(data.power[i])) powers.push(data.power[i]);
	}

	let powerClusters = [];
	if (clusterArray === undefined) {
		let promptInputs = powers.map((power, index) => {
			return index + "	" + power;
		});
		console.log(powers);

		let types = [
			"Low Idle",
			"Low Load",
			"High Idle",
			"High Load",
			"GPU Low Idle",
			"GPU Low Load",
			"GPU Mid Idle",
			"GPU Mid Load",
			"GPU High Idle",
			"GPU High Load",
			"Anomalous",
			"Mid Idle",
			"Mid Load",
		];
		types.forEach((type, index) => {
			console.log(index + " - " + type);
		});

		prompt.start();

		await new Promise(function (resolve, reject) {
			prompt.get(promptInputs, function (err, result) {
				if (err) {
					console.log(err);
					return reject("error");
				}
				for (const [key, value] of Object.entries(result)) {
					powerClusters.push({ power: parseFloat(key.split("\t")[1]), cluster: parseFloat(value) });
				}
				resolve("success");
			});
		});
	} else {
		for (let i = 0; i < powers.length; i++) {
			powerClusters.push({ power: powers[i], cluster: clusterArray[i] });
		}
	}

	for (let i = 0; i < data.power.length; i++) {
		let powerCluster = powerClusters.find((e) => e.power === data.power[i]);
		if (powerCluster === undefined || powerCluster.cluster === undefined) {
			data.cluster.push(-1);
		} else {
			data.cluster.push(powerCluster.cluster);
		}
	}

	data.clusterSequence = powerClusters.map((powerCluster) => {
		return powerCluster.cluster;
	});
	return data;
}

function splitAlgorithms(inputData, algoTypeIndex, runData) {
	let algorithms = [];

	const algoType = ["idle", "sorting", "graphs", "heuristicSearch", "searching"][algoTypeIndex + 1];

	let algorithmTypes = {
		sorting: ["bogoSort", "bubbleSort", "selectionSort", "quickSort", "radixSort"],
		graphs: ["preOrderDepthFirstSearch", "breadthFirstSearch", "dijkstrasAlgorithm", "primsAlgorithm"],
		heuristicSearch: ["randomMutationHillClimbing", "randomRestartHillClimbing", "stochasticHillClimbing", "simulatedAnnealing"],
		searching: ["binarySearchRecursive", "binarySearchIterative", "linearSearch", "jumpSearch", "interpolationSearch"],
	};

	let idleClusters = [0, 2, 4, 6, 8, 10, 11];
	for (let i = 0; i < inputData.power.length; i++) {
		if (idleClusters.includes(inputData.cluster[i])) continue;
		if (idleClusters.includes(inputData.cluster[i - 1])) {
			algorithms.push({
				algorithm: algorithmTypes[algoType][algorithms.length],
				iterations: runData[algoType][algorithmTypes[algoType][algorithms.length]].iterations,
				power: [],
				cluster: [],
			});
		}
		algorithms[algorithms.length - 1].power.push(inputData.power[i]);
		algorithms[algorithms.length - 1].cluster.push(inputData.cluster[i]);
	}

	return { algorithmsSplit: algorithms, clusterSequence: inputData.clusterSequence };
}

function getClusterAverages(inputData) {
	let clusterAverages = [];
	for (let i = 0; i < inputData.power.length; i++) {
		let clusterAveragesIndex = clusterAverages.findIndex((e) => e.cluster === inputData.cluster[i]);
		if (clusterAveragesIndex === -1) {
			clusterAverages.push({ cluster: inputData.cluster[i], powers: [inputData.power[i]] });
		} else {
			clusterAverages[clusterAveragesIndex].powers.push(inputData.power[i]);
		}
	}
	for (let i = 0; i < clusterAverages.length; i++) {
		let total = 0;
		for (let j = 0; j < clusterAverages[i].powers.length; j++) total += clusterAverages[i].powers[j];
		clusterAverages[i].average = total / clusterAverages[i].powers.length;
		delete clusterAverages[i].powers;
	}
	let clusterSequence = [];
	for (let i = 0; i < inputData.cluster.length; i++) {
		if (clusterSequence.length === 0 || clusterSequence[clusterSequence.length - 1] !== inputData.cluster[i])
			clusterSequence.push(inputData.cluster[i]);
	}
	return { clusterAverages, clusterSequence };
}

function calculateAlgorithmPower(algorithm, idleClustersData) {
	let total = 0;
	for (let i = 0; i < algorithm.power.length; i++) {
		let idleCluster = idleClustersData.find((e) => e.cluster === algorithm.cluster[i] - 1);
		if (idleCluster === undefined) continue;
		total += algorithm.power[i] - idleCluster.power;
	}

	let duration = 110; // In Minutes
	algorithm.milliwatts = total / algorithm.power.length; // Milliwatts
	algorithm.milliwattsPerIter = total / algorithm.power.length / (algorithm.iterations / duration / 60 / 2); // Milliwatts per Iteration
	//algorithm.milliwattsPerIter = total / algorithm.iterations; // Milliwatts per Iteration
	delete algorithm.power;
	delete algorithm.cluster;
	return algorithm;
}

function getFileName(algoType, input, fileDate) {
	switch (algoType) {
		case -1:
			return "Idle/turing2-idle-" + fileDate;
		case 0:
			return "Sorting/turing2-sorting-i" + input.toString() + "-" + fileDate;
		case 1:
			return "Graphs/turing2-graphs-i" + input.toString() + "-" + fileDate;
		case 2:
			return "HeuristicSearch/turing2-heuristic-search-i" + input.toString() + "-" + fileDate;
		case 3:
			return "Searching/turing2-searching-i" + input.toString() + "-" + fileDate;
	}
}

async function calculatePower() {
	// Start of User-Definable Parameters
	const algoType = 0;
	const fileDate = "02-feb";
	const input = 1;
	// End of User-Definable Parameters

	const fileName = getFileName(algoType, input, fileDate);

	let inputData = JSON.parse(fs.readFileSync("../05 Remove Noise System/data/" + fileName + "-noise-removed.json", "utf8"));

	// Idle
	if (algoType === -1) {
		let newData = await selectClustersData(inputData); // Get Clusters

		let { clusterAverages, clusterSequence } = getClusterAverages(newData); // Get Average Power of each Cluster

		fs.writeFileSync("./data/" + fileName + "-clusters.json", JSON.stringify({ clusters: clusterAverages, sequence: clusterSequence }));
	} else {
		const idleClustersDates = [
			"21-jan",
			"29-jan",
			"08-feb",
			"13-feb",
			"18-feb",
			"19-feb",
			"20-feb",
			"21-feb",
			"22-feb",
			"08-mar",
			"11-mar",
			"16-mar",
		];

		const idleClustersData = getIdleClustersData(idleClustersDates);

		const runData = JSON.parse(fs.readFileSync("../02 Run Algorithms System/data/" + fileName + "-run.json", "utf8"));

		const newData = await selectClustersData(inputData); // Get Clusters, Optional Second Input for Cluster Sequence

		const { algorithmsSplit, clusterSequence } = splitAlgorithms(newData, algoType, runData); // Get Algorithms

		const algorithms = algorithmsSplit.map((algorithm) => calculateAlgorithmPower(algorithm, idleClustersData));

		fs.writeFileSync("./data/" + fileName + "-power.json", JSON.stringify({ algorithms, clusterSequence }));
	}
}
calculatePower();
