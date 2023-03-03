import fs from "fs";

import { Sorting } from "./Sorting.js";
import { Graphs } from "./Graphs.js";
import { HeuristicSearch } from "./HeuristicSearch.js";
import { Searching } from "./Searching.js";

var algoType = 0;
var inputIndex = 0;

// All in Milliseconds
var runtime = 110 * 60 * 1000;
var cooltime = 10 * 60 * 1000;
var waittime = 60 * 60 * 1000;

var runData = {
	runtime,
	cooltime,
	algoType,
	inputIndex,
	sorting: {
		bogoSort: { iterations: 0, startTime: 0 },
		bubbleSort: { iterations: 0, startTime: 0 },
		selectionSort: { iterations: 0, startTime: 0 },
		quickSort: { iterations: 0, startTime: 0 },
		radixSort: { iterations: 0, startTime: 0 },
	},
	graphs: {
		preOrderDepthFirstSearch: { iterations: 0, startTime: 0 },
		breadthFirstSearch: { iterations: 0, startTime: 0 },
		dijkstrasAlgorithm: { iterations: 0, startTime: 0 },
		primsAlgorithm: { iterations: 0, startTime: 0 },
	},
	heuristicSearch: {
		randomMutationHillClimbing: { iterations: 0, startTime: 0 },
		randomRestartHillClimbing: { iterations: 0, startTime: 0 },
		stochasticHillClimbing: { iterations: 0, startTime: 0 },
		simulatedAnnealing: { iterations: 0, startTime: 0 },
	},
	searching: {
		binarySearchRecursive: { iterations: 0, startTime: 0 },
		binarySearchIterative: { iterations: 0, startTime: 0 },
		linearSearch: { iterations: 0, startTime: 0 },
		jumpSearch: { iterations: 0, startTime: 0 },
		interpolationSearch: { iterations: 0, startTime: 0 },
	},
};

async function runAlgo(algoType, algoKey, algorithm, args) {
	runData[algoType][algoKey].startTime = Date.now();
	while (Date.now() - runData[algoType][algoKey].startTime < runtime) {
		algorithm(...args);
		runData[algoType][algoKey].iterations++;
	}

	await new Promise((resolve) => setTimeout(resolve, cooltime));
}

async function run() {
	const { bogoSort, bubbleSort, selectionSort, quickSort, radixSort } = Sorting();
	const { preOrderDepthFirstSearch, breadthFirstSearch, dijkstrasAlgorithm, primsAlgorithm } = Graphs();
	const { randomMutationHillClimbing, randomRestartHillClimbing, stochasticHillClimbing, simulatedAnnealing } = HeuristicSearch();
	const { binarySearchRecursive, binarySearchIterative, linearSearch, jumpSearch, interpolationSearch } = Searching();

	const inputs = JSON.parse(fs.readFileSync("inputs.json", "utf8"));

	console.log("Parameters: algoType=" + algoType + ", " + "inputIndex=" + inputIndex);

	await new Promise((resolve) => setTimeout(resolve, waittime));

	console.log("Running...");

	switch (algoType) {
		// Sorting
		case 0:
			await runAlgo("sorting", "bogoSort", bogoSort, [JSON.parse(JSON.stringify(inputs.sorting[inputIndex]))]);

			await runAlgo("sorting", "bubbleSort", bubbleSort, [JSON.parse(JSON.stringify(inputs.sorting[inputIndex]))]);

			await runAlgo("sorting", "selectionSort", selectionSort, [JSON.parse(JSON.stringify(inputs.sorting[inputIndex]))]);

			await runAlgo("sorting", "quickSort", quickSort, [JSON.parse(JSON.stringify(inputs.sorting[inputIndex]))]);

			await runAlgo("sorting", "radixSort", radixSort, [JSON.parse(JSON.stringify(inputs.sorting[inputIndex]))]);
			break;

		// Graphs
		case 1:
			await runAlgo("graphs", "preOrderDepthFirstSearch", preOrderDepthFirstSearch, [
				JSON.parse(JSON.stringify(inputs.graphs[inputIndex])),
				0,
			]);

			await runAlgo("graphs", "breadthFirstSearch", breadthFirstSearch, [JSON.parse(JSON.stringify(inputs.graphs[inputIndex])), 0]);

			await runAlgo("graphs", "dijkstrasAlgorithm", dijkstrasAlgorithm, [
				JSON.parse(JSON.stringify(inputs.graphs[inputIndex])),
				0,
				JSON.parse(JSON.stringify(inputs.graphs[inputIndex])).length - 1,
			]);

			await runAlgo("graphs", "primsAlgorithm", primsAlgorithm, [JSON.parse(JSON.stringify(inputs.graphs[inputIndex]))]);

			break;

		// Heuristic Search
		case 2:
			let iter = 100;

			await runAlgo("heuristicSearch", "randomMutationHillClimbing", randomMutationHillClimbing, [
				JSON.parse(JSON.stringify(inputs.weights[inputIndex])),
				iter,
			]);

			await runAlgo("heuristicSearch", "randomRestartHillClimbing", randomRestartHillClimbing, [
				JSON.parse(JSON.stringify(inputs.weights[inputIndex])),
				iter,
			]);

			await runAlgo("heuristicSearch", "stochasticHillClimbing", stochasticHillClimbing, [
				JSON.parse(JSON.stringify(inputs.weights[inputIndex])),
				iter,
			]);

			await runAlgo("heuristicSearch", "simulatedAnnealing", simulatedAnnealing, [
				JSON.parse(JSON.stringify(inputs.weights[inputIndex])),
				iter,
			]);

			break;

		// Searching
		case 3:
			await runAlgo("searching", "binarySearchRecursive", binarySearchRecursive, JSON.parse(JSON.stringify(inputs.searching[inputIndex])));

			await runAlgo("searching", "binarySearchIterative", binarySearchIterative, JSON.parse(JSON.stringify(inputs.searching[inputIndex])));

			await runAlgo("searching", "linearSearch", linearSearch, JSON.parse(JSON.stringify(inputs.searching[inputIndex])));

			await runAlgo("searching", "jumpSearch", jumpSearch, JSON.parse(JSON.stringify(inputs.searching[inputIndex])));

			await runAlgo("searching", "interpolationSearch", interpolationSearch, JSON.parse(JSON.stringify(inputs.searching[inputIndex])));

			break;
	}

	// Save Data
	var fileName =
		"./data/turing2-" +
		(algoType === 0 ? "sorting" : algoType === 1 ? "graphs" : algoType === 2 ? "heuristic-search" : "searching") +
		"-i" +
		(inputIndex + 1) +
		"-" +
		new Date(Date.now()).toLocaleString("en-GB", { day: "numeric", month: "short" }).toLocaleLowerCase().split(" ").join("-") +
		"-run.json";
	console.log("Results Saved to: " + fileName);
	fs.writeFileSync(fileName, JSON.stringify(runData));
}
run();
