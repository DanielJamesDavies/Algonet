import fs from "fs";

function createDataset() {
	// Start of User-Definable Parameters
	const files = [
		"turing2-sorting-i1-02-feb",
		"turing2-sorting-i2-15-feb",
		"turing2-sorting-i3-17-feb",
		"turing2-graphs-i1-06-feb",
		"turing2-graphs-i2-16-feb",
		"turing2-graphs-i3-10-feb",
		"turing2-heuristic-search-i1-12-mar",
		"turing2-heuristic-search-i2-13-mar",
		"turing2-heuristic-search-i3-15-mar",
		"turing2-searching-i1-23-feb",
		"turing2-searching-i2-07-mar",
		"turing2-searching-i3-28-feb",
	];

	const includeBogoSort = true;
	// End of User-Definable Parameters

	let dataset = [];

	const implementationsStructured = JSON.parse(
		fs.readFileSync("../04 Code-to-Numerical System/data/implementationsStructured.json", "utf8")
	)?.bags;

	const inputs = JSON.parse(fs.readFileSync("../02 Run Algorithms System/inputs.json", "utf8"));

	const algorithmNames = [
		"bogoSort",
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
	];

	files.map((fileName) => {
		let inputIndex = 0;
		let inputKey = fileName.split("-")[1];
		let inputSizeMultiplier = 1;
		let folder = "";
		switch (inputKey) {
			case "sorting":
				folder = "Sorting";
				inputIndex = parseInt(fileName.split("-")[2].split("i")[1]) - 1;
				break;
			case "graphs":
				folder = "Graphs";
				inputIndex = parseInt(fileName.split("-")[2].split("i")[1]) - 1;
				break;
			case "heuristic":
				folder = "HeuristicSearch";
				inputIndex = parseInt(fileName.split("-")[3].split("i")[1]) - 1;
				inputKey = "weights";
				inputSizeMultiplier = 100;
				break;
			case "searching":
				folder = "Searching";
				inputIndex = parseInt(fileName.split("-")[2].split("i")[1]) - 1;
				break;
		}
		let algorithmsPower = JSON.parse(
			fs.readFileSync("../06 Calculate Power System/data/" + folder + "/" + fileName + "-power.json", "utf8")
		)?.algorithms;
		if (!algorithmsPower) return;
		algorithmsPower.map((algorithm) => {
			let inputSize = inputs[inputKey][inputIndex].length;
			if (inputKey === "graphs") inputSize = inputs[inputKey][inputIndex].length * inputs[inputKey][inputIndex].length;
			if (includeBogoSort || algorithm.algorithm !== "bogoSort")
				dataset.push(
					[algorithm.milliwattsPerIter, inputSize * inputSizeMultiplier].concat(
						implementationsStructured[algorithmNames.findIndex((e) => e === algorithm.algorithm)]
					)
				);
		});
		return;
	});

	console.log("Data Point Count: " + dataset.length);
	fs.writeFileSync("./data/dataset.json", JSON.stringify({ data: dataset }));
}
createDataset();
