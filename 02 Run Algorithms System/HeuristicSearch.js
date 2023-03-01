export const HeuristicSearch = () => {
	function generateRandomSolution(n) {
		var solution = "";
		for (let i = 0; i < n; i++) {
			solution += Math.floor(Math.random() * 2);
		}
		return solution;
	}

	function smallChange(solution) {
		var solutionSplit = solution.split("");
		var changeIndex = Math.floor(Math.random() * solutionSplit.length);
		if (solutionSplit[changeIndex] === "1") {
			solutionSplit[changeIndex] = "0";
		} else {
			solutionSplit[changeIndex] = "1";
		}
		return solutionSplit.join("");
	}

	function getFitness(solution, weights) {
		var solutionSplit = solution.split("");
		var left = 0;
		var right = 0;
		for (let i = 0; i < weights.length; i++) {
			if (solutionSplit[i] === "0") {
				left += weights[i];
			} else {
				right += weights[i];
			}
		}
		return Math.abs(left - right);
	}

	function randomMutationHillClimbing(weights, iter) {
		var solution = generateRandomSolution(weights.length);
		var fit = getFitness(solution, weights);
		for (let i = 0; i < iter; i++) {
			var newSolution = smallChange(solution);
			var newFit = getFitness(newSolution, weights);
			if (newFit < fit) {
				solution = newSolution;
				fit = newFit;
			}
		}
		return solution;
	}

	function randomRestartHillClimbing(weights, iter) {
		var solution = generateRandomSolution(weights.length);
		var fit = getFitness(solution, weights);
		var bestSolution = solution;
		var bestFit = fit;
		for (let i = 0; i < iter; i++) {
			var random = Math.floor(Math.random() * 10) / 10;
			if (0.1 > random) {
				solution = generateRandomSolution(weights.length);
				fit = getFitness(solution, weights);
				if (fit < bestFit) {
					bestSolution = solution;
					bestFit = fit;
				}
			}
			var newSolution = smallChange(solution);
			var newFit = getFitness(newSolution, weights);
			if (newFit < fit) {
				solution = newSolution;
				fit = newFit;
			}
			if (newFit < bestFit) {
				bestSolution = newSolution;
				bestFit = newFit;
			}
		}
		return bestSolution;
	}

	function stochasticHillClimbing(weights, iter) {
		var solution = generateRandomSolution(weights.length);
		var fit = getFitness(solution, weights);
		var t = 1;
		for (let i = 0; i < iter; i++) {
			var newSolution = smallChange(solution);
			var newFit = getFitness(newSolution, weights);
			var random = Math.floor(Math.random() * 10) / 10;
			var accept = 1 / (1 + Math.exp((newFit - fit) / t));
			if (newFit < fit || random < accept) {
				solution = newSolution;
				fit = newFit;
			}
		}
		return solution;
	}

	function simulatedAnnealing(weights, iter) {
		var solution = generateRandomSolution(weights.length);
		var fit = getFitness(solution, weights);
		var initialTemperature = 100;
		var temperature = initialTemperature;
		for (let i = 0; i < iter; i++) {
			var newSolution = smallChange(solution);
			var newFit = getFitness(newSolution, weights);

			temperature = initialTemperature / (i + 1);
			var random = Math.floor(Math.random() * 10) / 10;
			var accept = 1 / (1 + Math.exp((newFit - fit) / temperature));
			if (newFit < fit || random < accept) {
				solution = newSolution;
				fit = newFit;
			}
		}
		return solution;
	}

	return { randomMutationHillClimbing, randomRestartHillClimbing, stochasticHillClimbing, simulatedAnnealing };
};
