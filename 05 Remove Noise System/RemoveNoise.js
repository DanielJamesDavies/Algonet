import fs from "fs";

class Data {
	constructor(data) {
		this.power = data.power;
		this.time = data.time;
	}

	get() {
		return { power: this.power, time: this.time };
	}

	smooth(changeThreshold) {
		this.power = smoothData(this.power, changeThreshold);
	}

	removeAnomalies(anomalyThreshold) {
		this.power = removeAnomalies(this.power, anomalyThreshold);
	}
}

removeNoise();

function removeNoise() {
	let algoType = 0;
	const input = 1;
	const fileDate = "02-feb";

	const fileName = getFileName(algoType, input, fileDate);

	var originalData = JSON.parse(fs.readFileSync("../03 Record Power System/data/" + fileName + "-rec.json", "utf8"));
	const data = new Data(originalData);

	// Operations
	data.smooth(2200);
	data.removeAnomalies(20);
	data.smooth(900);
	data.removeAnomalies(400);
	data.smooth(1000);
	data.removeAnomalies(400);

	fs.writeFileSync("./data/" + fileName + "-noise-removed1.json", JSON.stringify(data.get()));
}

function getFileName(algoType, input, fileDate) {
	let fileName = "";
	switch (algoType) {
		case -1:
			algoType = "idle";
			fileName += "Idle/turing2-idle";
			break;
		case 0:
			algoType = "sorting";
			fileName += "Sorting/turing2-sorting-i" + input.toString();
			break;
		case 1:
			algoType = "graphs";
			fileName += "Graphs/turing2-graphs-i" + input.toString();
			break;
		case 2:
			algoType = "heuristicSearch";
			fileName += "HeuristicSearch/turing2-heuristic-search-i" + input.toString();
			break;
		case 3:
			algoType = "searching";
			fileName += "Data/Searching/turing2-searching-i" + input.toString();
			break;
	}
	fileName += "-" + fileDate;
	return fileName;
}

function smoothData(inputData, changeThreshold) {
	console.log("Smoothing Data \t\t\t changeThreshold=" + changeThreshold);

	var oldData = JSON.parse(JSON.stringify(inputData));
	var newData = [];

	var startIndex = 0;
	var total = 0;
	for (let i = 0; i < oldData.length; i++) {
		total += oldData[i];
		if (i === oldData.length - 1 || Math.abs(oldData[i + 1] - oldData[i - 1]) > changeThreshold) {
			var avg = Math.floor((total / (i - startIndex + 1)) * 1000) / 1000;
			for (let j = 0; j < i - startIndex + 1; j++) {
				newData.push(avg);
			}
			startIndex = i + 1;
			total = 0;
		}
	}
	return newData;
}

function removeAnomalies(inputData, anomalyThreshold) {
	var data = JSON.parse(JSON.stringify(inputData));

	// Find Anomalies
	let frequencies = [];
	for (let i = 0; i < data.length; i++) {
		let index = frequencies.findIndex((e) => e.power === data[i]);
		if (index === -1) {
			frequencies.push({ power: data[i], frequency: 1 });
		} else {
			frequencies[index].frequency++;
		}
	}
	let anomalies = frequencies.filter((e) => e.frequency < anomalyThreshold).map((e) => e.power);
	const anomaliesCount = anomalies.length;

	// Remove Anomalies
	while (anomalies.length > 0) {
		// Console Log Progress
		if (Math.floor(100 - (anomalies.length / anomaliesCount) * 100) % 5 === 0) {
			process.stdout.clearLine(0);
			process.stdout.cursorTo(0);
			process.stdout.write(
				"Removing Anomalies from Data \t anomalyThreshold=" +
					anomalyThreshold +
					" \t Progress: " +
					Math.floor(100 - (anomalies.length / anomaliesCount) * 100) +
					"%"
			);
		}

		// Find Next Anomaly
		const i = data.findIndex((e) => e === anomalies[0]);
		if (i === -1) {
			anomalies.shift();
			continue;
		}

		// Find Next Non-Anomalous Power
		let nextPowerIndex = 0;
		for (let j = i + 1; j < data.length; j++) {
			if (!anomalies.includes(data[j])) {
				nextPowerIndex = j;
				break;
			}
		}

		for (let j = i; j < nextPowerIndex; j++) {
			const powerDifference = data[nextPowerIndex] - data[j - 1];
			if (i === 0) {
				data[j] = data[nextPowerIndex];
			} else if (data[j] > data[j - 1] + powerDifference / 2) {
				if (Math.sign(powerDifference) === -1) {
					data[j] = data[j - 1];
				} else {
					data[j] = data[nextPowerIndex];
				}
			} else {
				if (Math.sign(powerDifference) === -1) {
					data[j] = data[nextPowerIndex];
				} else {
					data[j] = data[j - 1];
				}
			}
		}
	}
	process.stdout.clearLine(0);
	process.stdout.cursorTo(0);
	process.stdout.write("Removing Anomalies from Data \t anomalyThreshold=" + anomalyThreshold + " \t Progress: 100%");
	process.stdout.write("\n");

	return data;
}
