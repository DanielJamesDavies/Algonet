import fs from "fs";

codeToNumerical();

function codeToNumerical() {
	let implementations = JSON.parse(fs.readFileSync("./code.json", "utf8")).code;
	let { keywords, keyFunctions, replaceWords } = JSON.parse(fs.readFileSync("./words.json", "utf8"));

	let implementationsBags = [];

	// For Each Implementation
	for (let i = 0; i < implementations.length; i++) {
		let structuredLines = [];
		let parent = [];
		let functionName = "";

		// For Each Line of Code in Implementation i
		for (let j = 0; j < implementations[i].length; j++) {
			// Remove Tabs, Round Brackets, and Square Brackers
			let line = implementations[i][j]
				.split("\t")
				.filter((e) => e !== "")
				.join("")
				.replace(/[(\[\])]/g, " ")
				.split(" ");

			// If Declaring Function
			if (line[0] === "function") {
				functionName = line[1].split("(")[0];
				continue;
			}

			// If Blank Line
			if (line.join("") === "") continue;

			// If Parent End
			if (line.filter((e) => e !== "").join("") === "}") {
				parent.pop();
				continue;
			}

			let indentation = getIndentation(implementations[i][j]);

			line = line
				.map((word) => {
					if (word === functionName) return "recursive";

					let split = word.split(""); // Split Word into Characters
					if (split[split.length - 1] === ";") split.pop(); // Remove End Semicolon

					// If Increment or Decrement Operator
					if (
						(split[split.length - 2] === "+" && split[split.length - 1] === "+") ||
						(split[split.length - 2] === "-" && split[split.length - 1] === "-")
					)
						return split[split.length - 2] + split[split.length - 1];

					// If word includes a key function, return key function
					if (keyFunctions.includes(word.split(".")[1])) return word.split(".")[1];

					return word;
				})
				.join(" ");

			// Filter Line to only include keywords and key functions
			line = line.split(" ").filter((word) => keywords.includes(word) || keyFunctions.includes(word));

			// Get current parent
			while (indentation < parent.length) parent.pop();
			parent = parent.filter((e) => e !== undefined);

			line = line.map((word) => {
				// Replace word from keyword to new word
				let replacementWord = replaceWords.find((e) => e.from === word)?.to;
				if (replacementWord !== undefined) word = replacementWord;

				if (parent.length === 0) return word;

				// Return word with parent word
				return parent[parent.length - 1] + word.charAt(0).toUpperCase() + word.slice(1);
			});

			structuredLines = structuredLines.concat(line); // Add new structured line to all structured lines

			// Add current line as next parent if next indentation is more than current
			if (j < implementations[i].length - 1 && getIndentation(implementations[i][j + 1]) > parent.length) parent.push(line[0]);
		}

		// Get Bag of Words
		var bag = {};
		for (let j = 0; j < structuredLines.length; j++) {
			if (bag[structuredLines[j]] === undefined) {
				bag[structuredLines[j]] = 1;
			} else {
				bag[structuredLines[j]]++;
			}
		}
		implementationsBags.push(bag);
	}

	// Get Written Words in Order
	let finalWords = [];
	for (let i = 0; i < implementationsBags.length; i++) {
		for (const [key] of Object.entries(implementationsBags[i])) if (!finalWords.includes(key)) finalWords.push(key);
	}
	finalWords = finalWords.sort((a, b) => a.split("").length - b.split("").length);

	// Get Bags of Ordered Final Words
	let bags = implementationsBags.map((bag) => {
		return finalWords.map((word) => {
			if (bag[word] === undefined) return 0;
			return bag[word];
		});
	});

	fs.writeFileSync("./data/implementationsStructured.json", JSON.stringify({ bags }));
	fs.writeFileSync("./data/finalWords.json", JSON.stringify({ finalWords }));
}

function getIndentation(line) {
	let indentation = 0;
	let lineSplit = line.split("\t");
	while (lineSplit[indentation] === "") indentation++;
	return indentation - 1;
}
