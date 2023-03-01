export const Graphs = () => {
	function preOrderDepthFirstSearch(graph, startingNode) {
		var currPath = [startingNode];
		var visited = [startingNode];

		do {
			var hasReachedDeadEnd = false;
			while (!hasReachedDeadEnd) {
				hasReachedDeadEnd = true;
				for (let i = 0; i < graph[currPath[currPath.length - 1]].length; i++) {
					if (graph[currPath[currPath.length - 1]][i] !== 0 && !visited.includes(i)) {
						hasReachedDeadEnd = false;
						visited.push(i);
						currPath.push(i);
						break;
					}
				}
			}
			currPath.pop();
		} while (currPath.length != 0);

		return visited.join(", ");
	}

	function breadthFirstSearch(graph, startingNode) {
		var visited = [startingNode];

		for (let i = 0; i < graph.length; i++) {
			for (let j = 0; j < graph[visited[i]].length; j++) {
				if (graph[visited[i]][j] !== 0 && !visited.includes(j)) {
					visited.push(j);
				}
			}
		}

		return visited.join(", ");
	}

	function dijkstrasAlgorithm(graph, startingNode, targetNode) {
		var nodes = [];
		for (let i = 0; i < graph.length; i++) {
			nodes.push({ from: 0, distance: "infinity" });
		}
		nodes[startingNode] = { from: 0, distance: 0 };

		var visited = [];

		var currNode = startingNode;
		while (currNode !== undefined && !visited.includes(targetNode)) {
			for (let i = 0; i < graph[currNode].length; i++) {
				if (graph[currNode][i] !== 0) {
					var newDistance = nodes[currNode].distance + graph[currNode][i];
					if (nodes[i].distance === "infinity" || newDistance < nodes[i].distance) {
						nodes[i] = { from: currNode, distance: newDistance };
					}
				}
			}
			visited.push(currNode);
			let nextNode;
			for (let i = 0; i < nodes.length; i++) {
				if (
					!visited.includes(i) &&
					(nextNode === undefined || nodes[i].distance < nodes[nextNode].distance) &&
					nodes[i].distance !== "infinity"
				)
					nextNode = i;
			}
			currNode = nextNode;
		}

		var path = [targetNode];
		while (path[0] !== startingNode) {
			path.splice(0, 0, nodes[path[0]].from);
		}

		return path.join(", ");
	}

	function primsAlgorithm(graph) {
		var minimumSpanningTree = [];
		for (let i = 0; i < graph.length; i++) {
			minimumSpanningTree.push([]);
			for (let j = 0; j < graph[i].length; j++) {
				minimumSpanningTree[i].push(0);
			}
		}

		var currNode = Math.floor(Math.random() * graph.length);
		var edgeQueue = [];
		var visited = [currNode];

		do {
			for (let i = 0; i < graph[currNode].length; i++) {
				if (graph[currNode][i] !== 0) {
					edgeQueue.push([currNode, i]);
				}
			}
			edgeQueue.sort((a, b) => graph[a[0]][a[1]] - graph[b[0]][b[1]]);

			var savedCurrNode = JSON.parse(JSON.stringify(currNode));
			while (edgeQueue.length !== 0 && currNode === savedCurrNode) {
				if (!visited.includes(edgeQueue[0][1])) {
					visited.push(edgeQueue[0][1]);
					minimumSpanningTree[edgeQueue[0][0]][edgeQueue[0][1]] = graph[edgeQueue[0][0]][edgeQueue[0][1]];
					minimumSpanningTree[edgeQueue[0][1]][edgeQueue[0][0]] = graph[edgeQueue[0][1]][edgeQueue[0][0]];
					currNode = edgeQueue[0][1];
				}
				edgeQueue.splice(0, 1);
			}
		} while (visited.length < graph.length);
		return minimumSpanningTree;
	}

	return { preOrderDepthFirstSearch, breadthFirstSearch, dijkstrasAlgorithm, primsAlgorithm };
};
