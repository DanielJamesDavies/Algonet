export const Sorting = () => {
	function bogoSort(array) {
		var isSorted = false;
		var oldArray = array;
		while (!isSorted) {
			isSorted = true;
			for (var i = 0; i < array.length - 1; i++) {
				if (array[i] > array[i + 1]) {
					isSorted = false;
				}
			}

			if (!isSorted) {
				oldArray = array;
				array = [];
				while (oldArray.length !== 0) {
					array.push(oldArray.splice(Math.floor(Math.random() * oldArray.length), 1)[0]);
				}
			}
		}
		return array;
	}

	function bubbleSort(array) {
		var noSwaps = false;
		var numSorted = 0;
		while (!noSwaps) {
			noSwaps = true;
			for (let i = 0; i < array.length - numSorted; i++) {
				if (array[i] > array[i + 1]) {
					var temp = array[i];
					array[i] = array[i + 1];
					array[i + 1] = temp;
					noSwaps = false;
				}
			}
			numSorted++;
		}
		return array;
	}

	function selectionSort(array) {
		for (let i = 0; i < array.length; i++) {
			var minIndex = i;
			for (let j = i + 1; j < array.length; j++) {
				if (array[j] < array[minIndex]) minIndex = j;
			}
			var temp = array[i];
			array[i] = array[minIndex];
			array[minIndex] = temp;
		}
		return array;
	}

	function quickSort(array) {
		if (array.length < 2) return array;
		var pivotIndex = Math.floor(Math.random() * array.length);
		var pivot = array[pivotIndex];
		var lower = [];
		var higher = [];
		for (let i = 0; i < array.length; i++) {
			if (i === pivotIndex) {
				continue;
			} else if (array[i] < pivot) {
				lower.push(array[i]);
			} else {
				higher.push(array[i]);
			}
		}
		return quickSort(lower).concat([pivot], quickSort(higher));
	}

	function radixSort(array) {
		var maxDigits = String(Math.max(...array)).split("").length;
		var buckets = [];
		for (let i = 0; i < maxDigits; i++) {
			buckets = [[], [], [], [], [], [], [], [], [], []];
			for (let j = 0; j < array.length; j++) {
				if (String(array[j]).split("").length < maxDigits - i) {
					buckets[0].push(array[j]);
				} else {
					buckets[Math.floor(Math.abs(array[j]) / Math.pow(10, i)) % 10].push(array[j]);
				}
			}
			array = [];
			buckets.forEach((bucket) => {
				array = array.concat(bucket);
			});
		}
		return array;
	}

	return { bogoSort, bubbleSort, selectionSort, quickSort, radixSort };
};
