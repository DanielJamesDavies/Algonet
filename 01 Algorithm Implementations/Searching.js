export const Searching = () => {
	function binarySearchRecursive(value, array, start, end) {
		if (start === undefined) {
			start = 0;
		}
		if (end === undefined) {
			end = array.length - 1;
		}
		const mid = Math.floor((start + end) / 2);
		if (array[mid] === value) {
			return mid;
		}
		if (start >= end) {
			return -1;
		}
		if (array[mid] >= value) {
			return binarySearchRecursive(value, array, start, mid - 1);
		}
		return binarySearchRecursive(value, array, mid + 1, end);
	}

	function binarySearchIterative(value, array) {
		let low = 0;
		let mid = 0;
		let high = array.length;
		while (true) {
			mid = Math.floor(low + (high - low) / 2);
			if (array[mid] < value) {
				low = mid + 1;
			}
			if (array[mid] > value) {
				high = mid - 1;
			}
			if (array[mid] === value) {
				break;
			}
			if (high < 0 || (mid === low && high === mid)) {
				return -1;
			}
		}
		return mid;
	}

	function linearSearch(value, array) {
		for (let i = 0; i < array.length; i++) {
			if (array[i] === value) {
				return i;
			}
		}
		return -1;
	}

	function jumpSearch(value, array) {
		let step = Math.floor(Math.sqrt(array.length));
		let index = 0;

		while (index < array.length && array[index] < value) {
			index += step;
		}
		if (index > array.length - 1) {
			index = array.length - 1;
		}

		let startStep = index - step;
		if (startStep < 0) startStep = 0;
		while (index >= startStep) {
			if (array[index] === value) {
				return index;
			}
			index--;
		}
		return -1;
	}

	function interpolationSearch(value, array) {
		let low = 0;
		let mid = 0;
		let high = array.length - 1;

		while (low <= high && value >= array[low] && value <= array[high]) {
			mid = low + (value - array[low]) * Math.floor((high - low) / (array[high] - array[low]));
			if (array[mid] == value) {
				return mid;
			}
			if (array[mid] < value) {
				low = mid + 1;
			}
			if (array[mid] > value) {
				high = mid - 1;
			}
		}
		return -1;
	}

	return { binarySearchRecursive, binarySearchIterative, linearSearch, jumpSearch, interpolationSearch };
};
