package utility;

public class ArraySumAndAverageAndMaxAndMin {
	public static double getSum(double[] array) {
		double sum = 0.0;
		for (double v : array) {
			sum += v;
		}
		return sum;
	}

	public static double getSum(int[] array) {
		int sum = 0;
		for (int v : array) {
			sum += v;
		}
		return sum;
	}

	public static double getAverage(double[] array) {
		return 1.0 * getSum(array) / array.length;
	}

	public static double getAverage(int[] array) {
		return 1.0 * getSum(array) / array.length;
	}

	public static double getMin(double[] array) {
		double minest = Double.MAX_VALUE;
		for (double v : array) {
			minest = Math.min(minest, v);
		}
		return minest;
	}

	public static double getMin(int[] array) {
		int minest = Integer.MAX_VALUE;
		for (int v : array) {
			minest = Math.min(minest, v);
		}
		return minest;
	}

	public static double getMax(double[] array) {
		double maxest = Double.MIN_VALUE;
		for (double v : array) {
			maxest = Math.max(maxest, v);
		}
		return maxest;
	}

	public static double getMax(int[] array) {
		int maxest = Integer.MIN_VALUE;
		for (int v : array) {
			maxest = Math.max(maxest, v);
		}
		return maxest;
	}
}
