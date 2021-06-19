package nlp;

public class EntropyHelper {
	public static double getEntropy(double[] probs) {
		double entropy = 0.0;
		for (double prob : probs) {
			if (prob > 0) {
				entropy += -prob * Math.log(prob);
			}
		}
		return entropy;
	}

	public static double getEntropy(int[] counts) {
		double[] probs = new double[counts.length];
		int sum = 0;
		for (int count : counts) {
			sum += count;
		}
		for (int i = 0; i < counts.length; ++i) {
			probs[i] = 1.0 * counts[i] / sum;
		}
		return getEntropy(probs);
	}
}
