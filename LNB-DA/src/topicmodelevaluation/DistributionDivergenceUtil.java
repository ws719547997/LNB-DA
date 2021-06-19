package topicmodelevaluation;

import main.Constant;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

/**
 * Kullback-Leibler divergence & Jensen-Shannon divergence.
 */
public class DistributionDivergenceUtil {
	/**
	 * Distribution1 and distribution2 should share the same indexes.
	 */
	public static double getKLDivergence(double[] distribution1,
			double[] distribution2) {
		assert (distribution1.length == distribution2.length) : "Distribution1 and distribution2 should share the same indexes.";

		double divergence = 0.0;
		for (int i = 0; i < distribution1.length; ++i) {
			double prob1 = distribution1[i];
			double prob2 = distribution2[i];
			divergence += prob1 * (Math.log(prob1) - Math.log(prob2));
		}
		return divergence;
	}

	/**
	 * Distribution1 and distribution2 should share the same indexes.
	 */
	public static double getSymmetricKLDivergence(double[] distribution1,
			double[] distribution2) {
		return 0.5 * (getKLDivergence(distribution1, distribution2) + getKLDivergence(
				distribution2, distribution1));
	}

	/**
	 * Distribution1 and distribution2 should share the same indexes.
	 */
	public static double getJSDivergence(double[] distribution1,
			double[] distribution2) {
		assert (distribution1.length == distribution2.length) : "Distribution1 and distribution2 should share the same indexes.";

		double[] mid = new double[distribution1.length];
		for (int i = 0; i < distribution1.length; ++i) {
			mid[i] = (distribution1[i] + distribution2[i]) * 0.5;
		}
		return 0.5 * getKLDivergence(distribution1, mid) + 0.5
				* getKLDivergence(distribution2, mid);
	}

	/**
	 * Calculate the KL-Divergence given the IDs with non-smooth probability.
	 */
	public static double getKLDivergence(Map<Integer, Double> map1,
			Map<Integer, Double> map2) {
		HashSet<Integer> hsUniqueIds = new HashSet<Integer>();
		double divergence = 0.0;
		for (Map.Entry<Integer, Double> entry : map1.entrySet()) {
			int id = entry.getKey();
			double prob1 = entry.getValue();
			if (!hsUniqueIds.contains(id)) {
				hsUniqueIds.add(id);

				double prob2 = Constant.SMOOTH_PROBABILITY;
				Double probObject2 = map2.get(id);
				if (probObject2 != null) {
					prob2 = probObject2;
				}
				divergence += prob1 * (Math.log(prob1) - Math.log(prob2));
			}
		}
		for (Map.Entry<Integer, Double> entry : map2.entrySet()) {
			int id = entry.getKey();
			double prob2 = entry.getValue();
			if (!hsUniqueIds.contains(id)) {
				hsUniqueIds.add(id);

				double prob1 = Constant.SMOOTH_PROBABILITY;
				Double probObject1 = map1.get(id);
				if (probObject1 != null) {
					prob1 = probObject1;
				}
				divergence += prob1 * (Math.log(prob1) - Math.log(prob2));
			}
		}
		return divergence;
	}

	public static double getSymmetricKLDivergence(Map<Integer, Double> map1,
			Map<Integer, Double> map2) {
		return 0.5 * (getKLDivergence(map1, map2) + getKLDivergence(map2, map1));
	}

	public static double getJSDivergence(Map<Integer, Double> map1,
			Map<Integer, Double> map2) {
		Map<Integer, Double> mid = new HashMap<Integer, Double>();
		for (Map.Entry<Integer, Double> entry : map1.entrySet()) {
			int w1 = entry.getKey();
			double prob1 = entry.getValue();
			double prob2 = Constant.SMOOTH_PROBABILITY;
			if (map2.containsKey(w1)) {
				prob2 = map2.get(w1);
			}
			double avg = (prob1 + prob2) / 2;
			mid.put(w1, avg);
		}
		for (Map.Entry<Integer, Double> entry : map2.entrySet()) {
			int w2 = entry.getKey();
			double prob2 = entry.getValue();
			double prob1 = Constant.SMOOTH_PROBABILITY;
			if (map1.containsKey(w2)) {
				prob1 = map1.get(w2);
			}
			double avg = (prob1 + prob2) / 2;
			mid.put(w2, avg);
		}

		return 0.5 * getKLDivergence(map1, mid) + 0.5
				* getKLDivergence(map2, mid);
	}
}
