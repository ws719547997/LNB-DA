package feature;

import java.util.HashMap;
import java.util.Map;

import utility.ExceptionUtility;
import nlp.Document;
import nlp.Documents;
import nlp.EntropyHelper;

/**
 * Note that information gain can only be calculated using training data as the
 * labels of testing data are unknown.
 */
public class InformationGain {
	public Map<String, Double> mpFeatureStrToInformationGain = null; // IG.
	public Map<String, Double> mpFeatureStrToPositiveGivenFeature = null; // P(+|f).
	public Map<String, Double> mpFeatureStrToFeatureGivenPositive = null; // P(f|+).
	public Map<String, Double> mpFeatureStrToFeatureGivenNegative = null; // P(f|-).

	// For Chi square test.
	// freq(+, f).
	public Map<String, Integer> mpFeatureStrToPositiveWithFeatureCount = null;
	// freq(-, f).
	public Map<String, Integer> mpFeatureStrToNegativeWithFeatureCount = null;
	// freq(+, ~f).
	public Map<String, Integer> mpFeatureStrToPositiveWithoutFeatureCount = null;
	// freq(-, ~f).
	public Map<String, Integer> mpFeatureStrToNegativeWithoutFeatureCount = null;

	public InformationGain(Documents trainingDocs) {
		mpFeatureStrToInformationGain = new HashMap<String, Double>();
		mpFeatureStrToPositiveGivenFeature = new HashMap<String, Double>();
		mpFeatureStrToFeatureGivenPositive = new HashMap<String, Double>();
		mpFeatureStrToFeatureGivenNegative = new HashMap<String, Double>();

		mpFeatureStrToPositiveWithFeatureCount = new HashMap<String, Integer>();
		mpFeatureStrToNegativeWithFeatureCount = new HashMap<String, Integer>();
		mpFeatureStrToPositiveWithoutFeatureCount = new HashMap<String, Integer>();
		mpFeatureStrToNegativeWithoutFeatureCount = new HashMap<String, Integer>();

		int D = trainingDocs.size();
		Map<String, Integer> mpFeatureStrToPositiveCount = new HashMap<String, Integer>();
		Map<String, Integer> mpFeatureStrToNegativeCount = new HashMap<String, Integer>();
		int positiveCount = 0;
		int negativeCount = 0;
		for (Document document : trainingDocs) {
			if (document.isPositive()) {
				++positiveCount;
			} else {
				++negativeCount;
			}

			// Use the features for SVM since each feature only appears once in
			// one document.
			for (Feature feature : document.featuresForSVM) {
				String featureStr = feature.featureStr;
				if (!mpFeatureStrToPositiveCount.containsKey(featureStr)) {
					mpFeatureStrToPositiveCount.put(featureStr, 0);
				}
				if (!mpFeatureStrToNegativeCount.containsKey(featureStr)) {
					mpFeatureStrToNegativeCount.put(featureStr, 0);
				}
				if (document.isPositive()) {
					// Positive.
					// ++positiveCount;
					mpFeatureStrToPositiveCount.put(featureStr,
							mpFeatureStrToPositiveCount.get(featureStr) + 1);
				} else {
					// Negative.
					// ++negativeCount;
					mpFeatureStrToNegativeCount.put(featureStr,
							mpFeatureStrToNegativeCount.get(featureStr) + 1);
				}
			}
		}
		double entropyD = EntropyHelper.getEntropy(new int[] { positiveCount,
				negativeCount });
		// Calculate the information gain for each feature.
		for (Map.Entry<String, Integer> entry : mpFeatureStrToPositiveCount
				.entrySet()) {
			String featureStr = entry.getKey();

			// Compute information gain.
			double informationGain = entropyD;
			// When the feature is present.
			// freq(f) = freq(+, f) + freq(-, f).
			int positiveWithFeatureCount = entry.getValue();
			int negativeWithFeatureCount = mpFeatureStrToNegativeCount
					.get(featureStr);
			int featureCount = positiveWithFeatureCount
					+ negativeWithFeatureCount;
			// p(f) = freq(f) / D.
			double probOfFeature = 1.0 * featureCount / D;
			informationGain -= probOfFeature
					* EntropyHelper
							.getEntropy(new int[] { positiveWithFeatureCount,
									negativeWithFeatureCount });
			// When the feature is not present.
			// freq(+, ~f) = freq(+) - freq(+, f).
			// freq(-, ~f) = freq(-) - freq(-, f).
			int positiveWithoutFeatureCount = positiveCount
					- positiveWithFeatureCount;
			int negativeWithoutFeatureCount = negativeCount
					- negativeWithFeatureCount;
			// P(~f) = 1 - P(f).
			double probOfWithoutFeature = 1.0 - probOfFeature;
			informationGain -= probOfWithoutFeature
					* EntropyHelper.getEntropy(new int[] {
							positiveWithoutFeatureCount,
							negativeWithoutFeatureCount });
			mpFeatureStrToInformationGain.put(featureStr, informationGain);
			mpFeatureStrToPositiveGivenFeature.put(featureStr, 1.0
					* positiveWithFeatureCount / featureCount);

			// P(f|+).
			double probOfFeatureGivenPositive = 1.0 * positiveWithFeatureCount
					/ positiveCount;
			mpFeatureStrToFeatureGivenPositive.put(featureStr,
					probOfFeatureGivenPositive);
			// P(f|-).
			double probOfFeatureGivenNegative = 1.0 * negativeWithFeatureCount
					/ negativeCount;
			mpFeatureStrToFeatureGivenNegative.put(featureStr,
					probOfFeatureGivenNegative);

			// freq(+, f).
			mpFeatureStrToPositiveWithFeatureCount.put(featureStr,
					positiveWithFeatureCount);
			// freq(-, f).
			mpFeatureStrToNegativeWithFeatureCount.put(featureStr,
					negativeWithFeatureCount);
			// freq(+, ~f).
			mpFeatureStrToPositiveWithoutFeatureCount.put(featureStr,
					positiveWithoutFeatureCount);
			// freq(-, ~f).
			mpFeatureStrToNegativeWithoutFeatureCount.put(featureStr,
					negativeWithoutFeatureCount);
		}
	}

	/**
	 * Get information gain.
	 */
	public double getIGGivenFeatureStr(String featureStr) {
		ExceptionUtility.assertAsException(
				mpFeatureStrToInformationGain.containsKey(featureStr),
				"The feature's information gain has not been computed!");
		return mpFeatureStrToInformationGain.get(featureStr);
	}

	/**
	 * Get P(+|f).
	 */
	public double getProbOfPositiveGivenFeatureStr(String featureStr) {
		ExceptionUtility.assertAsException(
				mpFeatureStrToPositiveGivenFeature.containsKey(featureStr),
				"The feature's information gain has not been computed!");
		return mpFeatureStrToPositiveGivenFeature.get(featureStr);
	}

	/**
	 * Get P(-|f).
	 */
	public double getProbOfNegativeGivenFeatureStr(String featureStr) {
		ExceptionUtility.assertAsException(
				mpFeatureStrToPositiveGivenFeature.containsKey(featureStr),
				"The feature's information gain has not been computed!");
		return 1.0 - mpFeatureStrToPositiveGivenFeature.get(featureStr);
	}

	/**
	 * Get P(f|+).
	 */
	public double getProbOfFeatureGivenPositive(String featureStr) {
		ExceptionUtility.assertAsException(
				mpFeatureStrToFeatureGivenPositive.containsKey(featureStr),
				"The feature's information gain has not been computed!");
		return mpFeatureStrToFeatureGivenPositive.get(featureStr);
	}

	/**
	 * Get P(f|-).
	 */
	public double getProbOfFeatureGivenNegative(String featureStr) {
		ExceptionUtility.assertAsException(
				mpFeatureStrToFeatureGivenNegative.containsKey(featureStr),
				"The feature's information gain has not been computed!");
		return mpFeatureStrToFeatureGivenNegative.get(featureStr);
	}
}
