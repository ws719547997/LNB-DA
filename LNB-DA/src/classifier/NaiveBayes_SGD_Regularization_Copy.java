package classifier;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import classificationevaluation.ClassificationEvaluation;
import feature.Feature;
import feature.FeatureIndexer;
import feature.FeatureSelection;
import feature.Features;
import main.Constant;
import nlp.Document;
import nlp.Documents;
import utility.ExceptionUtility;
import utility.FileOneByOneLineWriter;
import utility.ItemWithValue;

/**
 * Implement the new model using stochastic gradient descent.
 * 
 * No feature selection.
 */
public class NaiveBayes_SGD_Regularization_Copy extends BaseClassifier {
	// Not as good as implementation as Lingpipe.
	private final String[] mCategories = { "+1", "-1" }; // The array of
															// categories.
	private final double mCategoryPrior = 0.5; // The prior probability of
												// category, used in smoothing.

	private FeatureIndexer featureIndexerAllDomains = null;
	private FeatureIndexer featureIndexerTargetDomain = null;
	List<Instance> data = null;

	// Count(+) and Count(-).
	public double[] classInstanceCount = null;
	// x[v][0]: the count for word v in positive class.
	// x[v][1]: the count for word v in negative class.
	private double[][] x = null;
	// sum_x[c] = sum_i{x[v][c]}.
	private double[] sum_x = null;
	private int V = 0;

	private Map<String, Double> mpFeatureStrToRegRatio = null;
	private Map<String, Double> mpFeatureStrToRegCoef = null;
	private double learningRate = 0.0;

	public NaiveBayes_SGD_Regularization_Copy(ClassifierParameters param2,
			FeatureSelection featureSelection2,
			ClassificationKnowledge knowledge2,
			Map<String, Double> mpFeatureStrToRegRatio2,
			Map<String, Double> mpFeatureStrToRegCoef2) {
		param = param2;
		featureSelection = featureSelection2;
		knowledge = knowledge2;
		mpFeatureStrToRegRatio = mpFeatureStrToRegRatio2;
		mpFeatureStrToRegCoef = mpFeatureStrToRegCoef2;

		featureIndexerAllDomains = new FeatureIndexer();
		featureIndexerTargetDomain = new FeatureIndexer();
	}

	/***************** Stochastic Gradient Descent *****************/
	private void SGDEntry() {
		// Shuffle the data.
		Collections.shuffle(data, new Random(Constant.RANDOMSEED));
		double before = getObjectiveFunctionValueForInstancesOriginal(data);
		int iter = 0;
		while (true) {
			learningRate = param.learningRate
					/ (1 + param.learningRateChange * param.learningRate
							* param.regCoefficientAlpha * iter);
			// System.out.println("Iteration " + (iter++) + " : " + learningRate
			// + " " + before);
			for (int d = 0; d < data.size(); ++d) {
				Instance instance = data.get(d);
				if (instance.y > 0) {
					SGDForPositiveInstance(instance);
				} else {
					SGDForNegativeInstance(instance);
				}
			}
			double after = getObjectiveFunctionValueForInstancesOriginal(data);
			if (Math.abs(after - before) <= param.convergenceDifference) {
				break;
			}
			before = after;
		}
	}

	private void SGDForPositiveInstance(Instance instance) {
		// Compute the values related with g(x) function.
		int lengthOfDocument = 0; // |d|.
		for (int frequency : instance.mpFeatureToFrequency.values()) {
			lengthOfDocument += frequency;
		}

		double positiveClassSum = (param.smoothingPriorForFeatureInNaiveBayes
				* V + sum_x[0]);
		double negativeClassSum = (param.smoothingPriorForFeatureInNaiveBayes
				* V + sum_x[1]);
		double ratioOfClasses = positiveClassSum / negativeClassSum;

		// Compute R.
		double R = getProbOfClass(1) / getProbOfClass(0); // Pr(-) / Pr(+).
		for (Map.Entry<Integer, Integer> entry : instance.mpFeatureToFrequency
				.entrySet()) {
			int featureId = entry.getKey();
			int frequency = entry.getValue();
			for (int i = 0; i < frequency; ++i) {
				// (lambda + x_-k) / (lambda + x_+k).
				R *= ratioOfClasses
						* (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][1])
						/ (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][0]);
			}
		}

		// Compute the gradient of each feature for the positive class.
		Map<Integer, Double> gradientOfFeatureWithPositiveClass = new HashMap<Integer, Double>();
		for (Map.Entry<Integer, Integer> entry : instance.mpFeatureToFrequency
				.entrySet()) {
			int featureId = entry.getKey();
			int frequency = entry.getValue();
			if (featureIndexerAllDomains.getFeatureStrGivenFeatureId(featureId)
					.equals("toy")) {
				// System.out.println("toy");
			}
			double part1 = frequency
					/ (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][0])
					/ (1.0 + R);
			double part2 = R / (1.0 + R) * lengthOfDocument / positiveClassSum;
			double gradient = 0.0;
			if (Double.isInfinite(R)) {
				gradient = lengthOfDocument
						/ positiveClassSum
						- frequency
						/ (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][0]);
			} else {
				gradient = part1
						+ part2
						- frequency
						/ (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][0]);
			}
			if (Double.isNaN(gradient) || Double.isInfinite(gradient)) {
				System.out.println("Nan");
			}
			if (Double.isInfinite(-gradient)) {
				System.out.println("Infinity");
			}
			// Regularization.
			gradient += getRegularizationTermGradient(featureId, 0);
			gradientOfFeatureWithPositiveClass.put(featureId, gradient);
		}

		// Compute the gradient of each feature for the negative class.
		Map<Integer, Double> gradientOfFeatureWithNegativeClass = new HashMap<Integer, Double>();
		for (Map.Entry<Integer, Integer> entry : instance.mpFeatureToFrequency
				.entrySet()) {
			int featureId = entry.getKey();
			int frequency = entry.getValue();
			if (featureIndexerAllDomains.getFeatureStrGivenFeatureId(featureId)
					.equals("toy")) {
				// System.out.println("toy");
			}

			double numerator = frequency
					/ (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][1])
					- lengthOfDocument / negativeClassSum;
			double denominator = 0.0;
			if (Double.isInfinite(R)) {
				denominator = 1.0;
			} else {
				denominator = 1.0 / R + 1;
			}
			double gradient = 0.0;
			if (Double.isInfinite(denominator)) {
				gradient = 0.0;
			} else {
				gradient = numerator / denominator;
			}
			if (Double.isNaN(gradient) || Double.isInfinite(gradient)) {
				System.out.println("Nan");
			}
			// Regularization.
			gradient += getRegularizationTermGradient(featureId, 1);
			if (Double.isInfinite(-gradient)) {
				System.out.println("Infinity");
				getRegularizationTermGradient(featureId, 1);
			}
			gradientOfFeatureWithNegativeClass.put(featureId, gradient);
		}

		// Note that we need to compute all of the gradients first and then
		// update
		// the counts.
		// Update the count of each feature in this document for positive class.
		updateXs(instance, gradientOfFeatureWithPositiveClass,
				gradientOfFeatureWithNegativeClass);
	}

	private void SGDForNegativeInstance(Instance instance) {
		// Compute the values related with g(x) function.
		int lengthOfDocument = 0;
		for (int frequency : instance.mpFeatureToFrequency.values()) {
			lengthOfDocument += frequency;
		}

		double positiveClassSum = (param.smoothingPriorForFeatureInNaiveBayes
				* V + sum_x[0]);
		double negativeClassSum = (param.smoothingPriorForFeatureInNaiveBayes
				* V + sum_x[1]);
		double ratioOfClasses = positiveClassSum / negativeClassSum;

		// Compute R.
		double R = getProbOfClass(1) / getProbOfClass(0); // Pr(-) / Pr(+).
		for (Map.Entry<Integer, Integer> entry : instance.mpFeatureToFrequency
				.entrySet()) {
			int featureId = entry.getKey();
			int frequency = entry.getValue();
			for (int i = 0; i < frequency; ++i) {
				// (lambda + x_-k) / (lambda + x_+k).
				R *= ratioOfClasses
						* (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][1])
						/ (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][0]);
			}
		}

		// Compute the gradient of each feature for the positive class.
		Map<Integer, Double> gradientOfFeatureWithPositiveClass = new HashMap<Integer, Double>();
		for (Map.Entry<Integer, Integer> entry : instance.mpFeatureToFrequency
				.entrySet()) {
			int featureId = entry.getKey();
			int frequency = entry.getValue();
			if (featureIndexerAllDomains.getFeatureStrGivenFeatureId(featureId)
					.equals("toy")) {
				// System.out.println("toy");
			}

			double part1 = frequency
					/ (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][0])
					/ (1.0 + R);
			double part2 = R / (1.0 + R) * lengthOfDocument / positiveClassSum;
			double gradient = 0.0;
			if (Double.isInfinite(R)) {
				gradient = lengthOfDocument / positiveClassSum
						- lengthOfDocument / positiveClassSum;
			} else {
				gradient = part1 + part2 - lengthOfDocument / positiveClassSum;
			}
			// verifyGradient(instance, featureId, gradient, 0);
			// verifyGradient(instance, featureId, gradient, 0);
			if (Double.isNaN(gradient) || Double.isInfinite(gradient)) {
				System.out.println("Nan");
			}
			// Regularization.
			gradient += getRegularizationTermGradient(featureId, 0);
			if (Double.isInfinite(-gradient)) {
				System.out.println("Infinity");
			}
			gradientOfFeatureWithPositiveClass.put(featureId, gradient);
		}

		// Compute the gradient of each feature for the negative class.
		Map<Integer, Double> gradientOfFeatureWithNegativeClass = new HashMap<Integer, Double>();
		for (Map.Entry<Integer, Integer> entry : instance.mpFeatureToFrequency
				.entrySet()) {
			int featureId = entry.getKey();
			int frequency = entry.getValue();
			if (featureIndexerAllDomains.getFeatureStrGivenFeatureId(featureId)
					.equals("toy")) {
				// System.out.println("toy");
			}

			double numerator = frequency
					/ (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][1])
					- lengthOfDocument / negativeClassSum;
			double denominator = 0.0;
			if (Double.isInfinite(R)) {
				denominator = 1.0;
			} else {
				denominator = 1.0 / R + 1;
			}
			double gradient = 0.0;
			if (Double.isInfinite(denominator)) {
				gradient = -numerator;
			} else {
				gradient = numerator / denominator - numerator;
			}
			if (Double.isNaN(gradient)) {
				System.out.println("Nan");
			}
			// Regularization.
			gradient += getRegularizationTermGradient(featureId, 1);
			if (Double.isInfinite(-gradient)) {
				System.out.println("Infinity");
			}
			gradientOfFeatureWithNegativeClass.put(featureId, gradient);
		}

		// Note that we need to compute all of the gradients first and then
		// update the counts.
		// Update the count of each feature in this document for positive class.
		updateXs(instance, gradientOfFeatureWithPositiveClass,
				gradientOfFeatureWithNegativeClass);
	}

	/**
	 * Regularization.
	 */
	private double getRegularizationTermGradient(int featureId, int classIndex) {
		if (mpFeatureStrToRegRatio == null) {
			return 0.0;
		}
		// Add L2 regularization.
		double regularizationGradient = 0;
		String featureStr = featureIndexerAllDomains
				.getFeatureStrGivenFeatureId(featureId);
		if (!mpFeatureStrToRegRatio.containsKey(featureStr)) {
			return 0.0;
		}
		double regRatio = mpFeatureStrToRegRatio.get(featureStr);
		double regCoef = mpFeatureStrToRegCoef.get(featureStr);
		if (classIndex == 0
				&& param.smoothingPriorForFeatureInNaiveBayes + x[featureId][0] >= param.smoothingPriorForFeatureInNaiveBayes
						+ x[featureId][1]) {
			// Positive side.
			double regPart1 = (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][0])
					/ (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][1])
					* (V * param.smoothingPriorForFeatureInNaiveBayes + sum_x[1])
					/ (V * param.smoothingPriorForFeatureInNaiveBayes + sum_x[0])

					- regRatio;
			double regPart2 = (V * param.smoothingPriorForFeatureInNaiveBayes + sum_x[1])
					/ (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][1]);
			double regPart3 = 1.0
					/ (V * param.smoothingPriorForFeatureInNaiveBayes + sum_x[0])

					- (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][0])
					/ (V * param.smoothingPriorForFeatureInNaiveBayes + sum_x[0])
					/ (V * param.smoothingPriorForFeatureInNaiveBayes + sum_x[0]);
			regularizationGradient = regCoef * regPart1 * regPart2 * regPart3;
			// System.out.println("+: " + regPart1 + " " +
			// regularizationGradient);
		} else if (classIndex == 1
				&& param.smoothingPriorForFeatureInNaiveBayes + x[featureId][0] <= param.smoothingPriorForFeatureInNaiveBayes
						+ x[featureId][1]) {
			// Negative side.
			double regPart1 = (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][0])
					/ (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][1])
					* (V * param.smoothingPriorForFeatureInNaiveBayes + sum_x[1])
					/ (V * param.smoothingPriorForFeatureInNaiveBayes + sum_x[0])

					- regRatio;
			double regPart2 = (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][0])
					/ (V * param.smoothingPriorForFeatureInNaiveBayes + sum_x[0]);
			double regPart3 = 1.0
					/ (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][1])

					- (V * param.smoothingPriorForFeatureInNaiveBayes + sum_x[1])
					/ (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][1])
					/ (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][1]);
			regularizationGradient = regCoef * regPart1 * regPart2 * regPart3;
			// System.out.println("-: " + regPart1 + " " +
			// regularizationGradient);
		}
		return regularizationGradient;
	}

	/***************** Components related with function g(x) *****************/
	/**
	 * g(x) = \Big(\frac{{}\lambda |V|+\sum\nolimits_{v}{x_{+,v}}}{{}\lambda
	 * |V|+\sum\nolimits_{v}{x_{-,v}}}\Big)^{|d_i|}
	 */
	// private double getGFunctionValue(int lengthOfDocument) {
	// double numerator = param.smoothingPriorForFeatureInNaiveBayes * V +
	// sum_x[0];
	// double denominator = param.smoothingPriorForFeatureInNaiveBayes * V +
	// sum_x[1];
	// double ratio = numerator / denominator;
	// return Math.pow(ratio, 1.0 * lengthOfDocument);
	// }

	/***************** Update Xs in SGD *****************/
	public void updateXs(Instance instance, Map<Integer, Double> deltaPositive,
			Map<Integer, Double> deltaNegative) {

		if (param.gradientVerificationUsingFiniteDifferences) {
			verifyGradient(instance, deltaPositive, 0);
			verifyGradient(instance, deltaNegative, 1);
		}

		for (Map.Entry<Integer, Integer> entry : instance.mpFeatureToFrequency
				.entrySet()) {
			int featureId = entry.getKey();
			if (featureId == 159) {
				// System.out.println("159");
			}

			// Here, we enforce the x to be non negative.
			if (x[featureId][0] - learningRate * deltaPositive.get(featureId) < 0) {
				double delta = x[featureId][0];
				x[featureId][0] -= delta;
				sum_x[0] -= delta;
			} else {
				x[featureId][0] -= learningRate * deltaPositive.get(featureId);
				sum_x[0] -= learningRate * deltaPositive.get(featureId);
			}

			if (x[featureId][1] - learningRate * deltaNegative.get(featureId) < 0) {
				if (featureId == 159) {
					// System.out.println("159");
				}
				double delta = x[featureId][1];
				x[featureId][1] -= delta;
				sum_x[1] -= delta;
				if (Double.isInfinite(sum_x[1])) {
					System.out.println("Infinity");
				}
			} else {
				x[featureId][1] -= learningRate * deltaNegative.get(featureId);
				sum_x[1] -= learningRate * deltaNegative.get(featureId);
				if (Double.isInfinite(sum_x[1])) {
					System.out.println("Infinity");
				}
			}
		}
	}

	/***************** Verify SGD using finite difference *****************/
	private void verifyGradient(Instance instance,
			Map<Integer, Double> mpFeatureIdToGradient, int classIndex) {
		double before = getObjectiveFunctionValueForSingleInstanceLog(instance);
		int dividedBy = 1;
		// Divide it into a value that is smaller than 1 for digit matching.
		while (before >= 1) {
			before /= 10;
			dividedBy *= 10;
		}
		if (before == Double.MIN_VALUE) {
			// Cannot verify due to the numerical issues.
			return;
		}
		for (int featureId : mpFeatureIdToGradient.keySet()) {
			double delta = param.gradientVerificationDelta;
			x[featureId][classIndex] += delta;
			sum_x[classIndex] += delta;
			double after = getObjectiveFunctionValueForSingleInstanceLog(instance);
			after /= dividedBy;
			double gradientCalculated = mpFeatureIdToGradient.get(featureId);
			if (!(Math.abs(before + delta * gradientCalculated - after) < Constant.SMALL_THAN_AS_ZERO)) {
				if (instance.y == -1 && classIndex == 1) {
					System.out.println("Document label " + instance.y);
					System.out.println("Class " + classIndex);
					System.out.println("Before : "
							+ (before + delta * gradientCalculated));
					System.out.println("After : " + after);
					System.out.println("Calculated Gradient: "
							+ gradientCalculated);
					System.out.println("Expected Gradient: " + (after - before)
							/ delta);
					System.out.println();
					getObjectiveFunctionValueForSingleInstanceLog(instance);
				}
			}
			// ExceptionUtility
			// .assertAsException(
			// Math.abs(before + delta * gradientCalculated
			// - after) < Constant.SMALL_THAN_AS_ZERO,
			// param.domain + " Gradient verification fails!");
			// Revert the actions.
			x[featureId][classIndex] -= delta;
			sum_x[classIndex] -= delta;
		}
	}

	/***************** Verify SGD using finite difference *****************/
	// private void verifyGradient(Instance instance, int featureId,
	// double gradientCalculated, int classIndex) {
	// double before = getObjectiveFunctionValueForSingleInstanceLog(instance);
	// int dividedBy = 1;
	// // Divide it into a value that is smaller than 1 for digit matching.
	// while (before >= 1) {
	// before /= 10;
	// dividedBy *= 10;
	// }
	// if (before == Double.MIN_VALUE) {
	// // Cannot verify due to the numerical issues.
	// return;
	// }
	// double delta = param.gradientVerificationDelta;
	// x[featureId][classIndex] += delta;
	// sum_x[classIndex] += delta;
	// double after = getObjectiveFunctionValueForSingleInstanceLog(instance);
	// after /= dividedBy;
	// if (!(Math.abs(before + delta * gradientCalculated - after) <
	// Constant.SMALL_THAN_AS_ZERO)) {
	// if (instance.y == -1 && classIndex == 1) {
	// System.out.println("Document label " + instance.y);
	// System.out.println("Class " + classIndex);
	// System.out.println("Before : "
	// + (before + delta * gradientCalculated));
	// System.out.println("After : " + after);
	// System.out
	// .println("Calculated Gradient: " + gradientCalculated);
	// System.out.println("Expected Gradient: " + (after - before)
	// / delta);
	// System.out.println();
	// getObjectiveFunctionValueForSingleInstanceLog(instance);
	// }
	// }
	// // ExceptionUtility
	// // .assertAsException(
	// // Math.abs(before + delta * gradientCalculated
	// // - after) < Constant.SMALL_THAN_AS_ZERO,
	// // param.domain + " Gradient verification fails!");
	// // Revert the actions.
	// x[featureId][classIndex] -= delta;
	// sum_x[classIndex] -= delta;
	// }

	/***************** Objective function *****************/
	/**
	 * Objective function in the log form.
	 */
	public double getObjectiveFunctionValueForInstancesLog(List<Instance> data) {
		double sum = 0.0;
		for (Instance instance : data) {
			sum += getObjectiveFunctionValueForSingleInstanceLog(instance);
		}
		return sum / data.size();
	}

	/**
	 * Objective function for single instance in the log form.
	 */
	private double getObjectiveFunctionValueForSingleInstanceLog(
			Instance instance) {
		double ob = 0.0;
		if (instance.y > 0) {
			ob = getObjectiveFunctionValueForSinglePositiveInstanceLog(instance);
		} else {
			ob = getObjectiveFunctionValueForSingleNegativeInstanceLog(instance);
		}

		double regularizationTerm = 0.0;
		if (mpFeatureStrToRegRatio != null) {
			// Regularization.
			for (Map.Entry<String, Double> entry : mpFeatureStrToRegRatio
					.entrySet()) {
				String featureStr = entry.getKey();
				int featureId = featureIndexerAllDomains
						.getFeatureIdGivenFeatureStr(featureStr);
				double regRatio = entry.getValue();
				double regCoef = mpFeatureStrToRegCoef.get(featureStr);
				double regPart1 = (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][0])
						/ (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][1])
						* (V * param.smoothingPriorForFeatureInNaiveBayes + sum_x[1])
						/ (V * param.smoothingPriorForFeatureInNaiveBayes + sum_x[0])
						- regRatio;
				regularizationTerm += 0.5 * regCoef * regPart1 * regPart1;
			}
		}

		return ob + regularizationTerm;
	}

	/**
	 * Objective function for single positive instance in the log form.
	 */
	private double getObjectiveFunctionValueForSinglePositiveInstanceLog(
			Instance instance) {
		double positiveClassSum = (param.smoothingPriorForFeatureInNaiveBayes
				* V + sum_x[0]);
		double negativeClassSum = (param.smoothingPriorForFeatureInNaiveBayes
				* V + sum_x[1]);
		double ratioOfClasses = positiveClassSum / negativeClassSum;

		// Compute R.
		double R = getProbOfClass(1) / getProbOfClass(0); // Pr(-) / Pr(+).
		for (Map.Entry<Integer, Integer> entry : instance.mpFeatureToFrequency
				.entrySet()) {
			int featureId = entry.getKey();
			int frequency = entry.getValue();
			for (int i = 0; i < frequency; ++i) {
				// (lambda + x_-k) / (lambda + x_+k).
				R *= ratioOfClasses
						* (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][1])
						/ (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][0]);
			}
		}
		if (Double.isInfinite(R)) {
			return Double.MIN_VALUE; // Invalid value.
		}
		if (R == 0) {
			return Double.MIN_VALUE; // Invalid value.
		}
		double value = 1.0 / 2 + R / 2.0; // (1 + R) / 2.
		if (Double.isInfinite(value)) {
			return Double.MIN_VALUE;
		}

		return Math.log(value);
	}

	/**
	 * Objective function for single negative instance in the log form.
	 */
	private double getObjectiveFunctionValueForSingleNegativeInstanceLog(
			Instance instance) {
		double positiveClassSum = (param.smoothingPriorForFeatureInNaiveBayes
				* V + sum_x[0]);
		double negativeClassSum = (param.smoothingPriorForFeatureInNaiveBayes
				* V + sum_x[1]);
		double ratioOfClasses = positiveClassSum / negativeClassSum;

		// Compute R.
		double R = getProbOfClass(1) / getProbOfClass(0); // Pr(-) / Pr(+).
		for (Map.Entry<Integer, Integer> entry : instance.mpFeatureToFrequency
				.entrySet()) {
			int featureId = entry.getKey();
			int frequency = entry.getValue();
			for (int i = 0; i < frequency; ++i) {
				// (lambda + x_-k) / (lambda + x_+k).
				R *= ratioOfClasses
						* (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][1])
						/ (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][0]);
			}
		}
		if (Double.isInfinite(R)) {
			return Math.log(0.5);
		}
		if (R == 0) {
			return Double.MIN_VALUE; // Invalid value.
		}
		double value = 1.0 / 2.0 / R + 1.0 / 2.0; // (1 + R) / (2 * R).
		if (Double.isInfinite(value)) {
			return Double.MIN_VALUE;
		}
		return Math.log(value);
	}

	/**
	 * sum_i {y * (Pr(+) - Pr(-))} for multiple instance.
	 */
	public double getObjectiveFunctionValueForInstancesOriginal(
			List<Instance> data) {
		double sum = 0.0;
		for (Instance instance : data) {
			sum += getObjectiveFunctionValueForSingleInstanceOriginal(instance);
		}
		double ob = sum / data.size();

		double regularizationTerm = 0.0;
		if (mpFeatureStrToRegRatio != null) {
			// Regularization.
			for (Map.Entry<String, Double> entry : mpFeatureStrToRegRatio
					.entrySet()) {
				String featureStr = entry.getKey();
				int featureId = featureIndexerAllDomains
						.getFeatureIdGivenFeatureStr(featureStr);
				double regRatio = entry.getValue();
				double regCoef = mpFeatureStrToRegCoef.get(featureStr);
				double regPart1 = (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][0])
						/ (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][1])
						* (V * param.smoothingPriorForFeatureInNaiveBayes + sum_x[1])
						/ (V * param.smoothingPriorForFeatureInNaiveBayes + sum_x[0])
						- regRatio;
				regularizationTerm += 0.5 * regCoef * regPart1 * regPart1;
			}
		}

		return ob + regularizationTerm;
	}

	/**
	 * y * (Pr(+) - Pr(-)) for single instance.
	 */
	public double getObjectiveFunctionValueForSingleInstanceOriginal(
			Instance instance) {
		double[] logps = new double[2];
		for (Map.Entry<Integer, Integer> entry : instance.mpFeatureToFrequency
				.entrySet()) {
			int featureId = entry.getKey();
			int frequency = entry.getValue();

			for (int i = 0; i < frequency; ++i) {
				double[] tokenCounts = x[featureId];
				if (tokenCounts == null) {
					continue;
				}
				for (int catIndex = 0; catIndex < mCategories.length; ++catIndex) {
					logps[catIndex] += com.aliasi.util.Math
							.log2(probTokenByIndexArray(catIndex, tokenCounts));
				}
			}
		}
		for (int catIndex = 0; catIndex < 2; ++catIndex) {
			logps[catIndex] += com.aliasi.util.Math
					.log2(getProbOfClass(catIndex));
		}
		double[] probs = logJointToConditional(logps);
		return instance.y * (probs[0] - probs[1]);
	}

	/***************** Train and test *****************/
	/**
	 * This only works for binary classification for now.
	 */
	@Override
	public void train(Documents trainingDocs) {
		// Convert the training documents to data.
		data = convertDocumentsToData(trainingDocs);
		// Also add the words from the source domains into the featureIndexer.
		for (String featureStr : knowledge.wordCountInPerClass
				.keySet()) {
			if (featureSelection != null
					&& !featureSelection.isFeatureSelected(featureStr)) {
				// The feature is not selected.
				continue;
			}
			featureIndexerAllDomains
					.addFeatureStrIfNotExistStartingFrom0(featureStr);
		}

		// Initialize array from data.
		classInstanceCount = new double[2];
		classInstanceCount[0] = trainingDocs.getNoOfPositiveLabels();
		classInstanceCount[1] = trainingDocs.getNoOfNegativeLabels();
		// Initialize array from knowledge.
		V = featureIndexerAllDomains.size();
		x = new double[V][];
		for (int v = 0; v < V; ++v) {
			String featureStr = featureIndexerAllDomains
					.getFeatureStrGivenFeatureId(v);
			if (knowledge.wordCountInPerClass
					.containsKey(featureStr)) {
				x[v] = knowledge.wordCountInPerClass
						.get(featureStr);
			} else {
				// The word only appears in the target domain.
				x[v] = new double[] { 0.0, 0.0 };
			}
		}
		sum_x = knowledge.countTotalWordsInPerClass;

		if (param.convergenceDifference != Double.MAX_VALUE) {
			// Stochastic gradient descent.
			SGDEntry();
		}

		// Check if any value in x is nan or infinity.
		for (int i = 0; i < x.length; ++i) {
			ExceptionUtility
					.assertAsException(!Double.isNaN(x[i][0]), "Is Nan");
			ExceptionUtility
					.assertAsException(!Double.isNaN(x[i][1]), "Is Nan");
			ExceptionUtility.assertAsException(!Double.isInfinite(x[i][0]),
					"Is Infinite");
			ExceptionUtility.assertAsException(!Double.isInfinite(x[i][1]),
					"Is Infinite");
		}

		// Update classification knowledge.
		knowledge = new ClassificationKnowledge();
		// knowledge.countDocsInPerClass = mCaseCounts;
		// knowledge.wordCountInPerClass =
		// mFeatureStrToCountsMap;
		// knowledge.countTotalWordsInPerClass =
		// mTotalCountsPerCategory;
	}

	private List<Instance> convertDocumentsToData(Documents trainingDocs) {
		List<Instance> data = new ArrayList<Instance>();
		for (Document trainingDoc : trainingDocs) {
			Instance instance = new Instance();
			for (Feature feature : trainingDoc.featuresForNaiveBayes) {
				String featureStr = feature.featureStr;
				if (featureSelection != null
						&& !featureSelection.isFeatureSelected(featureStr)) {
					// The feature is not selected.
					continue;
				}
				int featureId = featureIndexerAllDomains
						.getFeatureIdOtherwiseAddFeatureStrStartingFrom0(featureStr);
				featureIndexerTargetDomain
						.addFeatureStrIfNotExistStartingFrom0(featureStr);
				if (!instance.mpFeatureToFrequency.containsKey(featureId)) {
					instance.mpFeatureToFrequency.put(featureId, 0);
				}
				instance.mpFeatureToFrequency.put(featureId,
						instance.mpFeatureToFrequency.get(featureId) + 1);
			}
			instance.y = Integer.parseInt(trainingDoc.label);
			data.add(instance);
		}
		return data;
	}

	@Override
	public ClassificationEvaluation test(Documents testingDocs) {
		String filepath = param.nbRootDirectory + param.domain + ".txt";
		FileOneByOneLineWriter writer = new FileOneByOneLineWriter(filepath);
		for (Document testingDoc : testingDocs) {
			testingDoc.predict = this.getBestClassByClassification(
					testingDoc.featuresForNaiveBayes, writer);
		}
		writer.close();
		ClassificationEvaluation evaluation = new ClassificationEvaluation(
				testingDocs.getLabels(), testingDocs.getPredicts(),
				param.domain);
		return evaluation;
	}

	/**
	 * Classify the content and return best category with highest probability.
	 */
	public String getBestClassByClassification(Features features,
			FileOneByOneLineWriter writer) {
		double[] categoryProb = getClassProbByClassification(features, writer);
		double maximumProb = -Double.MAX_VALUE;
		int maximumIndex = -1;
		for (int i = 0; i < categoryProb.length; ++i) {
			if (maximumProb < categoryProb[i]) {
				maximumProb = categoryProb[i];
				maximumIndex = i;
			}
		}
		return mCategories[maximumIndex];
	}

	/**
	 * Classify the content and return the probability of each category.
	 */
	public double[] getClassProbByClassification(Features features,
			FileOneByOneLineWriter writer) {
		double[] logps = new double[2];
		for (Feature feature : features) {
			String featureStr = feature.featureStr;
			if (!featureIndexerAllDomains.containsFeatureStr(featureStr)) {
				continue;
			}
			int featureId = featureIndexerAllDomains
					.getFeatureIdGivenFeatureStr(featureStr);
			double[] tokenCounts = x[featureId];
			if (tokenCounts == null) {
				continue;
			}
			for (int catIndex = 0; catIndex < mCategories.length; ++catIndex) {
				logps[catIndex] += com.aliasi.util.Math
						.log2(probTokenByIndexArray(catIndex, tokenCounts));
				writer.writeLine(featureStr + " : "
						+ probTokenByIndexArray(catIndex, tokenCounts));
			}
		}
		for (int catIndex = 0; catIndex < 2; ++catIndex) {
			logps[catIndex] += com.aliasi.util.Math
					.log2(getProbOfClass(catIndex));
		}
		return logJointToConditional(logps);
	}

	/**
	 * Pr(w|c) = (0.5 + count(w,c)) / (|V| * 0.5 + count(all of w, c)).
	 */
	private double probTokenByIndexArray(int catIndex, double[] tokenCounts) {
		double tokenCatCount = tokenCounts[catIndex];
		if (tokenCatCount < 0) {
			tokenCatCount = 0;
		}
		double totalCatCount = sum_x[catIndex];
		return (tokenCatCount + param.smoothingPriorForFeatureInNaiveBayes)
				/ (totalCatCount + featureIndexerAllDomains.size()
						* param.smoothingPriorForFeatureInNaiveBayes);
	}

	/**
	 * P(c) = (0.5 + count(c)) / (|C) * 0.5 + count(all of c)).
	 */
	private double getProbOfClass(int catIndex) {
		return (classInstanceCount[catIndex] + mCategoryPrior)
				/ (classInstanceCount[0] + classInstanceCount[1] + mCategories.length
						* mCategoryPrior);
	}

	/**
	 * Normalize the probability array. Copy from the class
	 * com.aliasi.classify.ConditionalClassification.
	 * 
	 * @param logJointProbs
	 * @return
	 */
	private double[] logJointToConditional(double[] logJointProbs) {
		for (int i = 0; i < logJointProbs.length; ++i) {
			if (logJointProbs[i] > 0.0 && logJointProbs[i] < 0.0000000001)
				logJointProbs[i] = 0.0;
			if (logJointProbs[i] > 0.0 || Double.isNaN(logJointProbs[i])) {
				StringBuilder sb = new StringBuilder();
				sb.append("Joint probs must be zero or negative."
						+ " Found log2JointProbs[" + i + "]="
						+ logJointProbs[i]);
				for (int k = 0; k < logJointProbs.length; ++k)
					sb.append("\nlogJointProbs[" + k + "]=" + logJointProbs[k]);
				throw new IllegalArgumentException(sb.toString());
			}
		}
		double max = com.aliasi.util.Math.maximum(logJointProbs);
		double[] probRatios = new double[logJointProbs.length];
		for (int i = 0; i < logJointProbs.length; ++i) {
			probRatios[i] = java.lang.Math.pow(2.0, logJointProbs[i] - max); // diff
																				// is
																				// <=
																				// 0.0
			if (probRatios[i] == Double.POSITIVE_INFINITY)
				probRatios[i] = Float.MAX_VALUE;
			else if (probRatios[i] == Double.NEGATIVE_INFINITY
					|| Double.isNaN(probRatios[i]))
				probRatios[i] = 0.0;
		}
		return com.aliasi.stats.Statistics.normalize(probRatios);
	}

	@Override
	public List<ItemWithValue> getFeaturesByRatio(Document testingDoc) {
		List<ItemWithValue> featuresWithRatios = new ArrayList<ItemWithValue>();
		for (Feature feature : testingDoc.featuresForNaiveBayes) {
			String featureStr = feature.featureStr;
			if (featureSelection != null
					&& !featureSelection.isFeatureSelected(featureStr)) {
				// The feature is not selected.
				continue;
			}
			int featureId = featureIndexerAllDomains
					.getFeatureIdGivenFeatureStr(featureStr);
			double[] tokenCounts = x[featureId];
			if (tokenCounts == null) {
				continue;
			}
			double[] ps = new double[2];
			for (int catIndex = 0; catIndex < mCategories.length; ++catIndex) {
				ps[catIndex] = probTokenByIndexArray(catIndex, tokenCounts);
			}
			ItemWithValue iwv = new ItemWithValue(featureStr, ps[0] / ps[1]);
			featuresWithRatios.add(iwv);
		}
		return featuresWithRatios;
	}

	@Override
	public void printMisclassifiedDocuments(Documents testingDocs, String misclassifiedDocumentsForOneCVFolderForOneDomainFilePath) {

	}

	@Override
	public double[] getCountsOfClasses(String featureStr) {
		int featureId = featureIndexerAllDomains
				.getFeatureIdGivenFeatureStr(featureStr);
		return x[featureId];
	}
}
