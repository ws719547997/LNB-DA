package classifier;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import classificationevaluation.ClassificationEvaluation;
import feature.Feature;
import feature.FeatureIndexer;
import feature.Features;
import main.Constant;
import nlp.Document;
import nlp.Documents;
import utility.ExceptionUtility;
import utility.FileOneByOneLineWriter;

/**
 * Implement the new model using stochastic gradient descent.
 * 
 * No feature selection.
 */
public class NaiveBayes_SGD_BigDecimal extends BaseClassifier {
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

	public NaiveBayes_SGD_BigDecimal(ClassifierParameters param2,
			ClassificationKnowledge knowledge2) {
		param = param2;
		knowledge = knowledge2;

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
			System.out.println("Iteration " + iter);
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
		// double gValue = getGFunctionValue(lengthOfDocument);
		// double gGradientWithPositiveClass =
		// getGFunctionGradientWithPositiveClass(lengthOfDocument);
		// double gGradientWithNegativeClass =
		// getGFunctionGradientWithNegativeClass(lengthOfDocument);

		double positiveClassSum = (param.smoothingPriorForFeatureInNaiveBayes
				* V + sum_x[0]);
		double negativeClassSum = (param.smoothingPriorForFeatureInNaiveBayes
				* V + sum_x[1]);
		double ratioOfClasses = positiveClassSum / negativeClassSum;

		// Compute R.
		BigDecimal R = BigDecimal.ONE;
		// Pr(-) / Pr(+).
		R = R.multiply(BigDecimal
				.valueOf(getProbOfClass(1) / getProbOfClass(0)));
		// ratioOfClasses ^ |d|.
		R = R.multiply(BigDecimal.valueOf(ratioOfClasses).pow(lengthOfDocument));
		for (Map.Entry<Integer, Integer> entry : instance.mpFeatureToFrequency
				.entrySet()) {
			int featureId = entry.getKey();
			int frequency = entry.getValue();
			double ratio = (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][1])
					/ (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][0]);
			R = R.multiply(BigDecimal.valueOf(ratio).pow(frequency));
		}

		// Compute the gradient of each feature for the positive class.
		Map<Integer, Double> gradientOfFeatureWithPositiveClass = new HashMap<Integer, Double>();
		for (Map.Entry<Integer, Integer> entry : instance.mpFeatureToFrequency
				.entrySet()) {
			int featureId = entry.getKey();
			int frequency = entry.getValue();

			BigDecimal numerator1 = BigDecimal
					.valueOf(frequency
							/ (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][0]));
			BigDecimal numerator2 = R.multiply(BigDecimal
					.valueOf(lengthOfDocument / positiveClassSum));
			BigDecimal numerator = numerator1.add(numerator2);
			BigDecimal denominator = R.add(BigDecimal.ONE);
			double gradient = numerator.divide(denominator, 30,
					RoundingMode.HALF_UP).doubleValue()
					- frequency
					/ (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][0]);
			if (Double.isNaN(gradient)) {
				System.out.println("Nan");
			}
			gradientOfFeatureWithPositiveClass.put(featureId, gradient);
		}

		// Compute the gradient of each feature for the negative class.
		Map<Integer, Double> gradientOfFeatureWithNegativeClass = new HashMap<Integer, Double>();
		for (Map.Entry<Integer, Integer> entry : instance.mpFeatureToFrequency
				.entrySet()) {
			int featureId = entry.getKey();
			int frequency = entry.getValue();

			BigDecimal numerator = BigDecimal
					.valueOf(frequency
							/ (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][1])
							- lengthOfDocument / negativeClassSum);
			BigDecimal denominator = BigDecimal.ONE.divide(R, 30,
					RoundingMode.HALF_UP).add(BigDecimal.ONE);
			double gradient = numerator.divide(denominator, 30,
					RoundingMode.HALF_UP).doubleValue();
			if (Double.isNaN(gradient)) {
				System.out.println("Nan");
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
		// double gValue = getGFunctionValue(lengthOfDocument);
		// double gGradientWithPositiveClass =
		// getGFunctionGradientWithPositiveClass(lengthOfDocument);
		// double gGradientWithNegativeClass =
		// getGFunctionGradientWithNegativeClass(lengthOfDocument);

		double positiveClassSum = (param.smoothingPriorForFeatureInNaiveBayes
				* V + sum_x[0]);
		double negativeClassSum = (param.smoothingPriorForFeatureInNaiveBayes
				* V + sum_x[1]);
		double ratioOfClasses = positiveClassSum / negativeClassSum;

		// Compute R.
		BigDecimal R = BigDecimal.ONE;
		// Pr(-) / Pr(+).
		R = R.multiply(BigDecimal
				.valueOf(getProbOfClass(1) / getProbOfClass(0)));
		// ratioOfClasses ^ |d|.
		R = R.multiply(BigDecimal.valueOf(ratioOfClasses).pow(lengthOfDocument));
		for (Map.Entry<Integer, Integer> entry : instance.mpFeatureToFrequency
				.entrySet()) {
			int featureId = entry.getKey();
			int frequency = entry.getValue();
			double ratio = (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][1])
					/ (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][0]);
			R = R.multiply(BigDecimal.valueOf(ratio).pow(frequency));
		}

		// Compute the gradient of each feature for the positive class.
		Map<Integer, Double> gradientOfFeatureWithPositiveClass = new HashMap<Integer, Double>();
		for (Map.Entry<Integer, Integer> entry : instance.mpFeatureToFrequency
				.entrySet()) {
			int featureId = entry.getKey();
			int frequency = entry.getValue();

			BigDecimal numerator1 = BigDecimal
					.valueOf(frequency
							/ (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][0]));
			BigDecimal numerator2 = R.multiply(BigDecimal
					.valueOf(lengthOfDocument / positiveClassSum));
			BigDecimal numerator = numerator1.add(numerator2);
			BigDecimal denominator = R.add(BigDecimal.ONE);
			double gradient = numerator.divide(denominator, 30,
					RoundingMode.HALF_UP).doubleValue()
					- lengthOfDocument / positiveClassSum;
			if (Double.isNaN(gradient)) {
				System.out.println("Nan");
			}
			gradientOfFeatureWithPositiveClass.put(featureId, gradient);
		}

		// Compute the gradient of each feature for the negative class.
		Map<Integer, Double> gradientOfFeatureWithNegativeClass = new HashMap<Integer, Double>();
		for (Map.Entry<Integer, Integer> entry : instance.mpFeatureToFrequency
				.entrySet()) {
			int featureId = entry.getKey();
			int frequency = entry.getValue();

			BigDecimal numerator = BigDecimal
					.valueOf(frequency
							/ (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][1])
							- lengthOfDocument / negativeClassSum);
			BigDecimal denominator = BigDecimal.ONE.divide(R, 30,
					RoundingMode.HALF_UP).add(BigDecimal.ONE);
			double gradient = numerator
					.divide(denominator, 30, RoundingMode.HALF_UP)
					.subtract(numerator).doubleValue();
			if (Double.isNaN(gradient)) {
				System.out.println("Nan");
			}
			gradientOfFeatureWithNegativeClass.put(featureId, gradient);
		}

		// Note that we need to compute all of the gradients first and then
		// update the counts.
		// Update the count of each feature in this document for positive class.
		updateXs(instance, gradientOfFeatureWithPositiveClass,
				gradientOfFeatureWithNegativeClass);
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
	//
	// private double getGFunctionGradientWithPositiveClass(int
	// lengthOfDocument) {
	// double numerator = param.smoothingPriorForFeatureInNaiveBayes * V +
	// sum_x[0];
	// double denominator = param.smoothingPriorForFeatureInNaiveBayes * V +
	// sum_x[1];
	// double ratio = numerator / denominator;
	// return lengthOfDocument * Math.pow(ratio, lengthOfDocument - 1.0)
	// / denominator;
	// }
	//
	// private double getGFunctionGradientWithNegativeClass(int
	// lengthOfDocument) {
	// double denominator = param.smoothingPriorForFeatureInNaiveBayes * V +
	// sum_x[1];
	// return (-1.0) * lengthOfDocument * getGFunctionValue(lengthOfDocument)
	// / denominator;
	// }

	/***************** Update Xs in SGD *****************/
	public void updateXs(Instance instance, Map<Integer, Double> deltaPositive,
			Map<Integer, Double> deltaNegative) {

		if (param.gradientVerificationUsingFiniteDifferences) {
			verifyGradient(instance, deltaPositive);
			verifyGradient(instance, deltaNegative);
		}

		for (Map.Entry<Integer, Integer> entry : instance.mpFeatureToFrequency
				.entrySet()) {
			int featureId = entry.getKey();

			// Here, we enforce the x to be non negative.
			if (x[featureId][0] - param.learningRate
					* deltaPositive.get(featureId) < 0) {
				double delta = x[featureId][0];
				x[featureId][0] -= delta;
				sum_x[0] -= delta;
			} else {
				x[featureId][0] -= param.learningRate
						* deltaPositive.get(featureId);
				sum_x[0] -= param.learningRate * deltaPositive.get(featureId);
			}

			if (x[featureId][1] - param.learningRate
					* deltaNegative.get(featureId) < 0) {
				double delta = x[featureId][1];
				x[featureId][1] -= delta;
				sum_x[1] -= delta;
			} else {
				x[featureId][1] -= param.learningRate
						* deltaNegative.get(featureId);
				sum_x[1] -= param.learningRate * deltaNegative.get(featureId);
			}
		}
	}

	/***************** Verify SGD using finite difference *****************/
	private void verifyGradient(Instance instance,
			Map<Integer, Double> mpFeatureIdToGradient) {
		double before = getObjectiveFunctionValueForSingleInstanceOriginal(instance);
		for (int featureId : mpFeatureIdToGradient.keySet()) {
			double delta = param.gradientVerificationDelta;
			x[featureId][0] += delta;
			sum_x[0] += delta;
			double after = getObjectiveFunctionValueForSingleInstanceOriginal(instance);
			double gradientCalculated = mpFeatureIdToGradient.get(featureId);
			if (Math.abs(before + delta * gradientCalculated - after) > Constant.SMALL_THAN_AS_ZERO) {
				System.out.println(before + delta * gradientCalculated);
				System.out.println(after);
			}
			ExceptionUtility
					.assertAsException(
							Math.abs(before + delta * gradientCalculated
									- after) < Constant.SMALL_THAN_AS_ZERO,
							"Gradient verification fails!");
			// Revert the actions.
			x[featureId][0] -= delta;
			sum_x[0] -= delta;
		}
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
		return sum / data.size();
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

	@Override
	public void printMisclassifiedDocuments(Documents testingDocs, String misclassifiedDocumentsForOneCVFolderForOneDomainFilePath) {

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
}

class Instance {
	Map<Integer, Integer> mpFeatureToFrequency = null;
	Map<Integer, Double> mpFeatureToFeatureValue = null;
	int y = 0;

	public Instance() {
		mpFeatureToFrequency = new HashMap<Integer, Integer>();
		mpFeatureToFeatureValue = new HashMap<Integer, Double>();
	}
}