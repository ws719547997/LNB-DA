package classifier;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import classificationevaluation.ClassificationEvaluation;
import feature.Feature;
import feature.FeatureSelection;
import feature.Features;
import nlp.Document;
import nlp.Documents;
import utility.ExceptionUtility;
import utility.FileOneByOneLineWriter;
import utility.ItemWithValue;

/**
 * Implement the naive Bayes, copying some from Lingpipe naive Bayes
 * implementation.
 */
public class NaiveBayes_Lifelong_WithoutCurrentDomainTraining extends
		BaseClassifier {
	// Not as good as implementation as Lingpipe.
	private final String[] mCategories = { "+1", "-1" }; // The array of
															// categories.
	private final double mCategoryPrior = 0.5; // The prior probability of
												// category, used in smoothing.

	private Map<String, double[]> mFeatureStrToCountsMap = new HashMap<String, double[]>(); // wordCount(w,c).
	private double[] mTotalCountsPerCategory = new double[2];; // SUM_w
																// wordCount(w,c)
																// indexed by c.
	private double[] mCaseCounts = new double[2]; // caseCount(c).
	private double mTotalCaseCount; // SUM_c caseCount(c).

	// /**
	// * Train the naive Bayes model. We assume that the word (feature) is
	// * separated by blank character (e.g., ' ').
	// */

	public NaiveBayes_Lifelong_WithoutCurrentDomainTraining() {
	}

	public NaiveBayes_Lifelong_WithoutCurrentDomainTraining(
			FeatureSelection featureSelection2,
			ClassificationKnowledge knowledge2, ClassifierParameters param2) {
		featureSelection = featureSelection2;
		knowledge = knowledge2;
		param = param2;
	}

	@Override
	public void train(Documents trainingDocs) {
		for (Document trainingDoc : trainingDocs) {
			this.train(trainingDoc.featuresForNaiveBayes, trainingDoc.label);
		}

		// Add knowledge of the words appear in the training data of the target
		// domain only.
		// for (String featureStr : mFeatureStrToCountsMap.keySet()) {
		// if (knowledge.wordCountInPerClass
		// .containsKey(featureStr)) {
		// double[] counts = mFeatureStrToCountsMap.get(featureStr);
		// double[] counts_knowledge =
		// knowledge.wordCountInPerClass
		// .get(featureStr);
		//
		// counts[0] += counts_knowledge[0];
		// mTotalCountsPerCategory[0] += counts_knowledge[0];
		// counts[1] += counts_knowledge[1];
		// mTotalCountsPerCategory[1] += counts_knowledge[1];
		// }
		// }

		// Add knowledge of all words in all domains. (Better performance).
		for (String featureStr : knowledge.wordCountInPerClass
				.keySet()) {
			double[] counts = mFeatureStrToCountsMap.get(featureStr);
			if (counts == null) {
				mFeatureStrToCountsMap.put(featureStr, new double[2]);
			}
			counts = mFeatureStrToCountsMap.get(featureStr);
			double[] counts_knowledge = knowledge.wordCountInPerClass
					.get(featureStr);

			counts[0] += counts_knowledge[0];
			mTotalCountsPerCategory[0] += counts_knowledge[0];
			counts[1] += counts_knowledge[1];
			mTotalCountsPerCategory[1] += counts_knowledge[1];
		}
	}

	@Override
	public ClassificationEvaluation test(Documents testingDocs) {
		String filepath = param.nbRootDirectory + param.domain + ".txt";
		FileOneByOneLineWriter writer = new FileOneByOneLineWriter(filepath);
		for (Document testingDoc : testingDocs) {
			testingDoc.predict = this.getBestCategoryByClassification(
					testingDoc.featuresForNaiveBayes, writer);
		}
		writer.close();
		ClassificationEvaluation evaluation = new ClassificationEvaluation(
				testingDocs.getLabels(), testingDocs.getPredicts(),
				param.domain);
		return evaluation;
	}

	public void train(Features features, String category) {
		// int catIndex = getIndex(category);
		// mCaseCounts[catIndex] += 1;
		// mTotalCaseCount += 1;

		for (Feature feature : features) {
			String featureStr = feature.featureStr;
			if (!featureSelection.isFeatureSelected(featureStr)) {
				// The feature is not selected.
				continue;
			}
			double[] tokenCounts = mFeatureStrToCountsMap.get(featureStr);
			if (tokenCounts == null) {
				tokenCounts = new double[mCategories.length];
				mFeatureStrToCountsMap.put(featureStr, tokenCounts);
			}
			// tokenCounts[catIndex] += 1;
			// mTotalCountsPerCategory[catIndex] += 1;
		}
	}

	/**
	 * Classify the content and return the probability of each category.
	 * 
	 * @param writer
	 */
	public double[] getCategoryProbByClassification(Features features,
			FileOneByOneLineWriter writer) {
		double[] logps = new double[2];
		for (Feature feature : features) {
			String featureStr = feature.featureStr;
			double[] tokenCounts = mFeatureStrToCountsMap.get(featureStr);
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
					.log2(probCatByIndex(catIndex));
		}
		return logJointToConditional(logps);
	}

	/**
	 * Classify the content and return best category with highest probability.
	 * 
	 * @param writer
	 */
	public String getBestCategoryByClassification(Features features,
			FileOneByOneLineWriter writer) {
		double[] categoryProb = getCategoryProbByClassification(features,
				writer);
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

	// /**
	// * Pr(w|c) = (0.5 + count(w,c)) / (|V| * 0.5 + count(all of w, c)).
	// */
	// private double probTokenByIndexArray(int catIndex, double[] tokenCounts)
	// {
	// double tokenCatCount = tokenCounts[catIndex];
	// double totalCatCount = mTotalCountsPerCategory[catIndex];
	// return (tokenCatCount + param.smoothingPriorForFeatureInNaiveBayes)
	// / (totalCatCount + mFeatureStrToCountsMap.size()
	// * param.smoothingPriorForFeatureInNaiveBayes);
	// }

	/**
	 * Pr(w|c) = (0.5 + count(w,c)) / (|V| * 0.5 + count(all of w, c)).
	 */
	private double probTokenByIndexArray(int catIndex, double[] tokenCounts) {
		double tokenCatCount = tokenCounts[catIndex];
		double totalCatCount = mTotalCountsPerCategory[catIndex];
		// double featureWithClassCount = 0;
		// if (knowledge.wordCountInPerClass
		// .containsKey(featureStr)) {
		// featureWithClassCount = knowledge.wordCountInPerClass
		// .get(featureStr)[catIndex];
		// }
		// return (tokenCatCount + param.smoothingPriorForFeatureInNaiveBayes +
		// featureWithClassCount)
		// / (totalCatCount + mFeatureStrToCountsMap.size()
		// * param.smoothingPriorForFeatureInNaiveBayes
		// + knowledge.wordCountInPerClass.size()
		// * param.smoothingPriorForFeatureInNaiveBayes +
		// knowledge.countTotalWordsInPerClass[catIndex]);
		return (tokenCatCount + param.smoothingPriorForFeatureInNaiveBayes)
				/ (totalCatCount + mFeatureStrToCountsMap.size()
						* param.smoothingPriorForFeatureInNaiveBayes);
	}

	/**
	 * P(c) = (0.5 + count(c)) / (|C) * 0.5 + count(all of c)).
	 */
	private double probCatByIndex(int catIndex) {
		double caseCountCat = mCaseCounts[catIndex];
		return (caseCountCat + mCategoryPrior)
				/ (mTotalCaseCount + mCategories.length * mCategoryPrior);
	}

	/**
	 * Get the index of the category in the array.
	 */
	private int getIndex(String category) {
		for (int i = 0; i < mCategories.length; ++i) {
			if (category.equals(mCategories[i])) {
				return i;
			}
		}
		ExceptionUtility.throwAndCatchException("No category is found!");
		return -1;
	}

	/**
	 * P(w|c): Returns the probability of the specified token in the specified
	 * category.
	 * 
	 * 
	 * @throws IllegalArgumentException
	 *             If the category is not known or the token is not known.
	 */
	public double probToken(String token, String cat) {
		int catIndex = getIndex(cat);
		double[] tokenCounts = mFeatureStrToCountsMap.get(token);
		if (tokenCounts == null) {
			String msg = "Requires known token." + " Found token=" + token;
			throw new IllegalArgumentException(msg);
		}
		return probTokenByIndexArray(catIndex, tokenCounts);
	}

	/**
	 * P(c) : Returns the probability estimate for the specified category.
	 * 
	 * @param category
	 *            Category whose probability is returned.
	 * @return Probability for category.
	 * 
	 */
	public double probCat(String category) {
		int catIndex = getIndex(category);
		return probCatByIndex(catIndex);
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
			double[] tokenCounts = mFeatureStrToCountsMap.get(featureStr);
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
		return mFeatureStrToCountsMap.get(featureStr);
	}
}
