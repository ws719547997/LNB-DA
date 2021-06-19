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
 * Note: This class can provide knowledge for Lifelong, and is also a Naive Bayes framework.
 *
 * Implement the naive Bayes, copying some from Lingpipe naive Bayes implementation.
 *
 * // Not as good as implementation as Lingpipe.
 */
public class NaiveBayes extends BaseClassifier {
	// The array of category, used in smoothing.
	private final String[] mCategories = { "+1", "-1" };

	// Freq(f, +) and Freq(f, -). -> wordCount(w,c)in POS and NEG documents, i.e., N_{+,w}^KB and N_{-,w}^KB
	private Map<String, double[]> mWordCountInPerClass = new HashMap<String, double[]>();
	private double[] mCountTotalWordsInPerClass = new double[2];// SUM_w {wordCount(w,c)} // indexed by c.
											   // the total number of words in positive and negative category
	private double[] mCountDocsInPerClass = new double[2]; // countDocs(c). -> count number of Docs in each category
	private double mTotalDocsInAllClasses; // SUM_c countDocs(c). -> count the total number of Docs in all categories

	public NaiveBayes() {
	}

	public NaiveBayes(FeatureSelection featureSelection2,
			ClassifierParameters param2) {
		featureSelection = featureSelection2;
		param = param2;
	}

	/**
	 * Train the naive Bayes model. We assume that the word (feature) is
	 * separated by blank character (e.g., ' ').
	 * @param trainingDocs
	 */
	@Override
	public void train(Documents trainingDocs) {
		for (Document trainingDoc : trainingDocs) {
			this.train(trainingDoc.featuresForNaiveBayes, trainingDoc.label);
		}

		// Update classification knowledge.
		knowledge = new ClassificationKnowledge();
		// total number of documents in POS and NEG category -> N_{+} and N_{-}
		knowledge.countDocsInPerClass = mCountDocsInPerClass;
		// Freq(f, +) and Freq(f, -). -> wordCount(w,c)in POS and NEG documents, i.e., N_{+,w}^KB and N_{-,w}^KB
		knowledge.wordCountInPerClass = mWordCountInPerClass;
		// sum_f{Freq(f, +)} and sum_f{Freq(f, -)}. -> number of total words in POS and NEG category
		knowledge.countTotalWordsInPerClass = mCountTotalWordsInPerClass;
	}

	/**
	 * Train the naive Bayes model.
	 * @param testingDocs
	 * @return
	 */
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

	public double[][] classificationPro(Documents testingDocs) {
		double[][] classificationProbability = new double[testingDocs.size()][2];
		String filepath = param.nbRootDirectory + param.domain + ".txt";
		FileOneByOneLineWriter writer = new FileOneByOneLineWriter(filepath);
		int i = 0;
		for (Document testingDoc : testingDocs) {
			classificationProbability[i] = this.getClassificationProbability(
					testingDoc.featuresForNaiveBayes, writer);
			i += 1;
		}
		writer.close();
		return classificationProbability;
	}


	/**
	 * True training the naive Bayes model. training for each review (many features)
	 *
	 * get P(+) and P(-), P(w|+) and P(w|-), i.e., N(+) and N(-), N(w|+) and N(w|-)
	 */
	public void train(Features features, String category) {
		int catIndex = getIndex(category); // get the index (0 or 1) for this category
		mCountDocsInPerClass[catIndex] += 1; // count number of reviews (or documents) of each category
		mTotalDocsInAllClasses += 1; // count the total number of reviews (or documents)

		for (Feature feature : features) {
		// each feature is a word
			String featureStr = feature.featureStr;
			if (featureSelection != null
					&& !featureSelection.isFeatureSelected(featureStr)) {
				// The feature is not selected.
				continue;
			}
			// get wordCount(w,c)in POS and NEG documents
			double[] tokenCounts = mWordCountInPerClass.get(featureStr);
			if (tokenCounts == null) {
				tokenCounts = new double[mCategories.length]; // mCategories = { "+1", "-1" }
				mWordCountInPerClass.put(featureStr, tokenCounts);
			}
			tokenCounts[catIndex] += 1; // change here, the above variable will also changes
			mCountTotalWordsInPerClass[catIndex] += 1;
		}
    }

	/**
	 * Classify the content and return the probability of each category. i.e., P(+|d) and P(-|d)
	 * 
	 * @param writer
	 */
	public double[] getCategoryProbByClassification(Features features,
			FileOneByOneLineWriter writer) {

	    // logps will add all P(w|c) and P(c) for each class
		// Why do this, why transfer to log?
		// The reason is that,
		// in this way, multiplication (\prod{p(w_i|c)}) will transfer to accumulation (\sum{log2(p(w_i|c))}).
		double[] logps = new double[2];
		for (Feature feature : features) {
			String featureStr = feature.featureStr;
			if (featureSelection != null
					&& !featureSelection.isFeatureSelected(featureStr)) {
				// The feature is not selected.
				continue;
			}
			// get wordCount(w,c)in POS or NEG documents
			double[] tokenCounts = mWordCountInPerClass.get(featureStr);
			if (tokenCounts == null) {
				continue;
			}
			double[] tempLogPs = new double[2]; // -> POS and NEG
			for (int catIndex = 0; catIndex < mCategories.length; ++catIndex) {
				// where mCategories.length is 2
				logps[catIndex] += com.aliasi.util.Math
						.log2(probTokenByIndexArray(catIndex, tokenCounts));
				tempLogPs[catIndex] += com.aliasi.util.Math
						.log2(probTokenByIndexArray(catIndex, tokenCounts));
			}
			// Normalize the probability array, including .Math.pow
			tempLogPs = logJointToConditional(tempLogPs);
			writer.write(featureStr + ":" + tempLogPs[0] + "/" + tempLogPs[1]
					+ " ");
		}

		// add log class probability, i.e., p(+) and p(-)
		for (int catIndex = 0; catIndex < mCategories.length; ++catIndex) {
			logps[catIndex] += com.aliasi.util.Math.log2(probCatByIndex(catIndex));
		}
		writer.writeLine(); // write "line.separator", i.e., "\n"
		// Normalize the probability array, including .Math.pow
		return logJointToConditional(logps);
	}

	/**
	 * Classify the content and return best category with highest probability.
	 * 
	 * @param writer
	 */
	public String getBestCategoryByClassification(Features features,
			FileOneByOneLineWriter writer) {
		// get the probability of each category. i.e., P(+|d) and P(-|d)
		double[] categoryProb = getCategoryProbByClassification(features, writer);

		// get the best category with highest probability
		double maximumProb = -Double.MAX_VALUE;
		int maximumIndex = -1;
		for (int i = 0; i < categoryProb.length; ++i) {
			if (maximumProb < categoryProb[i]) {
				maximumProb = categoryProb[i];
				maximumIndex = i;
			}
		}
		return mCategories[maximumIndex]; // mCategories = { "+1", "-1" }
	}

	public double[] getClassificationProbability(Features features,
												  FileOneByOneLineWriter writer) {
		// get the probability of each category. i.e., P(+|d) and P(-|d)
		return getCategoryProbByClassification(features, writer);
	}

	/**
	 * Pr(w|c) = (0.5 + count(w,c)) / (|V| * 0.5 + count(all of w, c)).
	 */
	public double probTokenByIndexArray(int catIndex, double[] tokenCounts) {
        double tokenCatCount = tokenCounts[catIndex];
        double totalCatCount = mCountTotalWordsInPerClass[catIndex];
        return (tokenCatCount + param.smoothingPriorForFeatureInNaiveBayes)
                / (totalCatCount + mWordCountInPerClass.size()
                * param.smoothingPriorForFeatureInNaiveBayes);
		// Note: mWordCountInPerClass.size() is the size of target training vocabulary,
		// i.e., the number of current featured words (selected features),
		// not the size of knowledge vocabulary.
	}

	/**
	 * P(c) = (0.5 + count(c)) / (|C|) * 0.5 + count(all of c)).
	 */
	private double probCatByIndex(int catIndex) {
		double caseCountCat = mCountDocsInPerClass[catIndex];

		return (caseCountCat + param.mCategoryPrior)
				/ (mTotalDocsInAllClasses + mCategories.length * param.mCategoryPrior);
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
	 * @throws IllegalArgumentException
	 *             If the category is not known or the token is not known.
	 */
	public double probToken(String token, String cat) {
		int catIndex = getIndex(cat);
		double[] tokenCounts = mWordCountInPerClass.get(token);
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
				// where POSITIVE_INFINITY = 1.0 / 0.0
				probRatios[i] = Float.MAX_VALUE;
			else if (probRatios[i] == Double.NEGATIVE_INFINITY
					|| Double.isNaN(probRatios[i]))
				// where NEGATIVE_INFINITY = -1.0 / 0.0
				probRatios[i] = 0.0;
		}
		return com.aliasi.stats.Statistics.normalize(probRatios);
	}

	public double[] getTotalCountsPerCategory() {
		return mCountTotalWordsInPerClass;
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
			double[] tokenCounts = mWordCountInPerClass.get(featureStr);
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
		return mWordCountInPerClass.get(featureStr);
	}

	public void verifyUnfoundFeatures() {
	    for (String featureStr : this.featureSelection.selectedFeatureStrs) {
            if (!this.mWordCountInPerClass.containsKey(featureStr)) {
                System.out.println(featureStr);
            }
        }
    }

    public static void main(String[] args) {
        double[] testPro = {-4, -9};
        NaiveBayes nb = new NaiveBayes();
        double[] result = nb.logJointToConditional(testPro);
        System.out.println(result[0] + "---" + result[1]);
    }
}
