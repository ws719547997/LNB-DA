package classifier;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import utility.ItemWithValue;
import classificationevaluation.ClassificationEvaluation;
import feature.FeatureSelection;
import nlp.Document;
import nlp.Documents;

/**
 * Implement the proposed lifelong model using stochastic gradient descent.
 *
 * In fact, it implements .train() and .test() in class NaiveBayes_SGD
 * for training and testing the proposed model.
 *
 */
public class NaiveBayes_SGD_Lifelong extends BaseClassifier {
	// using for calling .train() and .test() in class NaiveBayes_SGD.
	private NaiveBayes_SGD sgdModel = null;

	public NaiveBayes_SGD_Lifelong(FeatureSelection featureSelection2,
			ClassificationKnowledge knowledge2, ClassifierParameters param2) {
		featureSelection = featureSelection2;
		if (knowledge2 != null) {
			knowledge = knowledge2.getDeepClone();
		} else {
			knowledge = new ClassificationKnowledge();
		}
		targetKnowledge = new ClassificationKnowledge();
		param = param2;
	}

	/**
	 * This only works for binary classification for now.
	 */
	@Override
	public void train(Documents trainingDocs) {
		// Expand knowledge-base with the knowledge from the classic
		// naive Bayes classifier for this target domain
		NaiveBayes nbClassifier = new NaiveBayes(featureSelection, param);
		// get knowledge for this training documents
		nbClassifier.train(trainingDocs);

		// check whether knowledge.wordCountInPerClass contains the key of selected features
		// (no need this as the selected features are from knowledge, Hao)
		// nbClassifier.verifyUnfoundFeatures();

		// TODO: store target knowledge, and update knowledge base
		// store current target knowledge
		targetKnowledge.addKnowledge(nbClassifier.knowledge);
		// add current knowledge from target domain into knowledge-base
		knowledge.addKnowledge(nbClassifier.knowledge);
		// the size of total knowledge vocabulary
		int knowledgeVocabularySize = knowledge.wordCountInPerClass.size();

		// ----------------------------------------------
		// For regularization ratio term and coefficient.
		// ----------------------------------------------
		// 1. reliable features
		Set<String> setOfReliableWordsInTargetDomain = new HashSet<String>();
		// 2. Expected wordCount in POS
		Map<String, Double> regExpectedWordCountInPositive = new HashMap<String, Double>();
		// 3. regularization coefficient \alpha
		Map<String, Double> regCoefficientAlphaInPositive = new HashMap<String, Double>();
		// 4. Expected wordCount in NEG
		Map<String, Double> regExpectedWordCountInNegative = new HashMap<String, Double>();
		// 5. regularization coefficient \alpha
		Map<String, Double> regCoefficientAlphaInNegative = new HashMap<String, Double>();

		// This for{} structure handles each featured word one by one.
		// where all selected features (also words) are covered by all documents of the target domain.
		// That is to say,
		// selected features are real valid features among target domain.
		for (String featureStr : nbClassifier.featureSelection.selectedFeatureStrs) {
			// get wordCount(w,c)in POS and NEG in training Docs in target domain
			double[] wordCountInPerClassInTargetDomain = nbClassifier
					.getCountsOfClasses(featureStr);

//			if (wordCountInPerClassInTargetDomain == null) {
//				System.out.println("watch out");
//			}

			// transfer wordCount(w,c) to Pr(w|c)
			double profOfFeatureGivenPositiveInTargetDomain = nbClassifier
					.probTokenByIndexArray(0, wordCountInPerClassInTargetDomain); // Pr(w|+)
			double profOfFeatureGivenNegativeInTargetDomain = nbClassifier
					.probTokenByIndexArray(1, wordCountInPerClassInTargetDomain); // Pr(w|-)

			// get Ratio: Pr(w|+)/Pr(w|-)
			double ratioInTargetDomain = profOfFeatureGivenPositiveInTargetDomain
					/ profOfFeatureGivenNegativeInTargetDomain;
			// get {Pr(w|+)*(\lambda|V|+\sum_v X_{+,v})} / {Pr(w|-)*(\lambda|V|+\sum_v X_{-,v})}
			double ratioExpected = profOfFeatureGivenPositiveInTargetDomain
					/ profOfFeatureGivenNegativeInTargetDomain
					* (knowledge.countTotalWordsInPerClass[0] + knowledgeVocabularySize
							* param.smoothingPriorForFeatureInNaiveBayes)
					/ (knowledge.countTotalWordsInPerClass[1] + knowledgeVocabularySize
							* param.smoothingPriorForFeatureInNaiveBayes);

			// Penalty term 1, i.e., Eq(7) in paper
			// Distinguishable target domain dependent words: V_T
			if ((ratioInTargetDomain >= param.positiveRatioThreshold
					&& wordCountInPerClassInTargetDomain[0] >= param.positiveOrNegativeFrequencyThreshold)
					|| (ratioInTargetDomain <= param.negativeRatioThreshold
					&& wordCountInPerClassInTargetDomain[1] >= param.positiveOrNegativeFrequencyThreshold)) {
				// add reliable feature (also word) into V_T
				setOfReliableWordsInTargetDomain.add(featureStr);
				// get wordCount(w,c) from knowledge base
				double[] wordCountInPerClassInKnowledge = knowledge.wordCountInPerClass
						.get(featureStr);
				// get word total count, i.e., wordCount(w,+) + wordCount(w,-)
				double wordTotalCountInAllClasses = wordCountInPerClassInKnowledge[0]
						+ wordCountInPerClassInKnowledge[1];
				// TODO: delete below
//				double c = profOfFeatureGivenPositiveInTargetDomain + profOfFeatureGivenNegativeInTargetDomain;
//				double a1 = wordTotalCountInAllClasses * profOfFeatureGivenPositiveInTargetDomain / c;
//				double a2 = wordTotalCountInAllClasses * ratioExpected / (ratioExpected + 1);
//				double a3 = wordCountInPerClassInTargetDomain[0];

//				// do it as Eq(7) in paper
//				regExpectedWordCountInPositive.put(featureStr,
//						wordCountInPerClassInTargetDomain[0]);
//				regCoefficientAlphaInPositive.put(featureStr,
//						param.regCoefficientAlpha);
//				regExpectedWordCountInNegative.put(featureStr,
//						wordCountInPerClassInTargetDomain[1]);
//				regCoefficientAlphaInNegative.put(featureStr,
//						param.regCoefficientAlpha);

				// Old codes which Nianzu sent me
				regExpectedWordCountInPositive.put(featureStr,
						wordTotalCountInAllClasses * ratioExpected / (ratioExpected + 1));
				regCoefficientAlphaInPositive.put(featureStr,
						param.regCoefficientAlpha);
				regExpectedWordCountInNegative.put(featureStr,
						wordTotalCountInAllClasses * 1.0 / (ratioExpected + 1));
				regCoefficientAlphaInNegative.put(featureStr,
						param.regCoefficientAlpha);
			}
		}

		// Penalty term 2, i.e., Eq(8) in paper
		// If knowledge reliable, utilize only those reliable parts of knowledge, i.e., will ignore penalty term 1.
		if (param.domainLevelKnowledgeSupportThreshold >= 0) {
			// Add domain level knowledge as regularization.
			for (Map.Entry<String, double[]> entry : knowledge.countDomainsInPerClass
					.entrySet()) {
				String featureStr = entry.getKey();
				// if (setOfReliableWordsInTargetDomain.contains(featureStr))
				// {
				// // Do not use knowledge when the features are reliable in
				// // the target domain.
				// continue;
				// }

				// Apply domain level knowledge.
				double[] domainCounts = knowledge.countDomainsInPerClass
						.get(featureStr);
				if (domainCounts != null
						&& (domainCounts[0] >= param.domainLevelKnowledgeSupportThreshold
						|| domainCounts[1] >= param.domainLevelKnowledgeSupportThreshold)) {
					double positivePercentage = domainCounts[0]
							/ (domainCounts[0] + domainCounts[1]); // R_w in Eq(8)
					// Note: knowledge.wordCountInPerClass.get(featureStr)[0] is equal to X_{+,w}^0 in paper,
					// and knowledge.wordCountInPerClass.get(featureStr)[1] is equal to X_{-,w}^0 in paper
					double expectedWordCountInPositive = knowledge.wordCountInPerClass
							.get(featureStr)[0] * positivePercentage;
					double expectedWordCountInNegative = knowledge.wordCountInPerClass
							.get(featureStr)[1] * (1 - positivePercentage);

					regExpectedWordCountInPositive.put(featureStr,
							expectedWordCountInPositive);
					regCoefficientAlphaInPositive.put(featureStr,
							param.regCoefficientAlpha);
					regExpectedWordCountInNegative.put(featureStr,
							expectedWordCountInNegative);
					regCoefficientAlphaInNegative.put(featureStr,
							param.regCoefficientAlpha);
				}
			}
		}

		sgdModel = new NaiveBayes_SGD(param, featureSelection, knowledge, targetKnowledge,
				regExpectedWordCountInPositive,
				regCoefficientAlphaInPositive,
				regExpectedWordCountInNegative,
				regCoefficientAlphaInNegative);
		sgdModel.train(trainingDocs);
	}

	@Override
	public ClassificationEvaluation test(Documents testingDocs) {
		return sgdModel.test(testingDocs);
	}

	public ClassificationEvaluation test_with_only_traing_vocab(Documents testingDocs, Documents trainingDocs) {
		return sgdModel.test_with_only_training_feature(testingDocs, trainingDocs);
	}

	@Override
	public List<ItemWithValue> getFeaturesByRatio(Document testingDoc) {
		return sgdModel.getFeaturesByRatio(testingDoc);
	}

	@Override
	public void printMisclassifiedDocuments(Documents testingDocs, String misclassifiedDocumentsForOneCVFolderForOneDomainFilePath) {
		sgdModel.printMisclassifiedDocuments(testingDocs, misclassifiedDocumentsForOneCVFolderForOneDomainFilePath);
	}

	@Override
	public double[] getCountsOfClasses(String featureStr) {
		return sgdModel.getCountsOfClasses(featureStr);
	}
}
