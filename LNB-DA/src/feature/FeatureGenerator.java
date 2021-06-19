package feature;

import java.util.HashSet;
import java.util.Map;
import java.util.HashMap;

import topicmodel.TopicModel;
import utility.ExceptionUtility;
import classifier.ClassifierParameters;
import nlp.Document;
import nlp.Documents;

public class FeatureGenerator {
	private ClassifierParameters param = null;

	// construction method
	public FeatureGenerator(ClassifierParameters param2) {
		param = param2;
	}

	// training and testing documents
	public void generateAndAssignFeaturesToTrainingAndTestingDocuments(
			Documents trainingDocs, Documents testingDocs,
			TopicModel topicModelForThisDomain) {
		// Generate features for Naive Bayes or Topic Model.
		generateAndAssignFeaturesToTrainingDocuments(trainingDocs,
				topicModelForThisDomain);
		generateAndAssignFeaturesToTestingDocuments(testingDocs,
				topicModelForThisDomain);

		// Generate features for SVM based on features for Naive Bayes.
		generateFeaturesForSVMBasedOnFeaturesForNaiveBayes(Documents
				.getMergedDocuments(trainingDocs, testingDocs));
	}

	// training documents
	private void generateAndAssignFeaturesToTrainingDocuments(
			Documents trainingDocs, TopicModel topicModelForThisDomain) {
		generateAndAssignFeaturesToAnyDocuments(trainingDocs,
				topicModelForThisDomain);
	}

	// testing documents
	private void generateAndAssignFeaturesToTestingDocuments(
			Documents testingDocs, TopicModel topicModelForThisDomain) {
		generateAndAssignFeaturesToAnyDocuments(testingDocs,
				topicModelForThisDomain);
	}

	/**
	 * (main body)
 	 */
	private void generateAndAssignFeaturesToAnyDocuments(Documents documents,
			TopicModel topicModelForThisDomain) {
		// handle each document (review) one by one
		for (Document document : documents) {
			document.featuresForNaiveBayes = new Features();
			// N-Gram features
			if (param.useNGramFeatures) {
				document.featuresForNaiveBayes
						.addFeatures(generate1ToNGramFeaturesForOneDocument(
								document, param.noOfGrams));
			}

			// Topic model features
			if (param.useTopicModelFeatures && topicModelForThisDomain != null) {
				int docIndex = document.docIndex;
				if (topicModelForThisDomain.y != null) {
					int[] y = topicModelForThisDomain.y[docIndex];
					if (y == null) {
						// The document is empty.
						continue;
					}
					// Add sentiment assignments into features.
					for (int sentiment : y) {
						String featureStr = "Sentiment-" + sentiment;
						Feature feature = new Feature(featureStr);
						document.featuresForNaiveBayes.addFeature(feature);
					}
				}
			}
		}

	}

	/**
	 * Generate 1 to Ngrams for one document.
	 */
	public Features generate1ToNGramFeaturesForOneDocument(Document document,
			int noOfGrams) {
		Features features = new Features();
		for (int ngram = 1; ngram <= noOfGrams; ++ngram) {
			features.addFeatures(generateNGramFeaturesForOneDocument(document,
					ngram));
		}
		return features;
	}

	/**
	 * Generate N-gram features for one document.
	 */
	private Features generateNGramFeaturesForOneDocument(Document document,
			int ngram) {
		Features features = new Features();
		String[] words = document.words;
		for (int i = 0; i < words.length - ngram + 1; ++i) {
			boolean completeFeature = true;
			StringBuilder sbFeatureStr = new StringBuilder();
			boolean firstWord = true;
			for (int j = i; j <= i + ngram - 1; ++j) {
				if (words[j].length() == 0) {
					// Empty string (invalid word).
					completeFeature = false;
					break;
				} else {
					if (firstWord == true) {
						firstWord = false;
					} else {
						sbFeatureStr.append("_");
					}
					sbFeatureStr.append(words[j]);

				}
			}
			if (completeFeature) {
				// The n-gram does not contain invalid words.
				String featureStr = sbFeatureStr.toString().trim();
				Feature feature = new Feature(featureStr);
				features.addFeature(feature);
			}
		}
		return features;
	}

	/************************* For SVM ****************************/
	/**
	 * Calculate IDF from both the training and testing data.
	 */
	private void generateFeaturesForSVMBasedOnFeaturesForNaiveBayes(
			Documents allDocuments) {
		// Calculate the TF.
		Map<Integer, Map<String, Integer>> mpDocumentIndexToFeatureStrToTF = new HashMap<Integer, Map<String, Integer>>();
		for (int d = 0; d < allDocuments.size(); ++d) {
			Document document = allDocuments.getDocument(d);
			Map<String, Integer> mpFeatureStrToTF = new HashMap<String, Integer>();
			for (Feature feature : document.featuresForNaiveBayes) {
				String featureStr = feature.featureStr;
				if (!mpFeatureStrToTF.containsKey(featureStr)) {
					mpFeatureStrToTF.put(featureStr, 0);
				}
				mpFeatureStrToTF.put(featureStr,
						mpFeatureStrToTF.get(featureStr) + 1);
			}
			mpDocumentIndexToFeatureStrToTF.put(d, mpFeatureStrToTF);
		}
		// Calculate the DF.
		Map<String, Integer> mpFeatureStrToDF = new HashMap<String, Integer>();
		for (Map.Entry<Integer, Map<String, Integer>> entry : mpDocumentIndexToFeatureStrToTF
				.entrySet()) {
			Map<String, Integer> mpFeatureStrToTF = entry.getValue();
			for (String featureStr : mpFeatureStrToTF.keySet()) {
				if (!mpFeatureStrToDF.containsKey(featureStr)) {
					mpFeatureStrToDF.put(featureStr, 0);
				}
				mpFeatureStrToDF.put(featureStr,
						mpFeatureStrToDF.get(featureStr) + 1);
			}
		}
		// Assign TF-IDF to each feature in each document.
		for (int d = 0; d < allDocuments.size(); ++d) {
			Document document = allDocuments.getDocument(d);
			document.featuresForSVM = new HashSet<Feature>();

			Map<String, Integer> mpFeatureStrToTF = mpDocumentIndexToFeatureStrToTF
					.get(d);

			for (Feature feature : document.featuresForNaiveBayes) {
				String featureStr = feature.featureStr;
				int tf = mpFeatureStrToTF.get(featureStr);
				int df = mpFeatureStrToDF.get(featureStr);
				double idf = Math.log(1.0 * allDocuments.size() / df);

				Feature newFeature = feature.getDeepClone();
				newFeature.featureValue = 0;
				if (param.featureValueSettingForSVM.equals("TF-IDF")) {
					newFeature.featureValue = tf * idf;
				} else if (param.featureValueSettingForSVM.equals("TF")) {
					newFeature.featureValue = tf;
				} else if (param.featureValueSettingForSVM.equals("1")) {
					newFeature.featureValue = 1;
				} else {
					ExceptionUtility
							.throwAndCatchException(param.featureValueSettingForSVM
									+ " is not a reconizable SVM feature setting!");
				}
				document.featuresForSVM.add(newFeature);
			}
		}
	}

	/**
	 * Calculate IDF from the training data only.
	 */
	// private void generateFeaturesForSVMBasedOnFeaturesForNaiveBayes(
	// Documents trainingDocs, Documents testingDocs) {
	// /* Deal with training data. */
	// // Calculate the TF in training data.
	// Map<Integer, Map<String, Integer>>
	// mpDocumentIndexToFeatureStrToTFInTraining = new HashMap<Integer,
	// Map<String, Integer>>();
	// for (int d = 0; d < trainingDocs.size(); ++d) {
	// Document document = trainingDocs.getDocument(d);
	// Map<String, Integer> mpFeatureStrToTFInTraining = new HashMap<String,
	// Integer>();
	// for (Feature feature : document.featuresForNaiveBayes) {
	// String featureStr = feature.featureStr;
	// if (!mpFeatureStrToTFInTraining.containsKey(featureStr)) {
	// mpFeatureStrToTFInTraining.put(featureStr, 0);
	// }
	// mpFeatureStrToTFInTraining.put(featureStr,
	// mpFeatureStrToTFInTraining.get(featureStr) + 1);
	// }
	// mpDocumentIndexToFeatureStrToTFInTraining.put(d,
	// mpFeatureStrToTFInTraining);
	// }
	// // Calculate the DF in the training data.
	// Map<String, Integer> mpFeatureStrToDFInTraining = new HashMap<String,
	// Integer>();
	// for (Map.Entry<Integer, Map<String, Integer>> entry :
	// mpDocumentIndexToFeatureStrToTFInTraining
	// .entrySet()) {
	// Map<String, Integer> mpFeatureStrToTFInTraining = entry.getValue();
	// for (String featureStr : mpFeatureStrToTFInTraining.keySet()) {
	// if (!mpFeatureStrToDFInTraining.containsKey(featureStr)) {
	// mpFeatureStrToDFInTraining.put(featureStr, 0);
	// }
	// mpFeatureStrToDFInTraining.put(featureStr,
	// mpFeatureStrToDFInTraining.get(featureStr) + 1);
	// }
	// }
	// // Assign TF-IDF to each feature in each training document.
	// for (int d = 0; d < trainingDocs.size(); ++d) {
	// Document document = trainingDocs.getDocument(d);
	// document.featuresForSVM = new HashSet<Feature>();
	//
	// Map<String, Integer> mpFeatureStrToTF =
	// mpDocumentIndexToFeatureStrToTFInTraining
	// .get(d);
	//
	// for (Feature feature : document.featuresForNaiveBayes) {
	// String featureStr = feature.featureStr;
	// int tf = mpFeatureStrToTF.get(featureStr);
	// int df = mpFeatureStrToDFInTraining.get(featureStr);
	// double idf = Math.log(1.0 * trainingDocs.size() / df);
	//
	// Feature newFeature = feature.getDeepClone();
	// newFeature.featureValue = 0;
	// if (param.featureValueSettingForSVM.equals("TF-IDF")) {
	// newFeature.featureValue = tf * idf;
	// } else if (param.featureValueSettingForSVM.equals("TF")) {
	// newFeature.featureValue = tf;
	// } else if (param.featureValueSettingForSVM.equals("1")) {
	// newFeature.featureValue = 1;
	// } else {
	// ExceptionUtility
	// .throwAndCatchException(param.featureValueSettingForSVM
	// + " is not a reconizable SVM feature setting!");
	// }
	// document.featuresForSVM.add(newFeature);
	// }
	// }
	//
	// /* Deal with testing data. */
	// // Assign TF-IDF to each feature in each testing document.
	// for (int d = 0; d < testingDocs.size(); ++d) {
	// Document document = testingDocs.getDocument(d);
	// document.featuresForSVM = new HashSet<Feature>();
	//
	// // Calculate TF.
	// Map<String, Integer> mpFeatureStrToTF = new HashMap<String, Integer>();
	// for (Feature feature : document.featuresForNaiveBayes) {
	// String featureStr = feature.featureStr;
	// if (!mpFeatureStrToTF.containsKey(featureStr)) {
	// mpFeatureStrToTF.put(featureStr, 0);
	// }
	// mpFeatureStrToTF.put(featureStr,
	// mpFeatureStrToTF.get(featureStr) + 1);
	// }
	//
	// for (Feature feature : document.featuresForNaiveBayes) {
	// String featureStr = feature.featureStr;
	// int tf = mpFeatureStrToTF.get(featureStr);
	// if (!mpFeatureStrToDFInTraining.containsKey(featureStr)) {
	// // This feature does not appear in the training data, ignore
	// // it.
	// continue;
	// }
	// int df = mpFeatureStrToDFInTraining.get(featureStr);
	// // Same idf value as that in the training data.
	// double idf = Math.log(1.0 * trainingDocs.size() / df);
	//
	// Feature newFeature = feature.getDeepClone();
	// newFeature.featureValue = 0;
	// if (param.featureValueSettingForSVM.equals("TF-IDF")) {
	// newFeature.featureValue = tf * idf;
	// } else if (param.featureValueSettingForSVM.equals("TF")) {
	// newFeature.featureValue = tf;
	// } else if (param.featureValueSettingForSVM.equals("1")) {
	// newFeature.featureValue = 1;
	// } else {
	// ExceptionUtility
	// .throwAndCatchException(param.featureValueSettingForSVM
	// + " is not a reconizable SVM feature setting!");
	// }
	// document.featuresForSVM.add(newFeature);
	// }
	// }
	// }
}
