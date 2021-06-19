//package multithread;
//
//import java.io.File;
//import java.util.ArrayList;
//import java.util.HashMap;
//import java.util.List;
//import java.util.Map;
//import java.util.concurrent.Callable;
//
//import classificationevaluation.ClassificationEvaluation;
//import classificationevaluation.ClassificationEvaluationAccumulator;
//import classifier.BaseClassifier;
//import classifier.ClassificationKnowledge;
//import classifier.ClassifierParameters;
//import feature.FeatureGenerator;
//import feature.FeatureSelection;
//import feature.InformationGain;
//import topicmodel.ModelLoader;
//import topicmodel.TopicModel;
//import utility.CrossValidationOperatorMaintainingLabelDistribution;
//import utility.FileReaderAndWriter;
//import utility.Pair;
//import nlp.Document;
//import nlp.Documents;
//
//public class SentimentClassificationCallable_old implements
//		Callable<ClassificationEvaluation> {
//	private Documents documents = null;
//	private List<Documents> documentsOfOtherDomains = null;
//	private ClassifierParameters param = null;
//
//	// The probabilities of the below map are from computed from the source
//	// domains.
//	private Map<String, ClassificationKnowledge> mpDomainToClassificationProbs = null;
//
//	public SentimentClassificationCallable_old(Documents documents2,
//			List<Documents> documentsOfOtherDomains2,
//			ClassifierParameters param2) {
//		documents = documents2;
//		documentsOfOtherDomains = documentsOfOtherDomains2;
//		param = param2;
//	}
//
//	public SentimentClassificationCallable_old(
//			Documents documents2,
//			Map<String, ClassificationKnowledge> mpDomainToClassificationProbs2,
//			ClassifierParameters param2) {
//		documents = documents2;
//		mpDomainToClassificationProbs = mpDomainToClassificationProbs2;
//		param = param2;
//	}
//
//	@Override
//	/**
//	 * Run the topic model in a domain and print it into the disk.
//	 */
//	public ClassificationEvaluation call() throws Exception {
//		try {
//			System.out.println("\"" + param.domain + "\" <"
//					+ param.classifierName + "> Starts...");
//
//			TopicModel topicModelForThisDomain = null;
//			if (param.useTopicModelFeatures) {
//				topicModelForThisDomain = readTopicModelForThisDomain(
//						param.domain, param.topicModelNameForFeatureGeneration,
//						param.topicModelSettingNameForFeatureGeneration,
//						param.outputTopicModelMultiDomainFilepath);
//			}
//
//			ClassificationEvaluation evaluation = getClassificationEvaluation(
//					documents, documentsOfOtherDomains,
//					topicModelForThisDomain, param);
//
//			System.out.println("\"" + param.domain + "\" <"
//					+ param.classifierName + ": " + evaluation.accuracy
//					+ "> Ends...");
//			return evaluation;
//		} catch (Exception ex) {
//			ex.printStackTrace();
//		}
//		return null;
//	}
//
//	public TopicModel readTopicModelForThisDomain(String domain,
//			String topicModelNameForFeatureGeneration,
//			String topicModelSettingNameForFeatureGeneration,
//			String outputTopicModelMultiDomainFilepath) {
//		// Read topic model if exists.
//		TopicModel topicModelForThisDomain = null;
//		String topicModelDirectory = outputTopicModelMultiDomainFilepath
//				+ topicModelSettingNameForFeatureGeneration + File.separator
//				+ "DomainModels" + File.separator + domain + File.separator;
//		if (new File(topicModelDirectory).exists()) {
//			ModelLoader modelLoader = new ModelLoader();
//			topicModelForThisDomain = modelLoader.loadModel(
//					topicModelNameForFeatureGeneration, domain,
//					topicModelDirectory);
//		}
//		return topicModelForThisDomain;
//	}
//
//	public ClassificationEvaluation getClassificationEvaluation(
//			Documents documents, List<Documents> documentsOfOtherDomains,
//			TopicModel topicModelForThisDomain, ClassifierParameters param) {
//		// Extract top features from other domains.
//		Map<String, Map<String, double[]>> mpSelectedFeatureStrToDomainToProbs = null;
//		if (param.useFeaturesFromOtherDomains) {
//			mpSelectedFeatureStrToDomainToProbs = getTopFeaturesFromOtherDomains(
//					documentsOfOtherDomains, param.noOfSelectedFeatures);
//		}
//
//		// Cross validation.
//		ClassificationEvaluationAccumulator evaluationAccumulator = new ClassificationEvaluationAccumulator();
//		CrossValidationOperatorMaintainingLabelDistribution cvo = new CrossValidationOperatorMaintainingLabelDistribution(
//				documents, param.noOfCrossValidationFolders);
//		for (int folderIndex = 0; folderIndex < param.noOfCrossValidationFolders; ++folderIndex) {
//			// System.out.println("Folder No " + folderIndex);
//			// Assign directory for SVM Light.
//			param.svmLightCVFoldDirectory = param.svmLightRootDirectory
//					+ param.domain + File.separator + "CV" + folderIndex
//					+ File.separator;
//
//			Pair<Documents, Documents> pair = cvo
//					.getTrainingAndTestingDocuments(folderIndex);
//			Documents trainingDocs = new Documents();
//			if (param.includeTargetDomainLabeledDataForTraining) {
//				trainingDocs.addDocuments(pair.t);
//			}
//			if (param.mergeSimplyLabeledDataFromOtherDomainsForTraining) {
//				for (Documents documentsForOneDomain : documentsOfOtherDomains) {
//					trainingDocs.addDocuments(documentsForOneDomain);
//				}
//			}
//			Documents testingDocs = pair.u;
//
//			// Feature generation.
//			FeatureGenerator featureGenerator = new FeatureGenerator(param);
//			featureGenerator
//					.generateAndAssignFeaturesToTrainingAndTestingDocuments(
//							trainingDocs, testingDocs, topicModelForThisDomain);
//			// Feature selection.
//			FeatureSelection featureSelection = FeatureSelection
//					.selectFeatureSelection(trainingDocs, param);
//
//			// Print out top features .
//			featureSelection
//					.printSelectedFeaturesToFile(param.outputTopFeaturesFilePath
//							+ param.domain + ".txt");
//
//			// Build the classifier.
//			BaseClassifier classifier = BaseClassifier.selectClassifier(
//					param.classifierName, featureSelection,
//					mpSelectedFeatureStrToDomainToProbs, param);
//			classifier.train(trainingDocs);
//
//			ClassificationEvaluation evaluation = classifier.test(testingDocs);
//
//			if (param.misclassifiedDocumentsFilePath != null) {
//				String misclassifiedDocumentsForOneCVFolderForOneDomainFilePath = param.misclassifiedDocumentsFilePath
//						+ param.domain
//						+ File.separator
//						+ "CV"
//						+ folderIndex
//						+ ".txt";
//				printMisclassifiedDocuments(testingDocs,
//						misclassifiedDocumentsForOneCVFolderForOneDomainFilePath);
//			}
//
//			evaluationAccumulator.addClassificationEvaluation(evaluation);
//		}
//		return evaluationAccumulator.getAverageClassificationEvaluation();
//	}
//
//	private void printMisclassifiedDocuments(Documents testingDocs,
//			String misclassifiedDocumentsForOneCVFolderForOneDomainFilePath) {
//		StringBuilder sbOutput = new StringBuilder();
//		sbOutput.append("Document\tLabel\tPredict");
//		sbOutput.append(System.lineSeparator());
//		for (Document testingDoc : testingDocs) {
//			if (!testingDoc.label.equals(testingDoc.predict)) {
//				// Misclassified Document.
//				sbOutput.append(testingDoc.text + "\t" + testingDoc.label
//						+ "\t" + testingDoc.predict);
//				sbOutput.append(System.lineSeparator());
//			}
//		}
//		FileReaderAndWriter.writeFile(
//				misclassifiedDocumentsForOneCVFolderForOneDomainFilePath,
//				sbOutput.toString());
//	}
//
//	private Map<String, Map<String, double[]>> getTopFeaturesFromOtherDomains(
//			List<Documents> documentsOfOtherDomains, int noOfSelectedFeatures) {
//		// Generate param for each domain.
//		List<ClassifierParameters> paramList = new ArrayList<ClassifierParameters>();
//		for (Documents documentsForOneDomain : documentsOfOtherDomains) {
//			paramList
//					.add(new ClassifierParameters(documentsForOneDomain, param));
//		}
//		List<FeatureSelection> featureSelectionList = new ArrayList<FeatureSelection>();
//		for (int i = 0; i < documentsOfOtherDomains.size(); ++i) {
//			Documents documentsForOneDomain = documentsOfOtherDomains.get(i);
//			ClassifierParameters paramForOneDomain = paramList.get(i);
//			// Feature generation for each domain.
//			FeatureGenerator featureGenerator = new FeatureGenerator(
//					paramForOneDomain);
//			featureGenerator.generateAndAssignFeaturesToDocuments(
//					documentsForOneDomain, null);
//			// Feature selection.
//			FeatureSelection featureSelection = FeatureSelection
//					.selectFeatureSelection(documentsForOneDomain,
//							paramForOneDomain);
//			featureSelectionList.add(featureSelection);
//		}
//
//		Map<String, Map<String, double[]>> mpFeatureStrToDomainToProbs = new HashMap<String, Map<String, double[]>>();
//		for (int i = 0; i < documentsOfOtherDomains.size(); ++i) {
//			Documents documentsForOneDomain = documentsOfOtherDomains.get(i);
//			InformationGain informationGainForOneDomain = new InformationGain(
//					documentsForOneDomain);
//			// ClassifierParameters paramForOneDomain = paramList.get(i);
//			FeatureSelection featureSelectionForOneDomain = featureSelectionList
//					.get(i);
//
//			String domain = documentsForOneDomain.domain;
//			List<String> selectedFeatureStrs = featureSelectionForOneDomain
//					.getSelectedFeatureStrs();
//			for (String featureStr : selectedFeatureStrs) {
//				double probOfFeatureGivenPositive = informationGainForOneDomain
//						.getProbOfFeatureGivenPositive(featureStr);
//				double probOfFeatureGivenNegative = informationGainForOneDomain
//						.getProbOfFeatureGivenNegative(featureStr);
//				double[] probs = new double[] { probOfFeatureGivenPositive,
//						documentsForOneDomain.getNoOfPositiveLabels(),
//						probOfFeatureGivenNegative,
//						documentsForOneDomain.getNoOfNegativeLabels() };
//				if (!mpFeatureStrToDomainToProbs.containsKey(featureStr)) {
//					mpFeatureStrToDomainToProbs.put(featureStr,
//							new HashMap<String, double[]>());
//				}
//				Map<String, double[]> mpDomainToProbs = mpFeatureStrToDomainToProbs
//						.get(featureStr);
//				mpDomainToProbs.put(domain, probs);
//			}
//		}
//
//		return mpFeatureStrToDomainToProbs;
//
//		// List<ItemWithValue> featuresWithSomeValue = new
//		// ArrayList<ItemWithValue>();
//		// Map<String, Integer> mpFeatureStrToPositiveFrequency = new
//		// HashMap<String, Integer>();
//		// Map<String, Integer> mpFeatureStrToNegativeFrequency = new
//		// HashMap<String, Integer>();
//		// Map<String, Double> mpFeatureStrToAverageEntropy = new
//		// HashMap<String, Double>();
//		// for (Map.Entry<String, Map<String, double[]>> entry :
//		// mpFeatureStrToDomainToProbs
//		// .entrySet()) {
//		// String featureStr = entry.getKey();
//		// Map<String, double[]> mpDomainToProbs = mpFeatureStrToDomainToProbs
//		// .get(featureStr);
//		// // Calculate feature frequency.
//		// int featureFrequency = mpDomainToProbs.size();
//		// if (featureFrequency <= param.featureFrequencyThreshold) {
//		// // Filter infrequent features.
//		// continue;
//		// }
//		// // Calculate P(+|f) and P(-|f) in each domain.
//		// mpFeatureStrToPositiveFrequency.put(featureStr, 0);
//		// mpFeatureStrToNegativeFrequency.put(featureStr, 0);
//		//
//		// double[] informationGains = new double[mpDomainToProbs.size()];
//		// int i = 0;
//		// for (double[] probsForOneDomain : mpDomainToProbs.values()) {
//		// double probOfPositiveGivenFeature = probsForOneDomain[0];
//		// double probOfNegativeGivenFeature = probsForOneDomain[1];
//		// double informationGain = probsForOneDomain[2];
//		// if (probOfPositiveGivenFeature >= probOfNegativeGivenFeature) {
//		// mpFeatureStrToPositiveFrequency
//		// .put(featureStr, mpFeatureStrToPositiveFrequency
//		// .get(featureStr) + 1);
//		// } else {
//		// mpFeatureStrToNegativeFrequency
//		// .put(featureStr, mpFeatureStrToNegativeFrequency
//		// .get(featureStr) + 1);
//		// }
//		// // double entropy = EntropyHelper
//		// // .getEntropy(new double[] { probOfPositiveGivenFeature,
//		// // probOfNegativeGivenFeature });
//		// informationGains[i++] = informationGain;
//		// }
//		//
//		// // Compute average information gain over domains.
//		// double averageIG = ArraySumAndAverageAndMaxAndMin
//		// .getAverage(informationGains);
//		// mpFeatureStrToAverageEntropy.put(featureStr, averageIG);
//		//
//		// // Computer entropy of this feature appearing in positive and
//		// // negative.
//		// double entropy = 10 - EntropyHelper.getEntropy(new int[] {
//		// mpFeatureStrToPositiveFrequency.get(featureStr),
//		// mpFeatureStrToNegativeFrequency.get(featureStr) });
//		//
//		// // Ranked by feature frequency.
//		// featuresWithSomeValue.add(new ItemWithValue(featureStr, averageIG
//		// * entropy));
//		// }
//		// // Collections.sort(featuresWithDomainFrequency,
//		// // Collections.reverseOrder());
//		// Collections.sort(featuresWithSomeValue);
//
//		// if (param.featuresRankedByInformationGainInOtherDomains != null) {
//		// // Print to file.
//		// String featuresRankedByInformationGainInOtherDomains =
//		// param.featuresRankedByInformationGainInOtherDomains
//		// + param.domain + ".txt";
//		// StringBuilder sbDebug = new StringBuilder();
//		// for (ItemWithValue iwv : featuresWithSomeValue) {
//		// String featureStr = iwv.getItem().toString();
//		// double value = iwv.getValue();
//		// sbDebug.append(featureStr + "\t" + value + "\t"
//		// + mpFeatureStrToPositiveFrequency.get(featureStr)
//		// + "\t"
//		// + mpFeatureStrToNegativeFrequency.get(featureStr));
//		// sbDebug.append(System.lineSeparator());
//		// }
//		// FileReaderAndWriter.writeFile(
//		// featuresRankedByInformationGainInOtherDomains,
//		// sbDebug.toString());
//		// }
//
//		// Map<String, Map<String, double[]>>
//		// mpSelectedFeatureStrToDomainToProbs = new HashMap<String, Map<String,
//		// double[]>>();
//		// for (int i = 0; i < featuresWithSomeValue.size(); ++i) {
//		// ItemWithValue iwv = featuresWithSomeValue.get(i);
//		// String featureStr = iwv.getItem().toString();
//		// // double value = iwv.getValue();
//		// if (i < param.noOfTopFeaturesByIGFromOtherDomains) {
//		// mpSelectedFeatureStrToDomainToProbs.put(featureStr,
//		// mpFeatureStrToDomainToProbs.get(featureStr));
//		// } else {
//		// // mpFeatureStrToSelection.put(featureStr, false);
//		// }
//		// }
//		// return mpSelectedFeatureStrToDomainToProbs;
//	}
// }
