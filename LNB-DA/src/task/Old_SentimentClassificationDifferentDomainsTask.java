//package task;
//
//import java.io.File;
//import java.util.Map;
//import java.util.TreeMap;
//
//import classificationevaluation.ClassificationEvaluation;
//import classifier.NaiveBayes_MyImplementation;
//import utility.ArraysSumAndAverageAndMaxAndMin;
//import utility.CrossValidationOperatorMaintainingLabelDistribution;
//import utility.Pair;
//import nlp.Document;
//import nlp.Documents;
//
//public class Old_SentimentClassificationDifferentDomainsTask {
//	public String inputReviewDirectory = "..\\Data\\Input\\DifferentDomains\\50Electronics_100+100-\\";
//	public int noOfCVFolder = 10;
//
//	public void run() {
//		try {
//			Map<String, Documents> mpDomainToDocuments = new TreeMap<String, Documents>();
//			Map<String, CrossValidationOperatorMaintainingLabelDistribution> mpDomainToCVO = new TreeMap<String, CrossValidationOperatorMaintainingLabelDistribution>();
//
//			File[] domainFiles = new File(inputReviewDirectory).listFiles();
//			for (File domainFile : domainFiles) {
//				// Get the domain name, i.e., file name without extension.
//				String domain = domainFile.getName().replaceFirst("[.][^.]+$",
//						"");
//				Documents documents = Documents.readDocuments(domain,
//						domainFile.getAbsolutePath());
//				mpDomainToDocuments.put(domain, documents);
//				CrossValidationOperatorMaintainingLabelDistribution cvo = new CrossValidationOperatorMaintainingLabelDistribution(
//						documents, noOfCVFolder);
//				mpDomainToCVO.put(domain, cvo);
//			}
//
//			// Train the model using training data on one domain and test
//			// on testing data on the same domain.
//			trainAndTestOnSameDomain(mpDomainToCVO);
//
//			// Train the model using training data on all domains and test
//			// on testing data on each domain.
//			// trainOnAllDomainsAndTestOnEachDomain(mpDomainToCVO);
//
//			// Train the model using training data on all domains except the
//			// target domain and test on testing data on the target domain.
//			// trainOnAllDomainsExceptTargetAndTestOnTargetDomain(mpDomainToCVO);
//
//		} catch (Exception ex) {
//			ex.printStackTrace();
//		}
//	}
//
//	/**
//	 * Train the model using training data on one domain and test on testing
//	 * data on the same domain.
//	 **/
//	private void trainAndTestOnSameDomain(
//			Map<String, CrossValidationOperatorMaintainingLabelDistribution> mpDomainToCVO) {
//		for (Map.Entry<String, CrossValidationOperatorMaintainingLabelDistribution> entry : mpDomainToCVO
//				.entrySet()) {
//			String domain = entry.getKey();
//			CrossValidationOperatorMaintainingLabelDistribution cvo = entry
//					.getValue();
//
//			double[] accuracies = new double[noOfCVFolder];
//			for (int folderIndex = 0; folderIndex < noOfCVFolder; ++folderIndex) {
//				Pair<Documents, Documents> pair = cvo
//						.getTrainingAndTestingDocuments(folderIndex);
//				Documents trainingDocs = pair.t;
//				Documents testingDocs = pair.u;
//
//				// Build the naive Bayes model.
//				NaiveBayes_MyImplementation classifierNB = new NaiveBayes_MyImplementation();
//				for (Document trainingDoc : trainingDocs) {
//					classifierNB.train(trainingDoc.text, trainingDoc.label);
//				}
//
//				// if (folderIndex == 0) {
//				// List<ItemWithValue> topPosFeatures = classifierNB
//				// .getTopPositiveFeaturesByProbOfClassGivenTokens(20);
//				// List<ItemWithValue> topNegFeatures = classifierNB
//				// .getTopNegativeFeaturesByProbOfClassGivenTokens(20);
//				// StringBuilder sbTopFeatures = new StringBuilder();
//				// sbTopFeatures.append(domain + "\t");
//				// for (ItemWithValue iwv : topPosFeatures) {
//				// String feature = iwv.getItem().toString();
//				// sbTopFeatures.append(feature + "\t");
//				// }
//				// System.out.println(sbTopFeatures.toString().trim());
//				// }
//
//				for (Document testingDoc : testingDocs) {
//					testingDoc.predict = classifierNB
//							.getBestCategoryByClassification(testingDoc.text);
//				}
//				ClassificationEvaluation evaluation = new ClassificationEvaluation(
//						testingDocs.getLabels(), testingDocs.getPredicts());
//				// System.out.println(evaluation.toString());
//				accuracies[folderIndex] = evaluation.accuracy;
//			}
//
//			double averageAccuracy = ArraysSumAndAverageAndMaxAndMin
//					.getAverage(accuracies);
//			System.out.println(domain + "\t"
//					+ String.format("%.2f", averageAccuracy));
//			// System.out.println(ArraysSumAndAverage.getAverage(accuracies));
//		}
//	}
//
//	/**
//	 * Train the model using training data on all domains and test on testing
//	 * data on each domain.
//	 */
//	private void trainOnAllDomainsAndTestOnEachDomain(
//			Map<String, CrossValidationOperatorMaintainingLabelDistribution> mpDomainToCVO) {
//		Map<String, double[]> mpDomainToAccuracies = new TreeMap<String, double[]>();
//		for (int folderIndex = 0; folderIndex < noOfCVFolder; ++folderIndex) {
//			// Merge training data from all domains.
//			Documents trainingDocsAllDomains = new Documents();
//			for (Map.Entry<String, CrossValidationOperatorMaintainingLabelDistribution> entry : mpDomainToCVO
//					.entrySet()) {
//				// String domain = entry.getKey();
//				CrossValidationOperatorMaintainingLabelDistribution cvo = entry
//						.getValue();
//				Documents trainingDocsOneDomain = cvo
//						.getTrainingDocuments(folderIndex);
//				trainingDocsAllDomains.addDocuments(trainingDocsOneDomain);
//			}
//
//			// Train the model using merged training data on all domains.
//			// Build the naive Bayes model.
//			NaiveBayes_MyImplementation classifierNB = new NaiveBayes_MyImplementation();
//			for (Document trainingDoc : trainingDocsAllDomains) {
//				classifierNB.train(trainingDoc.text, trainingDoc.label);
//			}
//
//			// if (folderIndex == 0) {
//			// List<ItemWithValue> topPosFeatures = classifierNB
//			// .getTopPositiveFeaturesByProbOfClassGivenTokens(20);
//			// List<ItemWithValue> topNegFeatures = classifierNB
//			// .getTopNegativeFeaturesByProbOfClassGivenTokens(20);
//			// StringBuilder sbTopFeatures = new StringBuilder();
//			// for (ItemWithValue iwv : topNegFeatures) {
//			// String feature = iwv.getItem().toString();
//			// sbTopFeatures.append(feature + "\t");
//			// }
//			// System.out.println(sbTopFeatures.toString().trim());
//			// }
//
//			// Test on the testing data on each domain.
//			for (Map.Entry<String, CrossValidationOperatorMaintainingLabelDistribution> entry : mpDomainToCVO
//					.entrySet()) {
//				String domain = entry.getKey();
//				CrossValidationOperatorMaintainingLabelDistribution cvo = entry
//						.getValue();
//				Documents testingDocsOneDomain = cvo
//						.getTestingDocuments(folderIndex);
//				for (Document testingDoc : testingDocsOneDomain) {
//					testingDoc.predict = classifierNB
//							.getBestCategoryByClassification(testingDoc.text);
//				}
//				ClassificationEvaluation evaluation = new ClassificationEvaluation(
//						testingDocsOneDomain.getLabels(),
//						testingDocsOneDomain.getPredicts());
//				if (!mpDomainToAccuracies.containsKey(domain)) {
//					mpDomainToAccuracies.put(domain, new double[noOfCVFolder]);
//				}
//				mpDomainToAccuracies.get(domain)[folderIndex] = evaluation.accuracy;
//			}
//		}
//		// Calculate the accuracy from fold cross-validation in each domain.
//		for (Map.Entry<String, double[]> entry : mpDomainToAccuracies
//				.entrySet()) {
//			String domain = entry.getKey();
//			double[] accuracies = entry.getValue();
//			double averageAccuracy = ArraysSumAndAverageAndMaxAndMin
//					.getAverage(accuracies);
//
//			System.out.println(domain + "\t"
//					+ String.format("%.2f", averageAccuracy));
//		}
//	}
//
//	// Train the model using training data on all domains except the
//	// target domain and test on testing data on the target domain.
//	private void trainOnAllDomainsExceptTargetAndTestOnTargetDomain(
//			Map<String, CrossValidationOperatorMaintainingLabelDistribution> mpDomainToCVO) {
//		for (Map.Entry<String, CrossValidationOperatorMaintainingLabelDistribution> entryTarget : mpDomainToCVO
//				.entrySet()) {
//			String domainTarget = entryTarget.getKey();
//			CrossValidationOperatorMaintainingLabelDistribution cvoTarget = entryTarget
//					.getValue();
//			double[] accuracies = new double[noOfCVFolder];
//			for (int folderIndex = 0; folderIndex < noOfCVFolder; ++folderIndex) {
//				// Merge training data from all domains except the target
//				// domain.
//				Documents trainingDocsAllDomainsExceptTargetDomain = new Documents();
//				for (Map.Entry<String, CrossValidationOperatorMaintainingLabelDistribution> entry : mpDomainToCVO
//						.entrySet()) {
//					String domain = entry.getKey();
//					if (domain.equals(domainTarget)) {
//						// Exclude the training data from the target domain.
//						continue;
//					}
//					CrossValidationOperatorMaintainingLabelDistribution cvo = entry
//							.getValue();
//					Documents trainingDocsOneDomain = cvo
//							.getTrainingDocuments(folderIndex);
//					trainingDocsAllDomainsExceptTargetDomain
//							.addDocuments(trainingDocsOneDomain);
//					break; // Only use the first domain.
//				}
//
//				// Train the model using merged training data on all domains.
//				// Build the naive Bayes model.
//				NaiveBayes_MyImplementation classifierNB = new NaiveBayes_MyImplementation();
//				for (Document trainingDoc : trainingDocsAllDomainsExceptTargetDomain) {
//					classifierNB.train(trainingDoc.text, trainingDoc.label);
//				}
//
//				// if (folderIndex == 0) {
//				// List<ItemWithValue> topPosFeatures = classifierNB
//				// .getTopPositiveFeaturesByProbOfClassGivenTokens(20);
//				// List<ItemWithValue> topNegFeatures = classifierNB
//				// .getTopNegativeFeaturesByProbOfClassGivenTokens(20);
//				// StringBuilder sbTopFeatures = new StringBuilder();
//				// for (ItemWithValue iwv : topNegFeatures) {
//				// String feature = iwv.getItem().toString();
//				// sbTopFeatures.append(feature + "\t");
//				// }
//				// System.out.println(sbTopFeatures.toString().trim());
//				// }
//
//				Documents testingDocsOneDomain = cvoTarget
//						.getTestingDocuments(folderIndex);
//				for (Document testingDoc : testingDocsOneDomain) {
//					testingDoc.predict = classifierNB
//							.getBestCategoryByClassification(testingDoc.text);
//				}
//				ClassificationEvaluation evaluation = new ClassificationEvaluation(
//						testingDocsOneDomain.getLabels(),
//						testingDocsOneDomain.getPredicts());
//				accuracies[folderIndex] = evaluation.accuracy;
//			}
//			double averageAccuracy = ArraysSumAndAverageAndMaxAndMin
//					.getAverage(accuracies);
//			System.out.println(domainTarget + "\t"
//					+ String.format("%.2f", averageAccuracy));
//		}
//	}
//}
