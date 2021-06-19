//package task;
//
//import java.io.File;
//import java.io.FilenameFilter;
//import java.util.List;
//import java.util.Map;
//import java.util.TreeMap;
//
//import classificationevaluation.ClassificationEvaluation;
//import classifier.NaiveBayes_MyImplementation;
//import utility.ArraysSumAndAverageAndMaxAndMin;
//import utility.CrossValidationOperatorMaintainingLabelDistribution;
//import utility.ItemWithValue;
//import utility.Pair;
//import nlp.Document;
//import nlp.Documents;
//
//public class Old_SentimentClassificationSameDomainDifferentProductsTask {
//	public String inputReviewDirectory = "..\\Data\\Input\\SameDomainDifferentProducts";
//	public int noOfCVFolder = 10;
//
//	public void run() {
//		try {
//			// Go through each domain.
//			String[] domainNames = new File(inputReviewDirectory)
//					.list(new FilenameFilter() {
//						@Override
//						public boolean accept(File current, String name) {
//							return new File(current, name).isDirectory();
//						}
//					});
//			for (String domain : domainNames) {
//				String domainDirectoryPath = inputReviewDirectory
//						+ File.separator + domain;
//				runOnOneDomain(domainDirectoryPath, domain);
//			}
//		} catch (Exception ex) {
//			ex.printStackTrace();
//		}
//	}
//
//	public void runOnOneDomain(String directoryPath, String domain) {
//		Map<String, Documents> mpProductIdToDocuments = new TreeMap<String, Documents>();
//		Map<String, CrossValidationOperatorMaintainingLabelDistribution> mpProductIdToCVO = new TreeMap<String, CrossValidationOperatorMaintainingLabelDistribution>();
//
//		File[] productFiles = new File(directoryPath).listFiles();
//		for (File productFile : productFiles) {
//			String productId = productFile.getName().replaceFirst("[.][^.]+$",
//					"");
//			String domainProductId = domain + "_" + productId;
//			Documents documents = Documents.readDocuments(domainProductId,
//					productFile.getAbsolutePath());
//			mpProductIdToDocuments.put(productId, documents);
//			CrossValidationOperatorMaintainingLabelDistribution cvo = new CrossValidationOperatorMaintainingLabelDistribution(
//					documents, noOfCVFolder);
//			mpProductIdToCVO.put(productId, cvo);
//		}
//
//		// Train the model using training data on one domain and test
//		// on testing data on the same domain.
//		// trainAndTestOnSameDomain(mpProductIdToCVO);
//
//		// Train the model using training data on all domains and test
//		// on testing data on each domain.
//		trainOnAllDomainsAndTestOnEachDomain(mpProductIdToCVO);
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
//				if (folderIndex == 0) {
//					List<ItemWithValue> topPosFeatures = classifierNB
//							.getTopPositiveFeaturesByProbOfClassGivenTokens(20);
//					List<ItemWithValue> topNegFeatures = classifierNB
//							.getTopNegativeFeaturesByProbOfClassGivenTokens(20);
//					StringBuilder sbTopFeatures = new StringBuilder();
//					sbTopFeatures.append(domain + "\t");
//					for (ItemWithValue iwv : topNegFeatures) {
//						String feature = iwv.getItem().toString();
//						sbTopFeatures.append(feature + "\t");
//					}
//					System.out.println(sbTopFeatures.toString().trim());
//				}
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
//			double averageAccuracy = ArraysSumAndAverageAndMaxAndMin.getAverage(accuracies);
//			// System.out.println(domain + "\t"
//			// + String.format("%.2f", averageAccuracy));
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
//			if (folderIndex == 0) {
//				List<ItemWithValue> topPosFeatures = classifierNB
//						.getTopPositiveFeaturesByProbOfClassGivenTokens(50);
//				List<ItemWithValue> topNegFeatures = classifierNB
//						.getTopNegativeFeaturesByProbOfClassGivenTokens(50);
//				StringBuilder sbTopFeatures = new StringBuilder();
//				// sbTopFeatures.append(domain + "\t");
//				for (ItemWithValue iwv : topNegFeatures) {
//					String feature = iwv.getItem().toString();
//					sbTopFeatures.append(feature + "\t");
//				}
//				System.out.println(sbTopFeatures.toString().trim());
//			}
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
//			double averageAccuracy = ArraysSumAndAverageAndMaxAndMin.getAverage(accuracies);
//
//			// System.out.println(domain + "\t"
//			// + String.format("%.2f", averageAccuracy));
//		}
//	}
//}
