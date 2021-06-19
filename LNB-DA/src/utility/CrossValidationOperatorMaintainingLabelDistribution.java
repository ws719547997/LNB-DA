package utility;

import nlp.Document;
import nlp.Documents;

/**
 * A simple implementation that only works for two label classes.
 */
public class CrossValidationOperatorMaintainingLabelDistribution {
	private CrossValidationOperator cvoPositive = null;
	private CrossValidationOperator cvoNegative = null;

    /**
     * deal with positive and negative document separately to maintaining label distribution
     * @param documents
     * @param noOfFolder
     */
	public CrossValidationOperatorMaintainingLabelDistribution(
			Documents documents, int noOfFolder) {
		Documents positiveDocuments = new Documents();
		Documents negativeDocuments = new Documents();
		for (Document document : documents) {
			if (document == null || document.label == null) {
				System.out.println("document.label == null");
			}
			// separate documents into POS and NEG
			if (document.label.startsWith("+")) {
				positiveDocuments.addDocument(document);
			} else {
				negativeDocuments.addDocument(document);
			}
		}
		cvoPositive = new CrossValidationOperator(positiveDocuments);
		cvoNegative = new CrossValidationOperator(negativeDocuments);
	}

	/**
	 * get training and test documents
	 * @param folderIndex
	 * @return
	 */
	public Pair<Documents, Documents> getTrainingAndTestingDocuments(
			int folderIndex, int noOfCVFolder) {
		Pair<Documents, Documents> trainingAndTestingPositive = cvoPositive
				.getTrainingAndTestingDocuments(folderIndex, noOfCVFolder);
		Pair<Documents, Documents> trainingAndTestingNegative = cvoNegative
				.getTrainingAndTestingDocuments(folderIndex, noOfCVFolder);

		// training docs: POS first, then NEG
		Documents trainingDocs = trainingAndTestingPositive.t;
		trainingDocs.addDocuments(trainingAndTestingNegative.t);
		// testing docs: POS first, then NEG
		Documents testingDocs = trainingAndTestingPositive.u;
		testingDocs.addDocuments(trainingAndTestingNegative.u);

		return new Pair<Documents, Documents>(trainingDocs, testingDocs); // where Pair.t is trainingDocs,
																			// Pair.u is testingDocs
	}

	/**
	 * get training documents
	 * @param folderIndex
	 * @return
	 */
	public Documents getTrainingDocuments(int folderIndex, int noOfCVFolder) {
		Pair<Documents, Documents> trainingAndTestingPositive = cvoPositive
				.getTrainingAndTestingDocuments(folderIndex, noOfCVFolder);
		Pair<Documents, Documents> trainingAndTestingNegative = cvoNegative
				.getTrainingAndTestingDocuments(folderIndex, noOfCVFolder);

		Documents trainingDocs = trainingAndTestingPositive.t;
		trainingDocs.addDocuments(trainingAndTestingNegative.t);

		return trainingDocs;
	}

	/**
	 * get testing documents
	 * @param folderIndex
	 * @return
	 */
	public Documents getTestingDocuments(int folderIndex, int noOfCVFolder) {
		Pair<Documents, Documents> trainingAndTestingPositive = cvoPositive
				.getTrainingAndTestingDocuments(folderIndex, noOfCVFolder);
		Pair<Documents, Documents> trainingAndTestingNegative = cvoNegative
				.getTrainingAndTestingDocuments(folderIndex, noOfCVFolder);

		Documents testingDocs = trainingAndTestingPositive.u;
		testingDocs.addDocuments(trainingAndTestingNegative.u);

		return testingDocs;
	}

}
