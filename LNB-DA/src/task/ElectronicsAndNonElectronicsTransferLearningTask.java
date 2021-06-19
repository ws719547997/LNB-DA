package task;

import java.util.List;

import classifier.NaiveBayes;
import nlp.Documents;

/**
 * Use the all the data from non-electronics as training data and test on each
 * electronics domain.
 * 
 * Use the all the data from electronics as training data and test on each
 * non-electronics domain.
 * 
 */
public class ElectronicsAndNonElectronicsTransferLearningTask {
	// Inputs.
	public String input1KReviewElectronicsDirectory = "..\\Data\\Input\\DifferentDomains\\50Electronics_100+100-\\";
	public String input1KReviewNonElectronicsDirectory = "..\\Data\\Input\\DifferentDomains\\50NonElectronics_100+100-\\";

	public double trainingDataPercentage = 0.2;

	public void run() {
		List<Documents> documentsListOfElectronicsDomain = Documents
				.readListOfDocumentsFromDifferentDomains(input1KReviewElectronicsDirectory);
		Documents documentsFromAllElectronicsDomain = new Documents();
		for (Documents documents : documentsListOfElectronicsDomain) {
			documentsFromAllElectronicsDomain.addDocuments(documents
					.selectSubsetOfDocumentsByOrder(trainingDataPercentage));
		}

		List<Documents> documentsListOfNonElectronicsDomain = Documents
				.readListOfDocumentsFromDifferentDomains(input1KReviewNonElectronicsDirectory);
		Documents documentsFromAllNonElectronicsDomain = new Documents();
		for (Documents documents : documentsListOfNonElectronicsDomain) {
			documentsFromAllNonElectronicsDomain.addDocuments(documents
					.selectSubsetOfDocumentsByOrder(trainingDataPercentage));
		}

		// Use the all the data from non-electronics as training data and test
		// on each electronics domain.
		NaiveBayes classifierNBNonElectronics = new NaiveBayes();
		classifierNBNonElectronics.train(documentsFromAllNonElectronicsDomain);
		for (Documents documentsOfE : documentsListOfElectronicsDomain) {
			String domain = documentsOfE.domain;
			double accuracy = classifierNBNonElectronics.test(documentsOfE).accuracy;
			System.out.println(domain + "\t" + accuracy);
		}

		// Use the all the data from electronics as training data and test on
		// each non-electronics domain.
		NaiveBayes classifierNBElectronics = new NaiveBayes();
		classifierNBElectronics.train(documentsFromAllElectronicsDomain);
		for (Documents documentsOfNonE : documentsListOfNonElectronicsDomain) {
			String domain = documentsOfNonE.domain;
			double accuracy = classifierNBElectronics.test(documentsOfNonE).accuracy;
			System.out.println(domain + "\t" + accuracy);
		}

	}
}
