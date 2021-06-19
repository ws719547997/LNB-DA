package task;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import classificationevaluation.ClassificationEvaluation;
import classifier.NaiveBayes;
import main.CmdOption;
import nlp.Documents;

/**
 * For each pair of domain A, we accumulate training data from domains except A
 * sequentially (simulating the lifelong learning scenario).
 */
public class SequentialDomainTransferLearningTask {
	// Inputs.
	public String input1KReviewElectronicsDirectory = "..\\Data\\Input\\DifferentDomains\\50Electronics_100+100-\\";
	public String input1KReviewNonElectronicsDirectory = "..\\Data\\Input\\DifferentDomains\\50NonElectronics_100+100-\\";
	public String inputSameDomainDifferntProductsDirectoy = "..\\Data\\Input\\SameDomainDifferentProducts\\";
	// Outputs.
	public String outputSequentialDomainTransferLearningFilepath = "..\\Data\\Output\\SequentialDomainTransferLearning\\Accuracy.txt";

	private List<Documents> documentsOfAllDomains = null;
	double[][] accuracyMatrix = null;

	public CmdOption cmdOption = null;

	public SequentialDomainTransferLearningTask(CmdOption cmdOption2) {
		cmdOption = cmdOption2;
	}

	public void run() {
		documentsOfAllDomains = new ArrayList<Documents>();

		// Electronics.
		// documentsOfAllDomains
		// .addAll(Documents
		// .readListOfDocumentsFromDifferentDomains(input1KReviewElectronicsDirectory));

		// Non-Electronics.
		// documentsOfAllDomains
		// .addAll(Documents
		// .readListOfDocumentsFromDifferentDomains(input1KReviewNonElectronicsDirectory));

		// Same domain different products.
		String productDomain = "Dining";
		documentsOfAllDomains
				.addAll(Documents
						.readListOfDocumentsFromDifferentDomains(inputSameDomainDifferntProductsDirectoy
								+ productDomain + File.separator));

		int noOfDomains = documentsOfAllDomains.size();

		accuracyMatrix = new double[noOfDomains][noOfDomains];
		for (int j = 0; j < documentsOfAllDomains.size(); ++j) {
			Documents documentsDomainB = documentsOfAllDomains.get(j);
			String domainB = documentsDomainB.domain;
			System.out.print(domainB + " : ");
			NaiveBayes classiferNB = new NaiveBayes();
			for (int i = 0; i < documentsOfAllDomains.size(); ++i) {
				Documents documentsDomainA = documentsOfAllDomains.get(i);
				String domainA = documentsDomainA.domain;
				System.out.print(domainA + " | ");
				ClassificationEvaluation evaluation = null;
				if (i == j) {
					// Exclude the training from the same domain. Mark the
					// accuracy of this scenario as 0.
					evaluation = new ClassificationEvaluation();
				} else {
					classiferNB.train(documentsDomainA);
					evaluation = classiferNB.test(documentsDomainB);
				}
				double accuracy = evaluation.accuracy;

				accuracyMatrix[i][j] = accuracy;
			}
			System.out.println();
		}
		PariwiseDomainTransferLearningTask task = new PariwiseDomainTransferLearningTask(
				cmdOption);
		task.outputResultMatrix(documentsOfAllDomains, accuracyMatrix,
				outputSequentialDomainTransferLearningFilepath);
	}

}
