package task;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import classificationevaluation.ClassificationEvaluation;
import classificationevaluation.ClassificationEvaluationAccumulator;
import classifier.NaiveBayes;
import utility.CrossValidationOperatorMaintainingLabelDistribution;
import utility.FileReaderAndWriter;
import utility.Pair;
import main.CmdOption;
import nlp.Documents;

/**
 * For each pair of domains (say A and B). The training data is all the labeled
 * data from domain A. The testing data is all the data from domain B.
 */
public class PariwiseDomainTransferLearningTask {
	// Inputs.
	public String input1KReviewElectronicsDirectory = "..\\Data\\Input\\DifferentDomains\\50Electronics_100+100-\\";
	public String input1KReviewNonElectronicsDirectory = "..\\Data\\Input\\DifferentDomains\\50NonElectronics_100+100-\\";
	public String inputSameDomainDifferntProductsDirectoy = "..\\Data\\Input\\SameDomainDifferentProducts\\";
	// Outputs.
	public String outputPairwiseDomainTransferLearningFilepath = "..\\Data\\Output\\PairwiseDomainTransferLearning\\Accuracy.txt";

	public CmdOption cmdOption = null;

	public PariwiseDomainTransferLearningTask(CmdOption cmdOption2) {
		cmdOption = cmdOption2;
	}

	public void run() {
		List<Documents> documentsOfAllDomains = new ArrayList<Documents>();
		double[][] accuracyOfDomainAToDomainB = null;

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

		accuracyOfDomainAToDomainB = new double[noOfDomains][noOfDomains];
		for (int i = 0; i < documentsOfAllDomains.size(); ++i) {
			Documents documentsDomainA = documentsOfAllDomains.get(i);
			String domainA = documentsDomainA.domain;
			System.out.print(domainA + " : ");
			NaiveBayes classiferNB = new NaiveBayes();
			classiferNB.train(documentsDomainA);
			for (int j = 0; j < documentsOfAllDomains.size(); ++j) {
				Documents documentsDomainB = documentsOfAllDomains.get(j);
				String domainB = documentsDomainB.domain;
				System.out.print(domainB + " | ");
				ClassificationEvaluation evaluation = null;
				if (i == j) {
					// Single domain for training and testing.
					evaluation = this.getClassificationEvaluation(
							documentsDomainA,
							cmdOption.noOfCrossValidationFolders);
				} else {
					evaluation = classiferNB.test(documentsDomainB);

				}
				double accuracy = evaluation.accuracy;

				accuracyOfDomainAToDomainB[i][j] = accuracy;
			}
			System.out.println();
		}
		outputResultMatrix(documentsOfAllDomains, accuracyOfDomainAToDomainB,
				outputPairwiseDomainTransferLearningFilepath);
	}

	public ClassificationEvaluation getClassificationEvaluation(
			Documents documents, int noOfCVFolder) {
		ClassificationEvaluationAccumulator evaluationAccumulator = new ClassificationEvaluationAccumulator();
		CrossValidationOperatorMaintainingLabelDistribution cvo = new CrossValidationOperatorMaintainingLabelDistribution(
				documents, noOfCVFolder);
		for (int folderIndex = 0; folderIndex < noOfCVFolder; ++folderIndex) {
			Pair<Documents, Documents> pair = cvo
					.getTrainingAndTestingDocuments(folderIndex, noOfCVFolder);
			Documents trainingDocs = pair.t;
			Documents testingDocs = pair.u;

			// Build the naive Bayes model.
			NaiveBayes classifierNB = new NaiveBayes();
			classifierNB.train(trainingDocs);

			ClassificationEvaluation evaluation = classifierNB
					.test(testingDocs);
			evaluationAccumulator.addClassificationEvaluation(evaluation);
		}
		return evaluationAccumulator.getAverageClassificationEvaluation();
	}

	public void outputResultMatrix(List<Documents> documentsOfAllDomains,
			double[][] matrix, String outputFilepath) {
		// Output the results to file.
		boolean[][] isMaximumOfEachColumnMatrix = getIsMaximumOfEachColumnMatrix(matrix);
		StringBuilder sbOutput = new StringBuilder();
		StringBuilder sbFirstLine = new StringBuilder();
		sbFirstLine.append("Domain/Accuracy");
		for (int i = 0; i < documentsOfAllDomains.size(); ++i) {
			Documents documentsDomainA = documentsOfAllDomains.get(i);
			String domainA = documentsDomainA.domain;
			sbFirstLine.append("\t" + domainA);
		}
		sbOutput.append(sbFirstLine.toString().trim());
		sbOutput.append(System.lineSeparator());

		for (int i = 0; i < documentsOfAllDomains.size(); ++i) {
			StringBuilder sbLine = new StringBuilder();
			Documents documentsDomainA = documentsOfAllDomains.get(i);
			String domainA = documentsDomainA.domain;
			sbLine.append(domainA + "\t");
			for (int j = 0; j < documentsOfAllDomains.size(); ++j) {
				double accuracy = matrix[i][j];
				if (isMaximumOfEachColumnMatrix[i][j]) {
					// Highlight the max value of each row.
					sbLine.append("MAX ");
				}
				sbLine.append(accuracy + "\t");
			}
			sbOutput.append(sbLine.toString().trim());
			sbOutput.append(System.lineSeparator());
		}
		FileReaderAndWriter.writeFile(outputFilepath, sbOutput.toString());
	}

	private boolean[][] getIsMaximumOfEachColumnMatrix(double[][] matrix) {
		boolean[][] isMaximumOfEachColumnMatrix = new boolean[matrix.length][matrix[0].length];
		for (int column = 0; column < matrix[0].length; ++column) {
			double maximum = Double.MIN_VALUE;
			for (int row = 0; row < matrix.length; ++row) {
				maximum = Math.max(maximum, matrix[row][column]);
			}
			for (int row = 0; row < matrix.length; ++row) {
				if (Math.abs(maximum - matrix[row][column]) < 1e-10) {
					isMaximumOfEachColumnMatrix[row][column] = true;
				} else {
					isMaximumOfEachColumnMatrix[row][column] = false;
				}
			}
		}
		return isMaximumOfEachColumnMatrix;
	}
}
