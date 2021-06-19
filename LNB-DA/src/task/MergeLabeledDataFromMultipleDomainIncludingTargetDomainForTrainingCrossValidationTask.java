package task;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import main.CmdOption;
import multithread.SentimentClassificationThreadPool;
import nlp.Documents;
import utility.ArraySumAndAverageAndMaxAndMin;
import utility.FileReaderAndWriter;
import classificationevaluation.ClassificationEvaluation;
import classifier.ClassifierParameters;

/**
 * The training and testing data are all from a single domain using fold cross
 * validation.
 */
public class MergeLabeledDataFromMultipleDomainIncludingTargetDomainForTrainingCrossValidationTask {
	public CmdOption cmdOption = null;
	private List<String> domainsToEvaluate = null;

	public MergeLabeledDataFromMultipleDomainIncludingTargetDomainForTrainingCrossValidationTask(
			CmdOption cmdOption2) {
		cmdOption = cmdOption2;

		if (cmdOption.inputListOfDomainsToEvaluate != null) {
			domainsToEvaluate = FileReaderAndWriter
					.readFileAllLines(cmdOption.inputListOfDomainsToEvaluate);
		}
	}

	public Map<String, ClassificationEvaluation> run() {
		List<Documents> documentsOfAllDomains = readDocuments();

		SentimentClassificationThreadPool threadPool = new SentimentClassificationThreadPool(cmdOption.nthreads);
		for (int domain_id = 0; domain_id < domainsToEvaluate.size(); ++domain_id) {
			String targetDomain = domainsToEvaluate.get(domain_id);
			// get documents of target domain -> targetDocs
			Documents documents = null;
			assert documentsOfAllDomains != null;
			for (int j = 0; j < documentsOfAllDomains.size(); ++j) {
				if (Objects.equals(targetDomain, documentsOfAllDomains.get(j).domain)) {
					documents = documentsOfAllDomains.get(j).getDeepClone();
					break;
				}
			}

			// only testing one domain
//			if (!domain.equals("CombAutomotive0-9")) {
//				continue;
//			}

			ClassifierParameters param = new ClassifierParameters(documents, cmdOption);

			List<Documents> documentsOfOtherDomains = new ArrayList<Documents>();
			if (param.includeSourceDomainsLabeledDataForTraining) {
				Documents addDocuments = new Documents();
				for (int j = 0; j < domainsToEvaluate.size(); ++j) {
					String addDomain = domainsToEvaluate.get(j);
					if (domain_id == j) {
						continue;
					}
					for (int jj = 0; jj < documentsOfAllDomains.size(); ++jj) {
						if (Objects.equals(addDomain, documentsOfAllDomains.get(jj).domain)) {
							addDocuments = documentsOfAllDomains.get(jj).getDeepClone();
							break;
						}
					}
					documentsOfOtherDomains.add(addDocuments.getDeepClone());
				}
			}

			threadPool.addTask(documents, documentsOfOtherDomains, param);
		}
		threadPool.awaitTermination();
		Map<String, ClassificationEvaluation> mpDomainToClassificationEvaluation = threadPool.mpClassificationEvaluation;
		// Accuracy.
		StringBuilder sbOutput = new StringBuilder();
		double[] accuracies = new double[mpDomainToClassificationEvaluation.size()];
		int i = 0;
		for (Map.Entry<String, ClassificationEvaluation> entry : mpDomainToClassificationEvaluation
				.entrySet()) {
			ClassificationEvaluation evaluation = entry.getValue();
			String domain = evaluation.domain;
			sbOutput.append(domain + "\t" + evaluation.accuracy);
//			sbOutput.append(evaluation.accuracy);
			sbOutput.append(System.lineSeparator());
			accuracies[i++] = evaluation.accuracy;
		}
		FileReaderAndWriter.writeFile(cmdOption.outputSentimentClassificationAccuracy
						+ "/" + "NaiveBayes" + "/" + "ACC_NB.txt",
				sbOutput.toString());
		System.out.println("Average Accuracy: "
				+ ArraySumAndAverageAndMaxAndMin.getAverage(accuracies));
		// F1-score in both classes.
		sbOutput = new StringBuilder();
		double[] f1Scores = new double[mpDomainToClassificationEvaluation
				.size()];
		i = 0;
		for (Map.Entry<String, ClassificationEvaluation> entry : mpDomainToClassificationEvaluation
				.entrySet()) {
			ClassificationEvaluation evaluation = entry.getValue();
			// String domain = evaluation.domain;
			// sbOutput.append(domain + "\t" + evaluation.accuracy);
			sbOutput.append(evaluation.f1scoreBothClasses);
			sbOutput.append(System.lineSeparator());
			f1Scores[i++] = evaluation.f1scoreBothClasses;
		}
		FileReaderAndWriter.writeFile(cmdOption.outputSentimentClassificationF1Score
				+ "/" + "NaiveBayes" + "/" + "F1_BothClasses_NB.txt", sbOutput.toString());
		System.out.println("Average F1-score (Both Classes): "
				+ ArraySumAndAverageAndMaxAndMin.getAverage(f1Scores));

		// F1-score in the positive class.
		sbOutput = new StringBuilder();
		f1Scores = new double[mpDomainToClassificationEvaluation.size()];
		i = 0;
		for (Map.Entry<String, ClassificationEvaluation> entry : mpDomainToClassificationEvaluation
				.entrySet()) {
			ClassificationEvaluation evaluation = entry.getValue();
			// String domain = evaluation.domain;
			// sbOutput.append(domain + "\t" + evaluation.accuracy);
			sbOutput.append(evaluation.f1score);
			sbOutput.append(System.lineSeparator());
			f1Scores[i++] = evaluation.f1score;
		}
		FileReaderAndWriter
				.writeFile(cmdOption.outputSentimentClassificationF1Score
						+ "/" + "NaiveBayes" + "/" + "F1_Positive_NB.txt", sbOutput.toString());
		System.out.println("Average F1-score (Positive): "
				+ ArraySumAndAverageAndMaxAndMin.getAverage(f1Scores));

		// F1-score in the negative class.
		sbOutput = new StringBuilder();
		f1Scores = new double[mpDomainToClassificationEvaluation.size()];
		i = 0;
		for (Map.Entry<String, ClassificationEvaluation> entry : mpDomainToClassificationEvaluation
				.entrySet()) {
			ClassificationEvaluation evaluation = entry.getValue();
			// String domain = evaluation.domain;
			// sbOutput.append(domain + "\t" + evaluation.accuracy);
			sbOutput.append(evaluation.f1scoreNegativeClass);
			sbOutput.append(System.lineSeparator());
			f1Scores[i++] = evaluation.f1scoreNegativeClass;
		}
		FileReaderAndWriter
				.writeFile(cmdOption.outputSentimentClassificationF1Score
						+ "/" + "NaiveBayes" + "/" + "F1_Negative_NB.txt", sbOutput.toString());
		System.out.println("Average F1-score (Negative): "
				+ ArraySumAndAverageAndMaxAndMin.getAverage(f1Scores));

		return mpDomainToClassificationEvaluation;
	}

	/**
	 * According to the configuration, we read documents from different
	 * directories.
	 */
	private List<Documents> readDocuments() {
		InputReaderTask task = new InputReaderTask(cmdOption);
		switch (cmdOption.datasetName) {
			case "100P100NDomains":
				return task.readDocumentsListFrom100P100NDomains();
			case "Reuters10":
				return task.readDocumentsFromstock();
			case "20Newgroup":
				return task.read20Newsgroup();
			case "PangAndLeeMovieReviews":
				return task.readDocumentsFromPangAndLeeMovieReview();
			case "1KP1KNDomains":
				return task.readDocumentsListFrom1KP1KNDomains();
			case "1KReviewNaturalClassDistributionDomains":
				return task.readDocumentsListFrom1KReviewsNaturalClassDistributionDomains();
			case "DifferentProductsOfSameDomain":
				return task.readDocumentsFromDifferentProductsOfSameDomain();
			case "BalancedWithMostNegativeReviews":
				return task.readDocumentsFromBalancedWithMostNegativeReviews();
		}
		return null;
	}

}
