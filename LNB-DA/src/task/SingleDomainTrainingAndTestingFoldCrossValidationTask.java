package task;

import java.util.List;
import java.util.Map;

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
public class SingleDomainTrainingAndTestingFoldCrossValidationTask {
	public CmdOption cmdOption = null;

	public SingleDomainTrainingAndTestingFoldCrossValidationTask(
			CmdOption cmdOption2) {
		cmdOption = cmdOption2;
	}

	public Map<String, ClassificationEvaluation> run() {
		List<Documents> documentsOfAllDomains = readDocuments();

		SentimentClassificationThreadPool threadPool = new SentimentClassificationThreadPool(
				cmdOption.nthreads);
		for (int i = 0; i < documentsOfAllDomains.size(); ++i) {
			Documents documents = documentsOfAllDomains.get(i);

			ClassifierParameters param = new ClassifierParameters(documents,
					cmdOption);

			threadPool.addTask(documents, param);
		}
		threadPool.awaitTermination();
		Map<String, ClassificationEvaluation> mpDomainToClassificationEvaluation = threadPool.mpClassificationEvaluation;
		StringBuilder sbAccuracyOutput = new StringBuilder();
		StringBuilder sbF1ScoreOutput = new StringBuilder();
		double[] accuracies = new double[mpDomainToClassificationEvaluation
				.size()];
		double[] f1scores = new double[mpDomainToClassificationEvaluation
				.size()];
		int i = 0;
		for (Map.Entry<String, ClassificationEvaluation> entry : mpDomainToClassificationEvaluation
				.entrySet()) {
			ClassificationEvaluation evaluation = entry.getValue();
			String domain = evaluation.domain;
			sbAccuracyOutput.append(domain + "\t" + evaluation.accuracy);
			// sbAccuracyOutput.append(evaluation.accuracy);
			sbAccuracyOutput.append(System.lineSeparator());
			sbF1ScoreOutput.append(evaluation.f1score);
			sbF1ScoreOutput.append(System.lineSeparator());
			accuracies[i] = evaluation.accuracy;
			f1scores[i] = evaluation.f1score;
			i++;
		}
		FileReaderAndWriter.writeFile(
				cmdOption.outputSentimentClassificationAccuracy,
				sbAccuracyOutput.toString());
		FileReaderAndWriter.writeFile(
				cmdOption.outputSentimentClassificationF1Score,
				sbF1ScoreOutput.toString());
		System.out.println("Average Accuracy: "
				+ ArraySumAndAverageAndMaxAndMin.getAverage(accuracies));
		System.out.println("Average F1-Score: "
				+ ArraySumAndAverageAndMaxAndMin.getAverage(f1scores));
		return mpDomainToClassificationEvaluation;
	}

	/**
	 * According to the configuration, we read documents from different
	 * directories.
	 */
	private List<Documents> readDocuments() {
		InputReaderTask task = new InputReaderTask(cmdOption);
		switch (cmdOption.datasetName) {
			case "Reuters10":
				task.readReuters10domains();
				break;
			case "20Newgroup":
				return task.read20Newsgroup();
			case "PangAndLeeMovieReviews":
				return task.readDocumentsFromPangAndLeeMovieReview();
			case "100P100NDomains":
				return task.readDocumentsListFrom100P100NDomains();
			case "1KP1KNDomains":
				return task.readDocumentsListFrom1KP1KNDomains();
			case "DifferentProductsOfSameDomain":
				return task.readDocumentsFromDifferentProductsOfSameDomain();
		}
		return null;
	}
}
