package classificationevaluation;

import nlp.Label;
import utility.ExceptionUtility;

public class ClassificationEvaluation {
	public int truePositive = 0;
	public int trueNegative = 0;
	public int falsePositive = 0;
	public int falseNegative = 0;

	public double precision = 0.0;
	public double recall = 0.0;
	public double f1score = 0.0;

	public double precisionNegativeClass = 0.0;
	public double recallNegativeClass = 0.0;
	public double f1scoreNegativeClass = 0.0;

	public double f1scoreBothClasses = 0.0;
	public double accuracy = 0.0;

	public double logloss = 0.0;

	public String domain = null;

	public ClassificationEvaluation() {
	}

	public ClassificationEvaluation(String[] labels, String[] predicts,
			String domain2) {
		ExceptionUtility.assertAsException(labels.length == predicts.length);
		domain = domain2;

		// Compute the precision, recall, F1-score and accuracy.
		for (int i = 0; i < labels.length; ++i) {
			String label = labels[i];
			String predict = predicts[i];
			if (label.equals(predict)) {
				if (Label.isPositive(predict)) {
					++truePositive;
				} else {
					++trueNegative;
				}
			} else {
				if (Label.isPositive(predict)) {
					++falsePositive;
				} else {
					++falseNegative;
				}
			}
		}

		if (truePositive + falsePositive == 0) {
			precision = 0;
		} else {
			precision = 1.0 * truePositive / (truePositive + falsePositive);
		}

		if (truePositive + falseNegative == 0) {
			recall = 0;
		} else {
			recall = 1.0 * truePositive / (truePositive + falseNegative);
		}

		if (precision + recall == 0) {
			f1score = 0;
		} else {
			f1score = 2.0 * precision * recall / (precision + recall);
		}

		// For negative class.
		if (trueNegative + falseNegative == 0) {
			precisionNegativeClass = 0;
		} else {
			precisionNegativeClass = 1.0 * trueNegative
					/ (trueNegative + falseNegative);
		}

		if (trueNegative + falsePositive == 0) {
			recallNegativeClass = 0;
		} else {
			recallNegativeClass = 1.0 * trueNegative
					/ (trueNegative + falsePositive);
		}

		if (precisionNegativeClass + recallNegativeClass == 0) {
			f1scoreNegativeClass = 0;
		} else {
			f1scoreNegativeClass = 2.0 * precisionNegativeClass
					* recallNegativeClass
					/ (precisionNegativeClass + recallNegativeClass);
		}

		f1scoreBothClasses = (f1score + f1scoreNegativeClass) / 2.0;

		accuracy = 1.0 * (truePositive + trueNegative)
				/ (truePositive + trueNegative + falsePositive + falseNegative);
	}

	public void updateLogLoss(String[] labels, double[] probsOfPositive,
			double[] probsOfNegative) {
		for (int i = 0; i < labels.length; ++i) {
			if (labels[i].startsWith("+")) {
				logloss += -Math.log(probsOfPositive[i]);
			} else {
				logloss += -Math.log(probsOfNegative[i]);
			}
		}
	}

	public void updateLogLoss(String label, double probOfPositivePredict,
			double probOfNegativePredict) {
		if (label.startsWith("+")) {
			logloss += -Math.log(probOfPositivePredict);
		} else {
			logloss += -Math.log(probOfNegativePredict);
		}
	}

	@Override
	public String toString() {
		StringBuilder sbEval = new StringBuilder();
		sbEval.append("Precision (Positive): " + precision + "\n");
		sbEval.append("Recall (Positive): " + recall + "\n");
		sbEval.append("F1-Score (Positive): " + f1score + "\n");
		sbEval.append("Accuracy : " + accuracy + "\n");
		return sbEval.toString();
	}

}
