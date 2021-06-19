package classificationevaluation;

import java.util.ArrayList;
import java.util.List;

import utility.ExceptionUtility;

public class ClassificationEvaluationAccumulator {
	public String domain = null;
	public List<Double> precisionList = null;
	public List<Double> recallList = null;
	public List<Double> f1scoreList = null;
	public List<Double> precisionNegativeClassList = null;
	public List<Double> recallNegativeClassList = null;
	public List<Double> f1scoreNegativeClassList = null;
	public List<Double> f1scoreBothClassesList = null;
	public List<Double> accuracyList = null;
	public List<Double> loglossList = null;

	public ClassificationEvaluationAccumulator() {
		precisionList = new ArrayList<Double>();
		recallList = new ArrayList<Double>();
		f1scoreList = new ArrayList<Double>();
		precisionNegativeClassList = new ArrayList<Double>();
		recallNegativeClassList = new ArrayList<Double>();
		f1scoreNegativeClassList = new ArrayList<Double>();
		f1scoreBothClassesList = new ArrayList<Double>();
		accuracyList = new ArrayList<Double>();
		loglossList = new ArrayList<Double>();
	}

	public void addClassificationEvaluation(ClassificationEvaluation evaluation) {
		if (domain == null) {
			domain = evaluation.domain;
		} else {
			ExceptionUtility
					.assertAsException(domain.equals(evaluation.domain));
		}
		precisionList.add(evaluation.precision);
		recallList.add(evaluation.recall);
		f1scoreList.add(evaluation.f1score);
		precisionNegativeClassList.add(evaluation.precisionNegativeClass);
		recallNegativeClassList.add(evaluation.recallNegativeClass);
		f1scoreNegativeClassList.add(evaluation.f1scoreNegativeClass);
		f1scoreBothClassesList.add(evaluation.f1scoreBothClasses);
		accuracyList.add(evaluation.accuracy);
		loglossList.add(evaluation.logloss);
	}

	public ClassificationEvaluation getAverageClassificationEvaluation() {
		ClassificationEvaluation evaluation = new ClassificationEvaluation();
		evaluation.domain = this.domain;
		evaluation.precision = getAverage(this.precisionList);
		evaluation.recall = getAverage(this.recallList);
		evaluation.f1score = getAverage(this.f1scoreList);
		evaluation.precisionNegativeClass = getAverage(this.precisionNegativeClassList);
		evaluation.recallNegativeClass = getAverage(this.recallNegativeClassList);
		evaluation.f1scoreNegativeClass = getAverage(this.f1scoreNegativeClassList);
		evaluation.f1scoreBothClasses = getAverage(this.f1scoreBothClassesList);
		evaluation.accuracy = getAverage(this.accuracyList);
		evaluation.logloss = getAverage(this.loglossList);
		return evaluation;
	}

	private double getAverage(List<Double> list) {
		double sum = 0.0;
		for (double v : list) {
			sum += v;
		}
		return sum / list.size();
	}
}
