package classifier;

import java.util.List;
import java.util.Map;

import classificationevaluation.ClassificationEvaluation;
import feature.FeatureSelection;
import utility.ExceptionUtility;
import utility.ItemWithValue;
import nlp.Document;
import nlp.Documents;

/**
 * This class supports for selecting classifier
 */
public abstract class BaseClassifier {
	public ClassifierParameters param = null;
	public FeatureSelection featureSelection = null;
	public Map<String, Map<String, double[]>> mpSelectedFeatureStrToDomainToProbs = null;
	public ClassificationKnowledge knowledge = null;
	public ClassificationKnowledge targetKnowledge = null;

	public static BaseClassifier selectClassifier(
			String classifierName,
			FeatureSelection featureSelection,
			Map<String, Map<String, double[]>> mpSelectedFeatureStrToDomainToProbs,
			ClassifierParameters param) {
		switch (classifierName) {
			case "NaiveBayes":
				return new NaiveBayes(featureSelection, param);
			case "LibSVM":
				return new LibSVM(featureSelection,
						mpSelectedFeatureStrToDomainToProbs, param);
			case "SVMLight":
				return new SVMLight(featureSelection,
						mpSelectedFeatureStrToDomainToProbs, param);
			case "LibLinear":
				return new LibLinear(featureSelection, param);
			default:
				ExceptionUtility
						.throwAndCatchException("The classifier name is not recognizable!");
				break;
		}
		return null;
	}

	public static BaseClassifier selectNewClassifier(String classifierName,
			FeatureSelection featureSelection,
			ClassificationKnowledge knowledge, ClassifierParameters param) {
		// Classic classifiers.
		switch (classifierName) {
			case "NaiveBayes_Sequence":
				return new NaiveBayes_Sequence(featureSelection, knowledge, param);
			case "KnowledgeableNB":
				return new Knowledgeable_NB(featureSelection, knowledge, param);
			case "NaiveBayes":
				return new NaiveBayes(featureSelection, param);
			case "LibLinear":
				return new LibLinear(featureSelection, param);
			case "LibSVM":
				return new LibSVM(featureSelection, param);
			case "SVMLight":
				return new SVMLight(featureSelection, null, param);
			case "LogisticRegressionFromLingpipe":
				return new LogisticRegressionFromLingpipe(featureSelection, param);
			case "MyLogisticRegression":
				return new MyLogisticRegression(featureSelection, param);
			case "LogisticRegressionFromMallet":
				return new LogisticRegressionFromMallet(featureSelection, param);
			case "MyLogisticRegressionFromMallet":
				return new MyLogisticRegressionFromMallet(featureSelection, param);


			// Proposed classifiers.
			case "NaiveBayes_Lifelong":
				return new NaiveBayes_Lifelong(featureSelection, knowledge, param);
			case "NaiveBayes_Lifelong_NewRatio":
				return new NaiveBayes_Lifelong_NewRatio(featureSelection,
						knowledge, param);
			case "NaiveBayes_Lifelong_WithoutCurrentDomainTraining":
				return new NaiveBayes_Lifelong_WithoutCurrentDomainTraining(
						featureSelection, knowledge, param);
			case "NaiveBayes_SGD_Lifelong":
			case "LSC_Stock":
				return new NaiveBayes_SGD_Lifelong(featureSelection, knowledge,
						param);
//			case "NaiveBayes_SGD_TargetDomainTrainingOnly":
//				return new NaiveBayes_SGD_TargetDomainTrainingOnly(
//						featureSelection, knowledge, param);
			default:
				ExceptionUtility
						.throwAndCatchException("The classifier name is not recognizable!");
				break;
		}
		return null;
	}

	public static BaseClassifier selectGobackClassifier(String classifierName,
														FeatureSelection featureSelection,
														Map<String, ClassificationKnowledge> pastKnowledgeList,
														ClassifierParameters param) {
		// Classic classifiers.
		switch (classifierName) {
			case "NaiveBayes_Sequence_GoBack":
				return new NaiveBayes_Sequence_GoBack(featureSelection, pastKnowledgeList, param);
			case "NaiveBayes_AddPastDomain":
				return new NaiveBayes(featureSelection, param);
			case "LibSVM":
				return new LibSVM(featureSelection, param);
			case "SVMLight":
				return new SVMLight(featureSelection, null, param);
			case "LibLinear":
				return new LibLinear(featureSelection, param);
			default:
				ExceptionUtility
						.throwAndCatchException("The classifier name is not recognizable!");
				break;
		}
		return null;
	}

	public abstract void train(Documents trainingDocs);

	public abstract ClassificationEvaluation test(Documents testingDocs);

	public List<ItemWithValue> getFeaturesByRatio(Document testingDoc) {
		return null;
	}

	public abstract void printMisclassifiedDocuments(Documents testingDocs,
													 String misclassifiedDocumentsForOneCVFolderForOneDomainFilePath);

	public double[] getCountsOfClasses(String featureStr) {
		return null;
	}
}
