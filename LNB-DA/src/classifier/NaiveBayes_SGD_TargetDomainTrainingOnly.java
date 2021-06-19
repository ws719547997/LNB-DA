//package classifier;
//
//import java.util.List;
//
//import utility.ItemWithValue;
//import classificationevaluation.ClassificationEvaluation;
//import feature.FeatureSelection;
//import nlp.Document;
//import nlp.Documents;
//
///**
// * Implement the new model using stochastic gradient descent.
// *
// * Only used training data from the target domain only.
// *
// * No feature selection.
// */
//public class NaiveBayes_SGD_TargetDomainTrainingOnly extends BaseClassifier {
//	private NaiveBayes_SGD sgdModel = null;
//
//	public NaiveBayes_SGD_TargetDomainTrainingOnly(
//			FeatureSelection featureSelection2,
//			ClassificationKnowledge knowledge2, ClassifierParameters param2) {
//		featureSelection = featureSelection2;
//		knowledge = knowledge2;
//		param = param2;
//	}
//
//	/**
//	 * This only works for binary classification for now.
//	 */
//	@Override
//	public void train(Documents trainingDocs) {
//		// Obtain the knowledge from the classic naive Bayes classifier.
//		NaiveBayes nbClassifier = new NaiveBayes(featureSelection, param);
//		nbClassifier.train(trainingDocs);
//
//		ClassificationKnowledge newKnowledge = nbClassifier.knowledge;
//		newKnowledge.countDomainsInPerClass = knowledge.countDomainsInPerClass;
//
//		sgdModel = new NaiveBayes_SGD(param, featureSelection, newKnowledge,
//				null, null, null, null);
//		sgdModel.train(trainingDocs);
//	}
//
//	@Override
//	public ClassificationEvaluation test(Documents testingDocs) {
//		return sgdModel.test(testingDocs);
//	}
//
//	@Override
//	public List<ItemWithValue> getFeaturesByRatio(Document testingDoc) {
//		return sgdModel.getFeaturesByRatio(testingDoc);
//	}
//
//	@Override
//	public double[] getCountsOfClasses(String featureStr) {
//		return sgdModel.getCountsOfClasses(featureStr);
//	}
//}
