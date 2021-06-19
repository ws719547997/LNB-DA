package task;

import feature.FeatureGenerator;
import topicmodel.TopicModel;
import utility.CrossValidationOperatorMaintainingLabelDistribution;
import utility.Pair;
import nlp.Documents;
import classificationevaluation.ClassificationEvaluation;
import classificationevaluation.ClassificationEvaluationAccumulator;
import classifier.BaseClassifier;
import classifier.ClassificationKnowledge;
import classifier.ClassifierParameters;
import feature.FeatureSelection;

public class TrainingCrossValidationToFindOptimalParametersTask {

	public TrainingCrossValidationToFindOptimalParametersTask() {

	}


	public ClassifierParameters parametersTuning(ClassifierParameters param2,
                                                 ClassificationKnowledge knowledge, Documents allTrainingDocs) {

	    ClassificationEvaluationAccumulator evaluationAccumulator = new ClassificationEvaluationAccumulator();
        CrossValidationOperatorMaintainingLabelDistribution cvo = new CrossValidationOperatorMaintainingLabelDistribution(
                allTrainingDocs, param2.noOfCVInTraining);

        ClassifierParameters optimalParameters = null;
        double highestF1Score = Double.MIN_VALUE;

        for (int numDomain : ClassifierParameters.domainLevelKnowledgeSupportThresholdCandidates) {
            for (double regularizationCoeff : ClassifierParameters.regularizationCoeffCandidates) {
                for (double positiveThreshold : ClassifierParameters.positiveRatioThresholdCandidates) {
                    for (double converg : ClassifierParameters.convergenceDifferenceCandidates) {
                        for (double learningRate : ClassifierParameters.learningRateCandidates) {

                            //get a new classifier param.
                            ClassifierParameters param = param2.getClone();
                            param.domainLevelKnowledgeSupportThreshold = numDomain;
                            param.regCoefficientAlpha = regularizationCoeff;
                            param.positiveRatioThreshold = positiveThreshold;
                            param.convergenceDifference = converg;
                            param.learningRate = learningRate;

                            for (int folderIndex = 0; folderIndex < param.noOfCVInTraining; ++folderIndex) {
                                Pair<Documents, Documents> pair = cvo.getTrainingAndTestingDocuments(folderIndex,
                                        param.noOfCrossValidationFolders);
                                //get training and testing document
                                Documents trainingDocs = pair.t;
                                Documents testingDocs = pair.u;

                                TopicModel topicModelForThisDomain = null;
                                FeatureGenerator featureGenerator = new FeatureGenerator(param);
                                featureGenerator
                                        .generateAndAssignFeaturesToTrainingAndTestingDocuments(
                                                trainingDocs, testingDocs, topicModelForThisDomain);
                                // Feature selection.
                                FeatureSelection featureSelection = FeatureSelection
                                        .selectFeatureSelection(trainingDocs, param);


                                BaseClassifier clf = BaseClassifier.selectNewClassifier(param.classifierName, featureSelection, knowledge, param);
                                clf.train(trainingDocs);
                                ClassificationEvaluation evaluation = clf.test(testingDocs);
                                evaluationAccumulator.addClassificationEvaluation(evaluation);

                            }

                            double tmpF1Score = evaluationAccumulator.getAverageClassificationEvaluation().f1score;
                            if (tmpF1Score > highestF1Score) {
                                highestF1Score = tmpF1Score;
                                optimalParameters = param.getClone();
                            }

                            System.out.println(param.domain + " : in training "
                                    + "learningRate = " + param.learningRate + " "
                                    + "convergenceDiff = " + param.convergenceDifference + " "
                                    + "F1Score = " + tmpF1Score);

                        }
                    }

                }
            }
        }

        System.out.println(optimalParameters.domain + " : in training "
                + optimalParameters.learningRate + " "
                + optimalParameters.convergenceDifference + " "
                + highestF1Score);
        return optimalParameters;

    }

	public ClassifierParameters run(ClassifierParameters param2,
			FeatureSelection featureSelection,
			ClassificationKnowledge knowledge, Documents allTrainingDocs) {
		// Prepare for cross validation in all training data.
		ClassificationEvaluationAccumulator evaluationAccumulator = new ClassificationEvaluationAccumulator();
		CrossValidationOperatorMaintainingLabelDistribution cvo = new CrossValidationOperatorMaintainingLabelDistribution(
				allTrainingDocs, param2.noOfCVInTraining);

		ClassifierParameters optimalParameters = null;
		double highestF1Score = Double.MIN_VALUE;
		for (double convergenceDifference : ClassifierParameters.convergenceDifferenceCandidates) {
			for (double learningRate : ClassifierParameters.learningRateCandidates) {
				// Get new classifier param.
				ClassifierParameters param = param2.getClone();
				param.convergenceDifference = convergenceDifference;
				param.learningRate = learningRate;

				for (int folderIndex = 0; folderIndex < param2.noOfCVInTraining; ++folderIndex) {
					Pair<Documents, Documents> pair = cvo
							.getTrainingAndTestingDocuments(folderIndex, param.noOfCrossValidationFolders);
					// Get training and testing documents.
					// No deep clone here because of single thread.
					Documents trainingDocs = pair.t;
					Documents testingDocs = pair.u;

					BaseClassifier classifier = BaseClassifier
							.selectNewClassifier(param.classifierName,
									featureSelection, knowledge, param);
					classifier.train(trainingDocs);

					ClassificationEvaluation evaluation = classifier
							.test(testingDocs);
					evaluationAccumulator
							.addClassificationEvaluation(evaluation);
				}
				double tmpF1Score = evaluationAccumulator
						.getAverageClassificationEvaluation().f1score;
				if (tmpF1Score > highestF1Score) {
					highestF1Score = tmpF1Score;
					optimalParameters = param.getClone();
				}
			}
		}
		System.out.println(optimalParameters.domain + " : in training "
				+ optimalParameters.learningRate + " "
				+ optimalParameters.convergenceDifference + " "
				+ highestF1Score);
		return optimalParameters;
	}
}
