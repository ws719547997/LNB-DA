package classifier;

import java.io.File;
import java.util.List;
import java.util.Map;

import svmlight.SVMLightFeatureWeightsAndBValue;
import svmlight.SVMLightHelper;
import utility.ExceptionUtility;
import nlp.Document;
import nlp.Documents;
import classificationevaluation.ClassificationEvaluation;
import feature.FeatureSelection;

/**
 * Use SVM Light program to do the training and testing.
 */
public class SVMLight extends BaseClassifier {
	private SVMLightHelper helper = null;
	private SVMLightFeatureWeightsAndBValue fw = null;
	private ClassifierParameters param = null;

	private String trainingDocsFilepath = null;
	// private String testingDocsFilepath = null;
	private String svmLearningModelFilepath = null;
	// private String testingResultsFilepath = null;
	private String featureWeightsOutputFilePath = null;

	public SVMLight(
			FeatureSelection featureSelection2,
			Map<String, Map<String, double[]>> mpSelectedFeatureStrToDomainToProbs2,
			ClassifierParameters param2) {
		helper = new SVMLightHelper();

		featureSelection = featureSelection2;
		mpSelectedFeatureStrToDomainToProbs = mpSelectedFeatureStrToDomainToProbs2;

		param = param2;

		trainingDocsFilepath = param.svmLightCVFoldDirectory + File.separator
				+ "TrainingDocs.txt";
		// testingDocsFilepath = param.svmLightCVFoldDirectory + File.separator
		// + "TestingDocs.txt";
		svmLearningModelFilepath = param.svmLightCVFoldDirectory
				+ File.separator + "Model.txt";
		// testingResultsFilepath = param.svmLightCVFoldDirectory +
		// File.separator
		// + "TestingResults.txt";
		featureWeightsOutputFilePath = param.svmLightCVFoldDirectory
				+ File.separator + "Features.txt";
	}

	@Override
	public void train(Documents trainingDocs) {
		// Print features to file.
		helper.printDocumentsToFile(trainingDocs, featureSelection,
				trainingDocsFilepath);
		// Call svm_learn to learn the model.
		helper.learnSVMLightModel(trainingDocsFilepath,
				svmLearningModelFilepath, param.cost_factor);
		// Read the model from file.
		fw = SVMLightFeatureWeightsAndBValue
				.readSVMLightFeatureWeightsAndBValue(svmLearningModelFilepath);
		fw.printFeaturesRankedByWeights(featureSelection,
				featureWeightsOutputFilePath);
	}

	@Override
	public ClassificationEvaluation test(Documents testingDocs) {
		List<String> predictedClassList = fw.getPredictedClasses(testingDocs,
				featureSelection);

		// helper.printDocumentsToFile(testingDocs, featureSelection,
		// testingDocsFilepath);
		// List<String> predictedClassList = helper.getPredictedClasses(
		// testingDocsFilepath, svmLearningModelFilepath,
		// testingResultsFilepath);

		// Verify both results.
		for (int i = 0; i < predictedClassList.size(); ++i) {
			ExceptionUtility.assertAsException(predictedClassList.get(i)
					.equals(predictedClassList.get(i)));
		}

		for (int i = 0; i < testingDocs.size(); ++i) {
			Document document = testingDocs.getDocument(i);
			document.predict = predictedClassList.get(i);
		}
		// helper.testModelBySVMLight(testingDocsFilepath,
		// svmLearningModelFilepath, testingResultsFilepath);
		ClassificationEvaluation evaluation = new ClassificationEvaluation(
				testingDocs.getLabels(), testingDocs.getPredicts(),
				param.domain);
		return evaluation;
	}

	@Override
	public void printMisclassifiedDocuments(Documents testingDocs, String misclassifiedDocumentsForOneCVFolderForOneDomainFilePath) {

	}
}
