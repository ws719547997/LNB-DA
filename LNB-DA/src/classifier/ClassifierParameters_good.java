package classifier;

import java.lang.reflect.Field;

import nlp.Documents;
import main.CmdOption;

/**
 * Note: do not put non-basic type parameters in this class.
 * ==============
 * This configuration get result:
 * Average Accuracy: 0.8337499999999999
 * Average F1-score (Both Classes): 0.8326728752665534
 * Average F1-score (Positive): 0.8381205052323789
 * Average F1-score (Negative): 0.8272252453007278
 */
public class ClassifierParameters_good implements Cloneable {
	// ------------------------------------------------------------------------
	// General Settings for Sentiment Classification
	// ------------------------------------------------------------------------
	public int noOfCrossValidationFolders = 5;
	public int noOfCVInTraining = 10;

	// ------------------------------------------------------------------------
	// Task Specific Parameters
	// ------------------------------------------------------------------------
	public String domain = null; // The name of the domain.

	// ------------------------------------------------------------------------
	// General Parameters for classifier.
	// ------------------------------------------------------------------------
	public String classifierName = "NaiveBayes_SGD_Lifelong"; // "LogisticRegressionFromLingpipe";
																		// //
																		// "MyLogisticRegression";
	// "NaiveBayes",
    // "MyLogisticRegressionFromMallet"
	// "LibSVM",
	// "LibLinear", (Best for sentiment classification).
	// "SVMLight",
	// "NaiveBayes_Lifelong",
	// "NaiveBayes_Lifelong_NewRatio",
	// "NaiveBayes_Lifelong_WithoutCurrentDomainTraining",
	// "NaiveBayes_SGD_Lifelong",
	// "NaiveBayes_SGD_TargetDomainTrainingOnly".
	public int D = 0; // #Documents.

	public double smoothingPriorForFeatureInNaiveBayes = -1; // Coming from
																// CmdOption.
	// ------------------------------------------------------------------------
	// Features.
	// ------------------------------------------------------------------------
	// Feature Generation.
	public int noOfGrams = 1;
	public boolean useNGramFeatures = true;

	// Feature Selection.
	public int noOfSelectedFeatures = -1; // 1000; // < 0 means no feature
	// selection.
	public String outputTopFeaturesFilePath = "..\\Data\\Output\\TopFeatures\\";
	public String featureSelectionSetting = "NoSelection"; // "NoSelection",
	// "TestOfProportion",
	// "InformationGain", "ChiSquare".
	public double featureSelectionSignificanceLevel = 0.01; // 0.05;
															// 0.01;

	// Feature value for SVM (also for logistic regression).
	public String featureValueSettingForSVM = "TF-IDF"; // "TF-IDF", "TF", "1";

	// Topic model results as features.
	public boolean useTopicModelFeatures = false;
	public String topicModelNameForFeatureGeneration = "JST_Seed";
	public String topicModelSettingNameForFeatureGeneration = "JST_Seed_Custom_Best";
	public String outputTopicModelMultiDomainFilepath = null;

	// ------------------------------------------------------------------------
	// Lifelong learning.
	// ------------------------------------------------------------------------
	// Different settings for training data.
	public boolean includeTargetDomainLabeledDataForTraining = true;
	public boolean includeOtherDomainsLabeledDataForTraining = false;
	public boolean mergeSimplyLabeledDataFromOtherDomainsForTraining = false;
	public boolean noTrainingDataFromTargetDomainNorMergingTrainingData = false;
	public boolean useFeaturesFromOtherDomains = false;
	public boolean featureSelectionInTargetDomain = false;

	// Features from other domains.
	// public int featureFrequencyThreshold = 0;
	// public int noOfTopFeaturesByIGFromOtherDomains = -1;

	public double trainingNegativeVSPositiveRatio = 0;
	// ------------------------------------------------------------------------
	// SVM Light.
	// ------------------------------------------------------------------------
	public final String svmLightRootDirectory = "..\\Data\\Output\\SentimentClassificaton\\SVMLight\\";
	public final String nbRootDirectory = "..\\Data\\Output\\SentimentClassificaton\\NaiveBayes\\";
	public String svmLightCVFoldDirectory = null;
	public double cost_factor = 1.0;

	/************************* Parameter for SGD ***********************/
	public double learningRate = 5; // Best: 0.1. ->was 5
	public double learningRateChange = 0.1; // Best: 0.
	// public int noOfSGDIterations = 0; // Best: 100.
	public int maxSGDIterations = 1000;
	public double convergenceDifference = 0.00001; //
	// Double.MAX_VALUE;
	// // 0.001;
	public boolean gradientVerificationUsingFiniteDifferences = false;
	public double gradientVerificationDelta = 1e-4;

	// Parameter tuning candidates.
	public boolean tuneParametersUsingCrossValidationInTraining = false;
	public static double[] learningRateCandidates = { 0.0001, 0.0005, 0.001,
			0.005, 0.1, 0.5, 1, 5, 10, 15, 20 };// {
	// 0.0001,
	// 0.0005,
	// 0.001,
	// 0.005,
	// 0.01,
	// 0.05,
	// 0.1,
	// 0.5,
	// 1, 5,
	// 10,
	// 20,
	// 50 };
	public static double[] convergenceDifferenceCandidates = { 0.001, 0.0001 }; // {
	// 0.01,
	// 0.001,
	// 0.0001 };

	// ------------------------------------------------------------------------
	// Debugging.
	// ------------------------------------------------------------------------
	public String misclassifiedDocumentsFilePath = null; // "..\\Data\\Output\\MisclassifiedInstances\\";
	public String topfeaturesInOtherDomainsDirectory = null; // "..\\Data\\Output\\OtherDomainsFeatures\\";
	public static String KnowledgeFromSourceDomainsDirectory = "..\\Data\\Intermediate\\ClassificationKnowledge\\";
	public String classificationDetailsFilePath = "..\\Data\\Output\\ClassificationDetails\\";

	// ------------------------------------------------------------------------
	// Lifelong sentiment classification.
	// ------------------------------------------------------------------------
	public double positiveRatioThreshold = 6;
	public double negativeRatioThreshold = 1.0 / positiveRatioThreshold;
	public double positiveOrNegativeFrequencyThreshold = 0.0;
	public double regularizationCoeff = 0.01;  // it is reported in paper that 0.1 is good, however, 0.01 is working better here.

	public int domainLevelKnowledgeSupportThreshold = 6; // 6; // 5; // 7; //
															// -1;

	public ClassifierParameters_good(Documents documents, CmdOption cmdOption) {
		D = documents.size();

		domain = documents.domain;
		noOfCrossValidationFolders = cmdOption.noOfCrossValidationFolders;
		smoothingPriorForFeatureInNaiveBayes = cmdOption.smoothingPriorForFeatureInNaiveBayes;

		outputTopicModelMultiDomainFilepath = cmdOption.outputTopicModelMultiDomainFilepath;
	}

	// public ClassifierParameters(Documents documents, ClassifierParameters
	// param) {
	// D = documents.size();
	//
	// domain = documents.domain;
	// classifierName = param.classifierName;
	// noOfCrossValidationFolders = param.noOfCrossValidationFolders;
	//
	// outputTopicModelMultiDomainFilepath =
	// param.outputTopicModelMultiDomainFilepath;
	// }

	@Override
	protected Object clone() throws CloneNotSupportedException {
		return (ClassifierParameters) super.clone();
	}

	public ClassifierParameters getClone() {
		try {
			// Clone basic types.
			ClassifierParameters clone = (ClassifierParameters) this.clone();
			return clone;
		} catch (Exception ex) {
			ex.printStackTrace();
		}
		return null;
	}

	@Override
	public String toString() {
		try {
			StringBuilder sbOutput = new StringBuilder();
			// Output each member of settings.
			for (Field field : this.getClass().getDeclaredFields()) {
				String typestr = field.getType().toString().toLowerCase();
				if (typestr.endsWith("string") || typestr.endsWith("int")
						|| typestr.endsWith("double")
						|| typestr.endsWith("float")
						|| typestr.endsWith("boolean")) {
					// We only print the fields with basic types.
					sbOutput.append(field.getName() + "=" + field.get(this));
					sbOutput.append(System.getProperty("line.separator"));
				}
			}
			return sbOutput.toString();
		} catch (Exception ex) {
			ex.printStackTrace();
		}
		return null;
	}
}
