package classifier;

import java.lang.reflect.Field;

import nlp.Documents;
import main.CmdOption;

/**
 * Note: do not put non-basic type parameters in this class.
 */
public class ClassifierParameters implements Cloneable {
	// ------------------------------------------------------------------------
	// General Settings for Sentiment Classification
	// ------------------------------------------------------------------------
	public int noOfCrossValidationFolders = 5;
	public int noOfCVInTraining = 10;

	public String domain = null; // The name of domain.
	public int D = 0; // #Documents.
	public int K = 0; // #Past Tasks

	// ------------------------------------------------------------------------
	// General Parameters for Naive Bayes
	// ------------------------------------------------------------------------
	public double smoothingPriorForFeatureInNaiveBayes = 0.1; // smoothing factor
	public double mCategoryPrior = 0.5; // The prior probability of category, used in smoothing.

	public double trainingNegativeVSPositiveRatio = 0; // in order balance the data
										// when the negative is more than positive

	//wangsong add
	public double domainSimilarity = 0;
	public String vtMode = "none";
	public String vkbMode = "none";

	// ------------------------------------------------------------------------
	// General Parameters for classifier.
	// ------------------------------------------------------------------------
	public boolean discardUnseenWords = false; // decide whether discard unseen words or use ACL2015
											// if false, using ACL2015
	public boolean lifelongSequenceSwitch = true; // if True, "NaiveBayes_SGD_Lifelong_Sequence" goback
	public String classifierName = "NaiveBayes_Sequence_GoBack"; // "NaiveBayes_SGD_Lifelong";
															// "NaiveBayes",
															// "NaiveBayes_Sequence",
															// "NaiveBayes_Sequence_GoBack",
															// "KnowledgeableNB",
															// "NaiveBayes_AddPastDomain", and so on...
															// "MyLogisticRegressionFromMallet"
															// "LibSVM",
															// "LifelongBagging"
															//  'LSC_Stock' (zyw project)
	// ------------------------------------------------------------------------
	// Different settings for training data.
	// ------------------------------------------------------------------------
	public boolean includeTargetDomainLabeledDataForTraining = false ; // only training target domain
	public boolean includeSourceDomainsLabeledDataForTraining =true; // also training source domains

	// ------------------------------------------------------------------------
	// Lifelong sentiment classification.
	// ------------------------------------------------------------------------
	public double gammaThreshold = 2;
	public double positiveRatioThreshold = 2; // 3 is best fo 200 reviews; 2 is best for 1000 reviews
	                                          // zhiyuan uses 6
	public double negativeRatioThreshold = 1.0 / positiveRatioThreshold;
	public double positiveOrNegativeFrequencyThreshold = 1;
	public double regCoefficientAlpha = 0.1; // \alpha // it is reported in paper that 0.1 is good,
	// however, 0.01 is working better here.
	public double domainLevelKnowledgeSupportThreshold = 12; // 8 is best for 200 reviews; 12 is best for 1000 reviews

	public boolean mergeSimplyLabeledDataFromOtherDomainsForTraining = false; // merge target and source domains
	// for training the model.
	// Why also define this due to even including source domains, here will decide whether merge them or not.

	public boolean noTrainingDataFromTargetDomainNorMergingTrainingData = false;
	public boolean useFeaturesFromOtherDomains = false;
	public boolean featureSelectionInTargetDomain = false;

	// Features from other domains.
	// public int featureFrequencyThreshold = 0;
	// public int noOfTopFeaturesByIGFromOtherDomains = -1;

	// ------------------------------------------------------------------------
	// Features.
	// ------------------------------------------------------------------------
	// Feature Generation.
	public int noOfGrams = 1; // N-Gram. // When N is 1, it is named unigram
	public boolean useNGramFeatures = true;

	// Feature Selection.
	public int noOfSelectedFeatures = -1; // 1000; // < 0 means NoSelection
	public String outputTopFeaturesFilePath = ".\\Data\\Output\\TopFeatures\\";
	public String featureSelectionSetting = "NoSelection"; // "NoSelection",
													// "TestOfProportion",
													// "InformationGain", or "ChiSquare".
	public double featureSelectionSignificanceLevel = 0.01; // 0.05; // 0.01;

	// Topic model results as features.
	public boolean useTopicModelFeatures = false;
	public String topicModelNameForFeatureGeneration = "JST_Seed";
	public String topicModelSettingNameForFeatureGeneration = "JST_Seed_Custom_Best";
	public String outputTopicModelMultiDomainFilepath = null;

	// ------------------------------------------------------------------------
	// input and output.
	// ------------------------------------------------------------------------
	public String inputTestingBigDataPath = null;
	public final String nbRootDirectory = ".\\Data\\Output\\SentimentClassificaton\\NaiveBayes\\";
	public final String lifelongWithOnlyTrainingData = ".\\Data\\Output\\SentimentClassificaton\\LifelongWithOnlyTrainingData\\";
	public static String KnowledgeFromSourceDomainsDirectory = ".\\Data\\Intermediate\\ClassificationKnowledge\\";
	// Debugging output file path.
	public String unseenWordsPath = null; // ".\\Data\\Output\\UnseenWords\\";
	public static String wordInformationFilepath = null; // ".\\Data\\Output\\WordInfoForEachDomain\\";
	public String misclassifiedDocumentsFilePath = ".\\Data\\Output\\MisclassifiedInstances\\";
	public String classificationDetailsFilePath = null; // ".\\Data\\Output\\ClassificationDetails\\";
	public String topfeaturesInOtherDomainsDirectory = null; // ".\\Data\\Output\\OtherDomainsFeatures\\";

	// ------------------------------------------------------------------------
	// SVM Light.
	// ------------------------------------------------------------------------
	public final String svmLightRootDirectory = ".\\Data\\Output\\SentimentClassificaton\\SVMLight\\";
	public String svmLightCVFoldDirectory = null;
	public double cost_factor = 1.0;
	// Feature value for SVM (also for logistic regression).
	public String featureValueSettingForSVM = "TF-IDF"; // "TF-IDF", "TF", "1";

	/************************* Parameter for SGD ***********************/
	public double learningRate = 5; // Best: 0.1. ->was 5
	public double learningRateChange = 0.1; // Best: 0.
	// public int noOfSGDIterations = 0; // Best: 100.
	public int maxSGDIterations = 1000;
	public double convergenceDifference = 0.00001; // 0.001 // 0.00001
	public boolean gradientVerificationUsingFiniteDifferences = false;
	public double gradientVerificationDelta = 1e-4;

	//====================================
    // lifelong learning parameter tuning
    //====================================
	public boolean tuneParametersUsingCrossValidationInTraining = false; // Parameter tuning candidates.
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
	public static double[] convergenceDifferenceCandidates = { 0.1, 0.001}; // {
	// 0.01,
	// 0.001,
	// 0.0001 };
    public static double[] positiveRatioThresholdCandidates = {6};
    // public static double[] regularizationCoeffCandidates = {0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10};
    public static double[] regularizationCoeffCandidates = {0.001, 0.01, 0.1, 5, 10};
    public static int[] domainLevelKnowledgeSupportThresholdCandidates = {5, 10, 15, 20};
    // tuning end ===========================
	public ClassifierParameters() {

	}

	public ClassifierParameters(Documents documents, CmdOption cmdOption) {
		D = documents.size(); // #Documents
		K = 0; // #Past Tasks

		domain = documents.domain; // The name of domain
		noOfCrossValidationFolders = cmdOption.noOfCrossValidationFolders;
		smoothingPriorForFeatureInNaiveBayes = cmdOption.smoothingPriorForFeatureInNaiveBayes;
		mCategoryPrior = cmdOption.mCategoryPrior;

		inputTestingBigDataPath = cmdOption.inputTestingBigDataPath;
		outputTopicModelMultiDomainFilepath = cmdOption.outputTopicModelMultiDomainFilepath;

		//wangsong add
		gammaThreshold = cmdOption.gammaThreshold;
		positiveRatioThreshold = cmdOption.positiveRatioThreshold;
		domainLevelKnowledgeSupportThreshold = cmdOption.domainNumLavege;

		domainSimilarity = cmdOption.domainSimilarity;
		vkbMode = cmdOption.vkbMode;
		vtMode = cmdOption.vtMode;
		noOfGrams = cmdOption.ngram;


	}

	//	 public ClassifierParameters(Documents documents, ClassifierParameters
//	 param) {
//		 D = documents.size();
//
//		 domain = documents.domain;
//		 classifierName = param.classifierName;
//		 noOfCrossValidationFolders = param.noOfCrossValidationFolders;
//
//		 outputTopicModelMultiDomainFilepath = param.outputTopicModelMultiDomainFilepath;
//	 }

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

//    @Override
//    /**
//     * param.domainLevelKnowledgeSupportThreshold = numDomain;
//       param.regCoefficientAlpha = regCoefficientAlpha;
//       param.positiveRatioThreshold = positiveThreshold;
//       param.convergenceDifference = converg;
//       param.learningRate = learningRate;
//     */
//    public String toString() {
//	    StringBuilder sbOutput = new StringBuilder();
//	    sbOutput.append("[domain level knowledge support threshold]: " + this.domainLevelKnowledgeSupportThreshold);
//	    sbOutput.append("[regularization coefficient]: " + this.regCoefficientAlpha);
//	    sbOutput.append("[positive ratio threshold]: " + this.positiveRatioThreshold);
//	    sbOutput.append("[convergenceDifference]: " + this.convergenceDifference);
//	    sbOutput.append("[learning rate]: " + this.learningRate);
//	    return sbOutput.toString();
//    }

}
