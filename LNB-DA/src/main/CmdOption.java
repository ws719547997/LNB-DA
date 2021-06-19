package main;

import org.kohsuke.args4j.Option;

/**
 * Command line options.
 * 
 * The program runs in parallel, which is controlled by the number of threads
 * (nthreads as below).
 */
public class CmdOption {

	// ------------------------------------------------------------------------
	// Input and Output
	// ------------------------------------------------------------------------

	@Option(name = "-i", usage = "Specify the input directory of each domain "
			+ "where each domain contains documents and vocabulary")
	public String inputCorporeaDirectory = "./Data/Input/Dataset/";

	@Option(name = "-o", usage = "Specify the output root directory of the program")
	public String outputRootDirectory = "./Data/Output/";

	@Option(name = "-sdocs", usage = "Specify the suffix of input docs file")
	public String suffixInputCorporeaDocs = ".docs";

	@Option(name = "-svocab", usage = "Specify the suffix of input vocab file")
	public String suffixInputCorporeaVocab = ".vocab";

	@Option(name = "-sori", usage = "Specify the suffix of input ori file")
	public String suffixInputCorporeaOriContext = ".ori";

	@Option(name = "-nlearn", usage = "Specify the number of learning iterations of task 0")
	public int nLearningIterations = 1;

	@Option(name = "-nthreads", usage = "Specify the number of maximum threads in multithreading")
	public int nthreads = 1;

	@Option(name = "-includeCurrentDomain", usage = "Specify if the current domain is included for knowledge extraction.")
	public boolean includeCurrentDomainAsKnowledgeExtraction = false;

	// ------------------------------------------------------------------------
	// General Settings for Topic Model
	// ------------------------------------------------------------------------

	@Option(name = "-ntopics", usage = "Specify the number of topics")
	public int nTopics = 15;

	@Option(name = "-burnin", usage = "Specify the number of iterations for burn-in period")
	public int nBurnin = 200;

	@Option(name = "-niters", usage = "Specify the number of Gibbs sampling iterations")
	public int nIterations = 2000;

	@Option(name = "-slag", usage = "Specify the length of interval to sample for "
			+ "calculating posterior distribution")
	public int sampleLag = 20; // -1 means only sample the last one.

	public String firstIterationModel = "JST_Seed"; // "LDA";

	@Option(name = "-mname", usage = "Specify the name of the topic model")
	public String modelName = "LDA_IncorporateRankingIntoSampling"; // "MustLDAWithDynamicMustSet";
	// "LTM";

	/******************* Hyperparameters *********************/
	@Option(name = "-alpha", usage = "Specify the hyperparamter alpha")
	public double alpha = 1.0;

	@Option(name = "-beta", usage = "Specify the hyperparamter beta")
	public double beta = 0.1;

	@Option(name = "-rseed", usage = "Specify the seed for random number generator")
	public int randomSeed = 837191;

	/******************* Output *********************/
	@Option(name = "-twords", usage = "Specify the number of top words for each topic")
	public int twords = 20; // Print out top words ranked by probabilities per
							// each topic. -1: print out all words under topic.

    /****************************** Attention!!! *******************************/
    /****************************** From here, *******************************/
    /******************* Lifelong sentiment classification *********************/
	// ------------------------------------------------------------------------
	// Settings for Sentiment Classification
	// ------------------------------------------------------------------------
	@Option(name = "-cv", usage = "Specify the number of folders in cross validation")
	public int noOfCrossValidationFolders = 5;
	public String dataVsSetting = "One-vs-Rest";

	public boolean randomlySampleNegativeToBalancedClass = false;
	public double positiveNegativeRatio = 0;

    public double smoothingPriorForFeatureInNaiveBayes = 0.1; // smoothing factor // was 0.1, 0.5
	public double mCategoryPrior = 0.5; // The prior probability of category, used in smoothing.

	// wangsong add
	public double domainSimilarity = 0;
	public double domainNumLavege = 0;
	public double gammaThreshold = 0;
	public double positiveRatioThreshold = 0;
	public int ngram = 1;
	public String attantionMode = "none"; //"none"
										//"att" "att_max" "att_avg" "att_percent"
	public String vkbMode = "none";  // "ds"(domainSimilarity)
	public String vtMode = "none";	//"add"

	// Inputs.
    /** Attention here. */
	public static int numberOfMaximumSourceDomains = 20; // need setting
    public String datasetName =  "1KReviewNaturalClassDistributionDomains"; // "1KReviewNaturalClassDistributionDomains"
													// "100P100NDomains"
													//(Reuters10)stock 因为读取任务中没有stock 我就借用了Reuters10的名字 里边的方法改了
	// public String datasetName = "20Newgroup";

    /** Attention here. */
	// natural domains
//	public String inputListOfDomainsToEvaluate = ".\\Data\\DomainToEvaluate\\DomainsToEvaluate_" + numberOfMaximumSourceDomains + "shuffle1.txt";
	public String outputListOfDomainsToEvaluate = ".\\Data\\DomainToEvaluate\\DomainsToEvaluate_" + numberOfMaximumSourceDomains;
	public String inputListOfDomainsToEvaluate = ".\\Data\\DomainToEvaluate\\stock.txt";
	// similar domains
	// public String inputListOfDomainsToEvaluate = ".\\Data\\DomainToEvaluate\\SimilarDomainsToEvaluate_" + numberOfMaximumSourceDomains + ".txt";
	// dissimilar domains
//	public String inputListOfDomainsToEvaluate = ".\\Data\\DomainToEvaluate\\DissimilarDomainsToEvaluate_" + numberOfMaximumSourceDomains + ".txt";
	// dissimilar and similar domains
	// public String inputListOfDomainsToEvaluate = ".\\Data\\DomainToEvaluate\\DisAndSimilarDomainsToEvaluate_" + numberOfMaximumSourceDomains + ".txt";


//    public String input100P100NReviewAllDirectory = ".\\Data\\Input\\20domains_1000reviews_original\\"; // 20domains_1000reviews_original; 20domains_200reviews_100P100N
	 public String input100P100NReviewAllDirectory = ".\\Data\\Input\\20domains_200reviews_100P100N\\";
	// public String input100P100NReviewAllDirectory = ".\\Data\\Input\\preprocessed_manual_similar_domains_new\\";
    // public String input100P100NReviewAllDirectory = ".\\Data\\Input\\preprocessed_manual_similar_domains\\";
    // public String input100P100NReviewAllDirectory = ".\\Data\\Input\\Test_Naive_Bayes_Classifier_bigger_testset\\";
	public String inputTestingBigDataPath = ".\\Data\\Input\\preprocessed_manual_similar_domains_new\\Automotive_bigTestData.txt";

	public String inputSameDomainDifferntProductsDirectoy = ".\\Data\\Input\\SameDomainDifferentProducts\\";
	public String input1KP1KNReviewDirectory = ".\\Data\\Input\\DifferentDomains\\1000+1000-\\";
	public String inputPangAndLeeReviewsDirectory = ".\\Data\\Input\\PangAndLee\\";
	public String inputReuters10DomainsDirectory = ".\\Data\\Input\\Reuters10domains\\";
	public String input20NewsgroupFilepath = ".\\Data\\Input\\20Newgroup\\20ng-stemmed.txt";
	public String input50Electronics1KReview = ".\\Data\\Input\\DifferentDomains\\1KAllReviews\\50Electronics_1000Reviews\\";
	public String input50NonElectronics1KReview = ".\\Data\\Input\\DifferentDomains\\1KAllReviews\\50NonElectronics_1000Reviews\\";
	/**
	 * 王松： 输入中没有这些子文件夹 我把他改成original
	 */
//	public String input1KReviewNaturalClassDistribution = ".\\Data\\Input\\DifferentDomains\\1KAllReviews_20Domains\\";
	public String input1KReviewNaturalClassDistribution = ".\\Data\\Input\\20domains_1000reviews_original\\";
	public String inputBalancedWithMostNegativeReviews = ".\\Data\\Input\\DifferentDomains\\BalancedWithMostNegativeReviews\\";
	public String inputstock = ".\\Data\\Input\\stock_comments\\";

	// Intermediate
	public String intermediateTrainingDocsDir = ".\\Data\\Intermediate\\TrainingDocs\\";
	public String intermediateTestingDocsDir = ".\\Data\\Intermediate\\TestingDocs\\";
	public String intermediateKnowledgeDir = ".\\Data\\Intermediate\\Knowledges\\";

	// Output
    public String outputSentimentClassificationAccuracy = ".\\Data\\Output\\SentimentClassificaton\\Accuracy_"
            + datasetName + numberOfMaximumSourceDomains;
    public String outputSentimentClassificationF1Score = ".\\Data\\Output\\SentimentClassificaton\\F1Score_"
            + datasetName + numberOfMaximumSourceDomains;
    public String outputTopicModelMultiDomainFilepath = ".\\Data\\Output\\TopicModel\\LearningIteration0\\";

	public CmdOption getSoftCopy() {
		CmdOption cmdOption2 = new CmdOption();
		cmdOption2.inputCorporeaDirectory = this.inputCorporeaDirectory;
		cmdOption2.outputRootDirectory = this.outputRootDirectory;
		cmdOption2.suffixInputCorporeaDocs = this.suffixInputCorporeaDocs;
		cmdOption2.suffixInputCorporeaVocab = this.suffixInputCorporeaVocab;
		cmdOption2.nLearningIterations = this.nLearningIterations;
		cmdOption2.nthreads = this.nthreads;
		cmdOption2.includeCurrentDomainAsKnowledgeExtraction = this.includeCurrentDomainAsKnowledgeExtraction;
		cmdOption2.nTopics = this.nTopics;
		cmdOption2.nBurnin = this.nBurnin;
		cmdOption2.nIterations = this.nIterations;
		cmdOption2.sampleLag = this.sampleLag;
		cmdOption2.modelName = this.modelName;
		cmdOption2.alpha = this.alpha;
		cmdOption2.beta = this.beta;
		cmdOption2.randomSeed = this.randomSeed;
		cmdOption2.twords = this.twords;
		cmdOption2.ngram=this.ngram;
		cmdOption2.attantionMode=this.attantionMode;
		cmdOption2.vkbMode=this.vkbMode;
		cmdOption2.vtMode=this.vtMode;
		cmdOption2.domainSimilarity=this.domainSimilarity;
		cmdOption2.domainNumLavege=this.domainNumLavege;
		cmdOption2.gammaThreshold=this.gammaThreshold;
		cmdOption2.positiveRatioThreshold=this.positiveRatioThreshold;

		return cmdOption2;
	}
}
