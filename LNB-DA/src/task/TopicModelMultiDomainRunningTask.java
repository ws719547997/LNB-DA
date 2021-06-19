package task;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import topicmodel.ModelLoader;
import topicmodel.ModelPrinter;
import topicmodel.TopicModel;
import topicmodel.TopicModelParameters;
import utility.FileReaderAndWriter;
import main.CmdOption;
import main.Constant;
import multithread.TopicModelMultiThreadPool;
import nlp.Corpus;
import nlp.Documents;

/**
 * Run the topic model on each domain.
 */
public class TopicModelMultiDomainRunningTask {
	// Inputs.
	// 1K reviews.
	public String input1KReviewElectronicsDirectory = "..\\Data\\Input\\DifferentDomains\\1KAllReviews\\50Electronics_1000Reviews\\";
	public String input1KReviewNonElectronicsDirectory = "..\\Data\\Input\\DifferentDomains\\1KAllReviews\\50NonElectronics_1000Reviews\\";
	public String input1KReviewCorporaForTopicModelDirectory = "..\\Data\\Input\\DifferentDomains\\1KAllReviews\\CorporaForTopicModel\\";
	// Outputs.
	public String outputTopicModelMultiDomainFilepath = "..\\Data\\Output\\TopicModel\\";
	public int noOfCVFolder = 10;

	public CmdOption cmdOption = null;

	public void run(CmdOption cmdOption2) {
		// Note that this threshold is 0 for classification and 5 for topic
		// model.
		Constant.INFREQUENT_WORD_REMOVAL_THRESHOLD = 5;

		cmdOption = cmdOption2;

		List<Documents> documentsOfAllDomains = new ArrayList<Documents>();
		// Electronics.
		documentsOfAllDomains
				.addAll(Documents
						.readListOfDocumentsFromDifferentDomains(input1KReviewElectronicsDirectory));
		// Non-Electronics.
		documentsOfAllDomains
				.addAll(Documents
						.readListOfDocumentsFromDifferentDomains(input1KReviewNonElectronicsDirectory));

		// List<Documents> documentsOfAllDomains2 = new ArrayList<Documents>();
		// documentsOfAllDomains2.add(documentsOfAllDomains.get(0));
		// documentsOfAllDomains = documentsOfAllDomains2;

		// Convert documents of each domain to corpus.
		List<Corpus> corpora = new ArrayList<Corpus>();
		for (Documents documents : documentsOfAllDomains) {
			String domain = documents.domain;
			Corpus corpus = null;
			String docsFilepath = input1KReviewCorporaForTopicModelDirectory
					+ domain + ModelPrinter.docsSuffix;
			String vocabFilepath = input1KReviewCorporaForTopicModelDirectory
					+ domain + ModelPrinter.vocabSuffix;
			if (new File(docsFilepath).exists()
					&& new File(docsFilepath).exists()) {
				// If the corpus file already exists, then read the corpus
				// directly.
				corpus = Corpus.getCorpusFromFile(domain, docsFilepath,
						vocabFilepath, null);
				corpus.documents = documents;
			} else {
				corpus = new Corpus(documents);
				// Write the corpus to the directory.
				ModelPrinter printer = new ModelPrinter(null);
				printer.printDocs(corpus.docs, docsFilepath);
				printer.printVocabulary(corpus.vocab, vocabFilepath);
			}
			corpora.add(corpus);
		}
		List<TopicModel> topicModelList = run(corpora, cmdOption.nTopics,
				cmdOption.firstIterationModel,
				outputTopicModelMultiDomainFilepath);
		if (cmdOption.firstIterationModel.startsWith("JST")) {
			EvaluateSentimentClassificationByTopicModelForEachDomainTask task = new EvaluateSentimentClassificationByTopicModelForEachDomainTask(
					topicModelList);
			task.run();
		}
	}

	/**
	 * Run a knowledge-based topic model and output it into the directory.
	 * 
	 * LearningIteration 0 is always LDA, i.e., without any knowledge.
	 * LearningIteration i with i > 0 is the knowledge-based topic model.
	 * 
	 * The knowledge used for LearningIteration i is extracted from
	 * LearningIteration i - 1, except LearningIteration 0 which is LDA.
	 */
	private List<TopicModel> run(List<Corpus> corpora, int nTopics,
			String modelName, String outputRootDirectory) {
		List<TopicModel> topicModelList_FirstIteration = null; // LDA
																// models.
		List<TopicModel> topicModelList_LastIteration = null;
		List<TopicModel> topicModelList_CurrentIteration = null;

		for (int iter = 0; iter < cmdOption.nLearningIterations; ++iter) {
			System.out.println("###################################");
			System.out.println("Learning Iteration " + iter + " Starts!");
			System.out.println("###################################");

			long startTime = System.currentTimeMillis();

			// The first LearningIteration is LDA, others are the
			// knowledge-based topic model.
			String currentIterationModelName = iter == 0 ? cmdOption.firstIterationModel
					: modelName;

			String currentIterationRootDirectory = outputRootDirectory
					+ "LearningIteration" + iter + File.separator
					+ currentIterationModelName + File.separator;

			// Run the topic model.
			System.out.println("-----------------------------------");
			System.out.println("Running Topic Model on each domain.");
			System.out.println("-----------------------------------");
			topicModelList_LastIteration = topicModelList_CurrentIteration;
			topicModelList_CurrentIteration = runTopicModelForOneLearningIteration(
					iter, corpora, nTopics, currentIterationModelName,
					currentIterationRootDirectory,
					topicModelList_LastIteration, topicModelList_FirstIteration);
			if (iter == 0) {
				topicModelList_FirstIteration = topicModelList_CurrentIteration;
			}

			double timeLength = (System.currentTimeMillis() - startTime) / 1000.0;

			System.out.println("###################################");
			System.out.println("Learning Iteration " + iter + " Ends! "
					+ timeLength + "seconds");
			System.out.println("###################################");
			System.out.println("");

			evaluateTopicModelForOneLearningIteration(
					topicModelList_CurrentIteration, modelName,
					currentIterationRootDirectory);
			System.out.println("###################################");
			System.out.println("Learning Iteration " + iter
					+ " Evaluation Ends!");
			System.out.println("###################################");
		}
		return topicModelList_CurrentIteration;
	}

	/**
	 * Run the topic model (LDA or knowledge-based topic model) for one learning
	 * iteration. We use multithreading and each thread handles the model in one
	 * domain.
	 */
	private List<TopicModel> runTopicModelForOneLearningIteration(int iter,
			List<Corpus> corpora, int nTopics,
			String currentIterationModelName,
			String currentIterationRootDirectory,
			List<TopicModel> topicModelList_LastIteration,
			List<TopicModel> topicModelList_FirstIteration) {
		List<TopicModel> topicModelList_CurrentIteration = new ArrayList<TopicModel>();
		TopicModelMultiThreadPool threadPool = new TopicModelMultiThreadPool(
				cmdOption.nthreads);

		for (Corpus corpus : corpora) {
			String currentIterationModelDirectory = currentIterationRootDirectory
					+ File.separator
					+ "DomainModels"
					+ File.separator
					+ corpus.domain + File.separator;

			if (new File(currentIterationModelDirectory).exists()) {
				// If the model of a domain in this learning
				// iteration already exists, we load it and add it into the
				// topic model list.
				ModelLoader modelLoader = new ModelLoader();
				TopicModel modelForDomain = modelLoader.loadModel(
						currentIterationModelName, corpus.domain,
						currentIterationModelDirectory);
				System.out.println("Loaded the model of domain "
						+ corpus.domain);
				topicModelList_CurrentIteration.add(modelForDomain);
			} else {
				// Run the model on each domain.
				// Construct all the parameters needed to run the model.
				TopicModelParameters param = new TopicModelParameters(corpus,
						nTopics, cmdOption);

				param.modelName = currentIterationModelName;
				param.outputModelDirectory = currentIterationModelDirectory;
				param.topicModelList_LastIteration = topicModelList_LastIteration;

				threadPool.addTask(corpus, param);
			}
		}
		threadPool.awaitTermination();
		topicModelList_CurrentIteration.addAll(threadPool.topicModelList);
		// Sort the topic model list based on the domain name alphabetically.
		Collections.sort(topicModelList_CurrentIteration,
				new Comparator<TopicModel>() {
					@Override
					public int compare(TopicModel o1, TopicModel o2) {
						return o1.corpus.domain.toLowerCase().compareTo(
								o2.corpus.domain.toLowerCase());
					}
				});
		return topicModelList_CurrentIteration;
	}

	/**
	 * 2. Evaluate the topic model. Read the topic model of each domain and run
	 * the evaluation.
	 */
	private void evaluateTopicModelForOneLearningIteration(
			List<TopicModel> topicModelList, String modelName,
			String currentIterationRootDirectory) {
		String topicCoherenceFilepath = currentIterationRootDirectory
				+ "Results" + File.separator + "TopicCoherence.txt";
		if (new File(topicCoherenceFilepath).exists()) {
			// If the Topic Coherence result file already exists, then skip the
			// evaluation.
			System.out.println("The Topic Coherence results alreay exist!");
			return;
		}

		// Write the Topic Coherence scores of the model of all domains into a
		// file.
		StringBuilder sbTopicCoherence = new StringBuilder();
		for (TopicModel topicModel : topicModelList) {
			// Evaluate topic coherence.
			double topicCoherence = 0.0;
			// Evaluate on the same corpus.
			topicCoherence = topicModel.getAverageTopicCoherence();

			// topicCoherenceList.add(topicCoherence);
			sbTopicCoherence.append(topicCoherence);
			sbTopicCoherence.append(System.lineSeparator());
		}
		if (sbTopicCoherence.length() > 0) {
			// Do not write the empty file.
			FileReaderAndWriter.writeFile(topicCoherenceFilepath,
					sbTopicCoherence.toString());
		}

		String numberOfTopicsFilePath = currentIterationRootDirectory
				+ "Results" + File.separator + "NumberOfTopics.txt";
		if (new File(numberOfTopicsFilePath).exists()) {
			// If the Topic Coherence result file already exists, then skip the
			// evaluation.
			System.out.println("The Topic Coherence results alreay exist!");
			return;
		}
		// Write the number of topics from the model in each domain into a
		// file.
		StringBuilder sbNumberOfTopics = new StringBuilder();
		for (TopicModel topicModel : topicModelList) {
			sbNumberOfTopics.append(topicModel.param.T);
			sbNumberOfTopics.append(System.lineSeparator());
		}
		if (sbNumberOfTopics.length() > 0) {
			// Do not write the empty file.
			FileReaderAndWriter.writeFile(numberOfTopicsFilePath,
					sbNumberOfTopics.toString());
		}
	}

}
