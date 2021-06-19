package topicmodel;

import java.util.ArrayList;

import utility.FileOneByOneLineWriter;
import utility.FileReaderAndWriter;
import utility.ItemWithValue;
import nlp.Corpus;
import nlp.Topic;
import nlp.Topics;
import nlp.Vocabulary;

/**
 * This class prints the outputs of a topic model.
 * 
 * For each model, we print:
 * 
 * 1. Model parameters
 * 
 * 2. z[][]
 * 
 * 3. Topic-word distribution
 * 
 * 4. Top words under each topic.
 * 
 * 5. Vocabulary
 */
public class ModelPrinter {
	// Suffix for model parameters.
	public static final String modelParamSuffix = ".param";
	// Suffix for topic word assignment file.
	public static final String tassignSuffix = ".tassign";
	// Suffix for document topic distribution.
	public static final String documentTopicDistSuff = ".dtopicdist";
	// Suffix for topic word distribution.
	public static final String topicWordDistSuff = ".twdist";
	// Suffix for file containing top words per topic.
	public static String twordsSuffix = ".twords";
	// Suffix for file containing the documents in the corpus.
	public static String docsSuffix = ".docs";
	// Suffix for file containing the vocabulary.
	public static String vocabSuffix = ".vocab";
	// Suffix for file containing the ori context.
	public static String oriContextSuffix = ".ori";
	// Suffix for file containing the knowledge.
	public static String knowledgeSuffix = ".knowl";

	// Suffix for file containing the reviews in the corpus.
	public static String reviewsSuffix = ".reviews";

	/*********************** For JST model. ************************/
	// Suffix for sentiment word assignment file.
	public static final String sassignSuffix = ".sassign";
	// Suffix for sentiment topic word assignment file.
	public static final String stassignSuffix = ".stassign";
	// Suffix for document sentiment distribution.
	public static final String documentSentimentDistSuff = ".dSentiDist";
	// Suffix for document sentiment topic distribution.
	public static String documentSentimentTopicDistSuff = ".dSentiTopicDist";
	// Suffix for sentiment topic word distribution.
	public static final String sentimentTopicWordDistSuff = ".stwdist";

	/*********************** Sentiment Seeds ************************/
	public static final String positiveSeedSuffix = ".pseeds";
	public static final String negativeSeedSuffix = ".nseeds";

	private TopicModel model = null;

	public ModelPrinter(TopicModel model2) {
		model = model2;
	}

	/**
	 * Print the model.
	 */
	public void printModel(String outputDirectory) {
		try {
			String domain = model.param.domain;
			printModelParameters(model.param, outputDirectory + domain
					+ modelParamSuffix);
			printTopicWordAssignment(model.z, model.corpus, outputDirectory
					+ domain + tassignSuffix);

			if (model.param.modelName.equals("LDA")) {
				printTwoDimensionalDistribution(
						model.getDocumentTopicDistrbution(), outputDirectory
								+ domain + documentTopicDistSuff);
				printTwoDimensionalDistribution(
						model.getTopicWordDistribution(), outputDirectory
								+ domain + topicWordDistSuff);
				ArrayList<ArrayList<ItemWithValue>> topWordsUnderTopics = model
						.getTopWordStrsWithProbabilitiesUnderTopics(model.param.twords);
				printTopWordsUnderTopics(topWordsUnderTopics, outputDirectory
						+ domain + twordsSuffix);
			} else if (model.param.modelName.startsWith("JST")) {
				// Print the document sentiment distribution.
				printTwoDimensionalDistribution(
						model.getDocumentSentimentDistribution(),
						outputDirectory + domain + documentSentimentDistSuff);
				printTopicWordAssignment(model.y, model.corpus, outputDirectory
						+ domain + sassignSuffix);
				// Print the sentiment topic word distribution.
				printSentimentTopicWordDistribution(
						model.getSentimentTopicWordDistribution(),
						model.param.S, outputDirectory, domain,
						topicWordDistSuff);
				printSentimentTopicWordAssignment(model.y, model.z,
						model.corpus, outputDirectory + domain + stassignSuffix);
				ArrayList<ArrayList<ItemWithValue>> topWordsUnderTopics = model
						.getTopWordStrsWithProbabilitiesUnderTopics(model.param.twords);
				printTopWordsUnderSentimentsAndTopics(topWordsUnderTopics,
						model.param.S, model.param.T, outputDirectory + domain
								+ twordsSuffix);
				model.printSentimentSeeds(outputDirectory + domain
						+ positiveSeedSuffix, outputDirectory + domain
						+ negativeSeedSuffix);
			}
			// Print the reviews.
			model.corpus.documents.printToFile(outputDirectory + domain
					+ reviewsSuffix);
			// Print the corpus.
			printDocs(model.corpus.docs, outputDirectory + domain + docsSuffix);
			printVocabulary(model.corpus.vocab, outputDirectory + domain
					+ vocabSuffix);

			// For knowledge-based topic models only.
			model.printKnowledge(outputDirectory + domain + knowledgeSuffix);
		} catch (Exception ex) {
			System.out.println("Error while printing the topic model: "
					+ ex.getMessage());
			ex.printStackTrace();
		}
	}

	private void printModelParameters(TopicModelParameters param,
			String filePath) {
		param.printToFile(filePath);
	}

	private void printTopicWordAssignment(int[][] z, Corpus corpus,
			String filePath) {
		assert (z != null && z.length != 0 && z[0].length != 0) : "The array z is not correct!";

		FileOneByOneLineWriter writer = new FileOneByOneLineWriter(filePath);

		int D = z.length;
		for (int d = 0; d < D; ++d) {
			StringBuilder sbLine = new StringBuilder();
			int N = z[d].length;
			for (int n = 0; n < N; ++n) {
				sbLine.append(corpus.vocab
						.getWordstrByWordid(corpus.docs[d][n])
						+ ":"
						+ z[d][n]
						+ " ");
			}
			writer.writeLine(sbLine.toString().trim());
		}
		writer.close();
	}

	private void printTwoDimensionalDistribution(double[][] dist,
			String filePath) {
		assert (dist != null && dist.length != 0 && dist[0].length != 0) : "The two dimensional distribution is not correct!";

		FileOneByOneLineWriter writer = new FileOneByOneLineWriter(filePath);

		int D1 = dist.length;
		for (int d1 = 0; d1 < D1; ++d1) {
			StringBuilder sbLine = new StringBuilder();
			int D2 = dist[d1].length;
			for (int d2 = 0; d2 < D2; ++d2) {
				sbLine.append(dist[d1][d2] + " ");
			}
			writer.writeLine(sbLine.toString().trim());
		}
		writer.close();
	}

	private void printTopWordsUnderTopics(
			ArrayList<ArrayList<ItemWithValue>> topWordsUnderTopics,
			String filepath) {
		StringBuilder sbOutput = new StringBuilder();

		// Print out the first row with "Topic k".
		for (int k = 0; k < topWordsUnderTopics.size(); ++k) {
			sbOutput.append("Topic " + k);
			sbOutput.append("\t");
		}
		sbOutput.append(System.getProperty("line.separator"));

		for (int pos = 0; pos < topWordsUnderTopics.get(0).size(); ++pos) {
			StringBuilder sbLine = new StringBuilder();
			for (int k = 0; k < topWordsUnderTopics.size(); ++k) {
				ArrayList<ItemWithValue> topWords = topWordsUnderTopics.get(k);
				ItemWithValue iwv = topWords.get(pos);
				String wordstr = iwv.getItem().toString();
				sbLine.append(wordstr);
				sbLine.append("\t");
			}
			sbOutput.append(sbLine.toString().trim());
			sbOutput.append(System.getProperty("line.separator"));
		}
		FileReaderAndWriter.writeFile(filepath, sbOutput.toString());
	}

	public void printTopWordsUnderTopics(Topics topics, int twords,
			String filepath) {
		StringBuilder sbOutput = new StringBuilder();

		// Print out the first row with "Topic k".
		for (int k = 0; k < topics.size(); ++k) {
			sbOutput.append("Topic " + k);
			sbOutput.append("\t");
		}
		sbOutput.append(System.getProperty("line.separator"));

		for (int pos = 0; pos < topics.topicList.get(0).topWordList.size()
				&& pos < twords; ++pos) {
			StringBuilder sbLine = new StringBuilder();
			for (int k = 0; k < topics.size(); ++k) {
				ArrayList<ItemWithValue> topWords = topics.topicList.get(k).topWordList;
				ItemWithValue iwv = topWords.get(pos);
				String wordstr = iwv.getItem().toString();
				sbLine.append(wordstr);
				sbLine.append("\t");
			}
			sbOutput.append(sbLine.toString().trim());
			sbOutput.append(System.getProperty("line.separator"));
		}
		FileReaderAndWriter.writeFile(filepath, sbOutput.toString());
	}

	public void printTopWordsAsRows(Topics topics, int twords, String filepath) {
		StringBuilder sbOutput = new StringBuilder();

		// Print out the first row with "Topic k".
		for (Topic topic : topics.topicList) {
			StringBuilder sbLine = new StringBuilder();
			for (ItemWithValue iwv : topic.topWordList) {
				String wordstr = iwv.getItem().toString();
				sbLine.append(wordstr);
				sbLine.append(" ");
			}
			sbOutput.append(sbLine.toString().trim());
			sbOutput.append(System.getProperty("line.separator"));
		}
		FileReaderAndWriter.writeFile(filepath, sbOutput.toString());
	}

	public void printDocs(int[][] docs, String filePath) {
		StringBuilder sbOutput = new StringBuilder();
		for (int[] doc : docs) {
			StringBuilder sbLine = new StringBuilder();
			for (int word : doc) {
				sbLine.append(word + " ");
			}
			sbOutput.append(sbLine.toString().trim());
			sbOutput.append(System.getProperty("line.separator"));
		}
		FileReaderAndWriter.writeFile(filePath, sbOutput.toString());
	}

	public void printVocabulary(Vocabulary vocab, String filePath) {
		vocab.printToFile(filePath);
	}

	/*********************** For JST model. ************************/
	private void printSentimentTopicWordAssignment(int[][] y, int[][] z,
			Corpus corpus, String filepath) {
		assert (z != null && z.length != 0 && z[0].length != 0) : "The array z is not correct!";

		FileOneByOneLineWriter writer = new FileOneByOneLineWriter(filepath);

		int D = z.length;
		for (int d = 0; d < D; ++d) {
			StringBuilder sbLine = new StringBuilder();
			int N = z[d].length;
			for (int n = 0; n < N; ++n) {
				sbLine.append(corpus.vocab
						.getWordstrByWordid(corpus.docs[d][n])
						+ ":"
						+ y[d][n]
						+ ":" + z[d][n] + " ");
			}
			writer.writeLine(sbLine.toString().trim());
		}
		writer.close();
	}

	private void printSentimentTopicWordDistribution(
			double[][][] sentimentTopicWordDistribution, int S,
			String outputDirectory, String domain, String topicworddistsuff2) {
		for (int s = 0; s < S; ++s) {
			printTwoDimensionalDistribution(sentimentTopicWordDistribution[s],
					outputDirectory + domain + "-" + s
							+ sentimentTopicWordDistSuff);
		}
	}

	private void printTopWordsUnderSentimentsAndTopics(
			ArrayList<ArrayList<ItemWithValue>> topWordsUnderTopics, int S,
			int T, String filepath) {
		StringBuilder sbOutput = new StringBuilder();

		// Print out the first row with "Topic k".
		for (int s = 0; s < S; ++s) {
			for (int t = 0; t < T; ++t) {
				sbOutput.append("S-" + s + " T-" + t);
				sbOutput.append("\t");
			}
		}
		sbOutput.append(System.getProperty("line.separator"));

		for (int pos = 0; pos < topWordsUnderTopics.get(0).size(); ++pos) {
			StringBuilder sbLine = new StringBuilder();
			for (int k = 0; k < topWordsUnderTopics.size(); ++k) {
				ArrayList<ItemWithValue> topWords = topWordsUnderTopics.get(k);
				ItemWithValue iwv = topWords.get(pos);
				String wordstr = iwv.getItem().toString();
				sbLine.append(wordstr);
				sbLine.append("\t");
			}
			sbOutput.append(sbLine.toString().trim());
			sbOutput.append(System.getProperty("line.separator"));
		}
		FileReaderAndWriter.writeFile(filepath, sbOutput.toString());

	}
}
