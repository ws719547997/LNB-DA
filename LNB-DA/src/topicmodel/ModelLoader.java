package topicmodel;

import java.io.File;
import java.util.ArrayList;

import knowledge.SentimentSeeds;
import nlp.Corpus;
import nlp.Documents;
import utility.ExceptionUtility;
import utility.FileReaderAndWriter;

public class ModelLoader {
	/**
	 * Load the topic model from the modelDirectory. For different model names,
	 * we may need to assign different variables.
	 */
	public TopicModel loadModel(String modelName, String domain,
			String modelDirectory) {
		TopicModel model = null;
		try {
			// Load the information for general topic models.
			// Read the model parameters.
			TopicModelParameters param = loadModelParameters(domain,
					modelDirectory);
			// Read the corpus.
			Corpus corpus = loadCorpus(domain, modelDirectory);
			// Read the reviews.
			corpus.documents = Documents.readDocuments(domain, modelDirectory
					+ File.separator + domain + ModelPrinter.reviewsSuffix);

			int[][] z = loadTopicWordAssignment(modelDirectory + domain
					+ ModelPrinter.tassignSuffix);
			if (modelName.equals("LDA")) {
				// Load posterior distribution.
				double[][] twdist = loadTwoDimentionalDistribution(modelDirectory
						+ File.separator
						+ domain
						+ ModelPrinter.topicWordDistSuff);
				double[][] dtdist = loadTwoDimentionalDistribution(modelDirectory
						+ File.separator
						+ domain
						+ ModelPrinter.documentTopicDistSuff);
				return new LDA(corpus, param, z, twdist, dtdist);
			} else if (modelName.startsWith("JST")) {
				int[][] y = loadTopicWordAssignment(modelDirectory + domain
						+ ModelPrinter.sassignSuffix);
				double[][] dsdist = loadTwoDimentionalDistribution(modelDirectory
						+ File.separator
						+ domain
						+ ModelPrinter.documentSentimentDistSuff);
				// Read the sentiment topic word distribution.
				double[][][] sentimentTopicWordDistribution = new double[param.S][][];
				for (int s = 0; s < param.S; ++s) {
					sentimentTopicWordDistribution[s] = loadTwoDimentionalDistribution(modelDirectory
							+ File.separator
							+ domain
							+ "-"
							+ s
							+ ModelPrinter.sentimentTopicWordDistSuff);
				}
				if (modelName.equals("JST")) {
					return new JST(corpus, param, z, y, dsdist,
							sentimentTopicWordDistribution);
				} else if (modelName.equals("JST_Seed")) {
					SentimentSeeds seeds = SentimentSeeds
							.readSeedsFromDirectory(domain, modelDirectory);
					return new JST_Seed(corpus, param, z, y, dsdist, seeds,
							sentimentTopicWordDistribution);
				}
			} else {
				ExceptionUtility
						.throwAndCatchException("The model name is not recognizable!");
			}
		} catch (Exception ex) {
			System.out.println("Error while loading the topic model: "
					+ ex.getMessage());
			ex.printStackTrace();
		}
		return model;
	}

	public TopicModel loadModelAsLDA(String modelName, String domain,
			String modelDirectory) {
		// Load the information for general topic models.
		TopicModelParameters param = loadModelParameters(domain, modelDirectory);
		Corpus corpus = loadCorpus(domain, modelDirectory);

		// Load posterior distribution.
		double[][] twdist = loadTwoDimentionalDistribution(modelDirectory
				+ File.separator + domain + ModelPrinter.topicWordDistSuff);
		double[][] dtdist = loadTwoDimentionalDistribution(modelDirectory
				+ File.separator + domain + ModelPrinter.documentTopicDistSuff);

		int[][] z = loadTopicWordAssignment(modelDirectory + domain
				+ ModelPrinter.tassignSuffix);
		return new LDA(corpus, param, z, twdist, dtdist);
	}

	public TopicModelParameters loadModelParameters(String domain,
			String modelDirectory) {
		String filepath = modelDirectory + domain
				+ ModelPrinter.modelParamSuffix;
		return TopicModelParameters.getModelParameters(filepath);
	}

	public int[][] loadTopicWordAssignment(String filepath) {
		ArrayList<String> lines = FileReaderAndWriter
				.readFileAllLines(filepath);

		int D = lines.size();
		int[][] z = new int[D][];
		for (int d = 0; d < D; ++d) {
			String line = lines.get(d);
			if (line.trim().length() == 0) {
				continue;
			}
			// Read the word with its topic.
			String[] wordsWithTopics = line.split("[ \t\r\n]");
			int N = wordsWithTopics.length;
			z[d] = new int[N];
			for (int n = 0; n < N; ++n) {
				String wordWithTopic = wordsWithTopics[n];
				String[] strSplits = wordWithTopic.split(":");
				if (strSplits.length != 2) {
					ExceptionUtility
							.throwAndCatchException("Incorrect format of word with topic!");
				}
				// String wordStr = strSplits[0];
				int topic = Integer.parseInt(strSplits[1]);
				z[d][n] = topic;
			}
		}
		return z;
	}

	public double[][] loadTwoDimentionalDistribution(String filepath) {
		ArrayList<String> lines = FileReaderAndWriter
				.readFileAllLines(filepath);

		int D1 = lines.size();
		double[][] twdist = new double[D1][];
		for (int d1 = 0; d1 < D1; ++d1) {
			String line = lines.get(d1);
			String[] strSplits = line.split("[ \t\r\n]");
			int D2 = strSplits.length;
			twdist[d1] = new double[D2];
			for (int d2 = 0; d2 < D2; ++d2) {
				twdist[d1][d2] = Double.parseDouble(strSplits[d2]);
			}
		}
		return twdist;
	}

	public Corpus loadCorpus(String domain, String modelDirectory) {
		String docsFilepath = modelDirectory + domain + ModelPrinter.docsSuffix;
		String vocabFilepath = modelDirectory + domain
				+ ModelPrinter.vocabSuffix;
		String oriContextFilepath = modelDirectory + domain
				+ ModelPrinter.oriContextSuffix;
		return Corpus.getCorpusFromFile(domain, docsFilepath, vocabFilepath,
				oriContextFilepath);
	}
}
