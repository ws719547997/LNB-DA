package topicmodel;

import java.io.File;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

import utility.ExceptionUtility;
import utility.FileReaderAndWriter;
import main.CmdOption;
import nlp.Corpus;

/**
 * This class contains all the parameters for a topic model.
 */
public class TopicModelParameters {
	// ------------------------------------------------------------------------
	// Task Specific Parameters
	// ------------------------------------------------------------------------

	public String modelName = null;
	public String domain = null; // The name of the domain.

	// ------------------------------------------------------------------------
	// General Parameters for topic models
	// ------------------------------------------------------------------------

	public int D = 0; // #Documents.
	public int V = 0; // #Words.
	public int T = 0; // #Topics.
	public int S = 3; // #Sentiments.

	// The number of iterations for burn-in period.
	public int nBurnin = 200;
	// The number of Gibbs sampling iterations.
	public int nIterations = 2000;
	// The length of interval to sample for calculating posterior distribution.
	public int sampleLag = 20;

	/******************* Hyperparameters *********************/
	public double alpha = 1.0;
	public double beta = 0.1;
	public double gamma = 1; // Following the ASUM model.

	// Random seed.
	public int randomSeed = 0;

	// ------------------------------------------------------------------------
	// Output
	// ------------------------------------------------------------------------
	public int twords = 0; // Print out top words per each topic.
	public String outputModelDirectory = null;

	// ------------------------------------------------------------------------
	// Knowledge
	// ------------------------------------------------------------------------
	// The topic models from last iteration, used to learn knowledge.
	public List<TopicModel> topicModelList_LastIteration;
	public boolean includeCurrentDomainAsKnowledgeExtraction = true;

	// Seeds for JST_Seed model
	public String seedName = "Custom_Paradigm_ASUM+";// "GoodAndBadOnly",
	// "Paradigm_ASUM",
	// "Paradigm_ASUM+";
	public String seedSentimentDirectory = "..\\Data\\Input\\SeedsForTopicModel\\"
			+ seedName + File.separator;

	private TopicModelParameters() {

	}

	public TopicModelParameters(Corpus corpus, int nTopics, CmdOption cmdOption) {
		domain = corpus.domain;

		D = corpus.getNoofDocuments();
		V = corpus.vocab.size();
		T = nTopics;

		nBurnin = cmdOption.nBurnin;
		nIterations = cmdOption.nIterations;
		sampleLag = cmdOption.sampleLag;

		alpha = cmdOption.alpha;
		beta = cmdOption.beta;

		randomSeed = cmdOption.randomSeed;

		twords = cmdOption.twords;

		includeCurrentDomainAsKnowledgeExtraction = cmdOption.includeCurrentDomainAsKnowledgeExtraction;
	}

	/**
	 * Read the model parameters from a file.
	 */
	public static TopicModelParameters getModelParameters(String filePath) {
		try {
			TopicModelParameters param = new TopicModelParameters();

			ArrayList<String> lines = FileReaderAndWriter
					.readFileAllLines(filePath);
			for (String line : lines) {
				if (line.trim().equals("")) {
					continue;
				}
				String[] strSplits = line.split("=");
				if (strSplits.length > 2) {
					ExceptionUtility
							.throwAndCatchException("The format of model parameters file is not correct!");
				}
				String key = strSplits[0];
				String valueString = "";
				if (strSplits.length >= 2) {
					valueString = strSplits[1];
				}

				try {
					Field field = param.getClass().getDeclaredField(key);
					field.setAccessible(true);

					if (field.getType().getSimpleName().equals("int")) {
						int value = Integer.valueOf(valueString);
						field.set(param, value);
					} else if (field.getType().getSimpleName().equals("double")) {
						double value = Double.valueOf(valueString);
						field.set(param, value);
					} else if (field.getType().getSimpleName()
							.equals("boolean")) {
						Boolean value = Boolean.valueOf(valueString);
						field.set(param, value);
					} else if (field.getType().getSimpleName().equals("String")) {
						String value = valueString;
						field.set(param, value);
					}
				} catch (Exception ex) {
					continue;
				}
			}
			return param;
		} catch (Exception ex) {
			ex.printStackTrace();
			return null;
		}
	}

	/**
	 * Print model parameters into a file.
	 */
	public void printToFile(String filePath) {
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
			FileReaderAndWriter.writeFile(filePath, sbOutput.toString());
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}
}
