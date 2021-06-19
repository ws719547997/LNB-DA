package task;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import nlp.EntropyHelper;
import nlp.Topic;
import nlp.Topics;
import topicmodel.ModelLoader;
import topicmodel.TopicModel;
import utility.ExceptionUtility;
import utility.FileReaderAndWriter;
import utility.ItemWithValue;

public class SentimentWordsPolarityExtractionTask {
	// Inputs.
	public String modelName = "JST_Seed";
	public String modelSettingFullName = "JST_Seed_Paradigm_ASUM+"; // "JST_Seed_Paradigm_ASUM";
																	// "JST_Seed_GoodAndBadOnly",
																	// "JST_Seed_Paradigm_ASUM+".
	public String inputTopicModelMultiDomainFilepath = "..\\Data\\Output\\TopicModel\\LearningIteration0\\"
			+ modelSettingFullName
			+ File.separator
			+ "DomainModels"
			+ File.separator;
	// Outputs.
	public String outputExtractedSentimentWords = "..\\Data\\Output\\ExtractedSentimentWords\\"
			+ modelSettingFullName + ".txt";
	public int twords = 20;

	public void run() {
		List<TopicModel> topicModelList = new ArrayList<TopicModel>();
		File[] topicModelFiles = new File(inputTopicModelMultiDomainFilepath)
				.listFiles();
		for (File topicModelFile : topicModelFiles) {
			String domain = topicModelFile.getName();
			ModelLoader modelLoader = new ModelLoader();
			TopicModel topicModelForThisDomain = modelLoader.loadModel(
					modelName, domain, topicModelFile.getAbsolutePath()
							+ File.separator);
			System.out.println("Loaded the model of domain " + domain);
			topicModelList.add(topicModelForThisDomain);
		}

		// Map<String, int[]> mpWordToSentimentCounts = new HashMap<String,
		// int[]>();
		// for (TopicModel topicModelForThisDomain : topicModelList) {
		// for (int s = 0; s < topicModelForThisDomain.param.S; ++s) {
		// Topics topics = topicModelForThisDomain.getTopics(twords, s);
		// for (Topic topic : topics) {
		// for (ItemWithValue iwv : topic.topWordList) {
		// String wordstr = iwv.getItem().toString();
		// if (!mpWordToSentimentCounts.containsKey(wordstr)) {
		// mpWordToSentimentCounts.put(wordstr, new int[] { 0,
		// 0, 0 });
		// }
		// int[] sentimentCounts = mpWordToSentimentCounts
		// .get(wordstr);
		// ++sentimentCounts[s];
		// }
		// }
		// }
		// }

		Map<String, int[]> mpWordToSentimentCounts = new HashMap<String, int[]>();
		for (TopicModel topicModelForThisDomain : topicModelList) {
			Topics topics = topicModelForThisDomain.getTopics(twords);
			for (Topic topic : topics) {
				int sentimentForThisTopic = -1;
				for (String positiveSeed : topicModelForThisDomain.seeds.positiveSeeds) {
					if (topic.containsWord(positiveSeed)) {
						sentimentForThisTopic = 0;
					}
				}
				for (String negativeSeed : topicModelForThisDomain.seeds.negativeSeeds) {
					if (topic.containsWord(negativeSeed)) {
						ExceptionUtility
								.assertAsException(
										sentimentForThisTopic == -1
												|| sentimentForThisTopic == topicModelForThisDomain.param.S - 1,
										"The topic contains both positive and negative sentiment seeds.");
						sentimentForThisTopic = topicModelForThisDomain.param.S - 1;
					}
				}
				if (sentimentForThisTopic == -1) {
					// The topic does not contain either sentiment seeds, and
					// thus hard to determine the polarity.
					continue;
				}
				for (ItemWithValue iwv : topic.topWordList) {
					String wordstr = iwv.getItem().toString();
					if (!mpWordToSentimentCounts.containsKey(wordstr)) {
						mpWordToSentimentCounts.put(wordstr, new int[] { 0, 0,
								0 });
					}
					int[] sentimentCounts = mpWordToSentimentCounts
							.get(wordstr);
					++sentimentCounts[sentimentForThisTopic];
				}
			}
		}

		// List<ItemWithValue> rankedWordWithEntropy =
		// rankWordsBySentimentEntropy(mpWordToSentimentCounts);
		List<ItemWithValue> rankedWordWithEntropy = rankWordsByPositiveNegativeCountDiffernce(
				mpWordToSentimentCounts, topicModelList.get(0).param.S);
		outputWordsToFile(outputExtractedSentimentWords, rankedWordWithEntropy,
				mpWordToSentimentCounts);
	}

	public List<ItemWithValue> rankWordsBySentimentEntropy(
			Map<String, int[]> mpWordToSentimentCounts) {
		List<ItemWithValue> wordWithEntropy = new ArrayList<ItemWithValue>();
		for (Map.Entry<String, int[]> entry : mpWordToSentimentCounts
				.entrySet()) {
			String wordstr = entry.getKey().toString();
			int[] counts = entry.getValue();
			double entropy = EntropyHelper.getEntropy(counts);
			wordWithEntropy.add(new ItemWithValue(wordstr, entropy));
		}
		Collections.sort(wordWithEntropy, Collections.reverseOrder());
		return wordWithEntropy;
	}

	public List<ItemWithValue> rankWordsByPositiveNegativeCountDiffernce(
			Map<String, int[]> mpWordToSentimentCounts, int S) {
		List<ItemWithValue> wordWithEntropy = new ArrayList<ItemWithValue>();
		for (Map.Entry<String, int[]> entry : mpWordToSentimentCounts
				.entrySet()) {
			String wordstr = entry.getKey().toString();
			int[] counts = entry.getValue();
			wordWithEntropy.add(new ItemWithValue(wordstr, counts[0]
					- counts[S - 1]));
		}
		Collections.sort(wordWithEntropy);
		return wordWithEntropy;
	}

	public void outputWordsToFile(String outputExtractedSentimentWords,
			List<ItemWithValue> rankedWordWithEntropy,
			Map<String, int[]> mpWordToSentimentCounts) {
		StringBuilder sbOutput = new StringBuilder();
		for (ItemWithValue iwv : rankedWordWithEntropy) {
			String wordstr = iwv.getItem().toString();
			double entropy = iwv.getValue();
			int[] counts = mpWordToSentimentCounts.get(wordstr);
			sbOutput.append(wordstr + "\t" + entropy);
			for (int count : counts) {
				sbOutput.append("\t" + count);
			}
			sbOutput.append(System.lineSeparator());
		}
		FileReaderAndWriter.writeFile(outputExtractedSentimentWords,
				sbOutput.toString());

	}
}
