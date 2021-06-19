package topicmodelevaluation;

import java.util.ArrayList;

import nlp.Corpus;
import nlp.Topic;
import nlp.Topics;
import topicmodel.TopicModel;
import utility.ItemWithValue;

/**
 * Compute the average Topic Coherence score over topics according to the
 * following paper:
 * 
 * David Mimno, et al. "Optimizing semantic coherence in topic models." EMNLP
 * 2011.
 */
public class TopicCoherenceEntry {
	private final static int twords = 30; // As suggested in David Mimno's EMNLP
											// 2011 paper.

	public static double computeTopicCoherenceGivenATopicModel(
			TopicModel model, Corpus corpus) {
		double averageScore = 0.0;

		ArrayList<ArrayList<ItemWithValue>> topWordStrsUnderTopics = model
				.getTopWordStrsWithProbabilitiesUnderTopics(twords);
		for (int t = 0; t < topWordStrsUnderTopics.size(); ++t) {
			ArrayList<ItemWithValue> topWords = topWordStrsUnderTopics.get(t);
			double value = 0.0;

			for (int i = 0; i < topWords.size(); ++i) {
				ItemWithValue iwp1 = topWords.get(i);
				String wordStr1 = iwp1.getItem().toString();
				for (int j = 0; j < i; ++j) {
					ItemWithValue iwp2 = topWords.get(j);
					String wordStr2 = iwp2.getItem().toString();
					double codocFrequency = corpus.getCoDocumentFrequency(
							wordStr1, wordStr2);
					double docFrequnety = corpus.getDocumentFrequency(wordStr2);
					value += Math.log((codocFrequency + 1) / docFrequnety);
				}
			}
			averageScore += value;
		}
		averageScore /= topWordStrsUnderTopics.size();
		return averageScore;
	}

	public static double computeTopicCoherenceGivenTopics(Topics topics,
			Corpus corpus) {
		double averageScore = 0.0;

		for (int t = 0; t < topics.size(); ++t) {
			Topic topic = topics.topicList.get(t);
			ArrayList<ItemWithValue> topWords = topic.topWordList;
			double value = 0.0;

			for (int i = 0; i < topWords.size(); ++i) {
				ItemWithValue iwp1 = topWords.get(i);
				String wordStr1 = iwp1.getItem().toString();
				for (int j = 0; j < i; ++j) {
					ItemWithValue iwp2 = topWords.get(j);
					String wordStr2 = iwp2.getItem().toString();
					double codocFrequency = corpus.getCoDocumentFrequency(
							wordStr1, wordStr2);
					double docFrequnety = corpus.getDocumentFrequency(wordStr2);
					value += Math.log((codocFrequency + 1) / docFrequnety);
				}
			}
			averageScore += value;
		}
		averageScore /= topics.size();
		return averageScore;
	}
}
