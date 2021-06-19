package nlp;

import java.util.ArrayList;

import utility.ItemWithValue;

/**
 * A topic is represented by its top words.
 */
public class Topic {
	// The top words with their original probabilities.
	public ArrayList<ItemWithValue> topWordList = null;

	public Topic(ArrayList<ItemWithValue> topWordList2) {
		topWordList = topWordList2;
	}

	public boolean containsWord(String wordstr) {
		for (ItemWithValue iwv : topWordList) {
			if (iwv.getItem().toString().equals(wordstr)) {
				return true;
			}
		}
		return false;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		for (ItemWithValue iwv : topWordList) {
			String word = iwv.getItem().toString();
			sb.append(word);
			sb.append(' ');
		}
		return sb.toString().trim();
	}

}
