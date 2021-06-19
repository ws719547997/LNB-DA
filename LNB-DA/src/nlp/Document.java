package nlp;

import java.util.HashSet;
import java.util.Set;

import org.apache.commons.lang3.StringUtils;

import feature.Feature;
import feature.Features;

/**
 * Each document refers to a review.
 */
public class Document implements Cloneable {
	public String reviewId = null;
	public String domain = null;

	public String productId = null;

	public String title = null;
	public String content = null;
	public String text = null; // For model learning and testing.
	public String textPreprocessed = null; // The concatenation of words
	// (array).

	public int ratingScore = 0; // 1-5 stars.
	public String label = null;
	public String predict = null;
	public double probOfPositivePredict = 0.0;
	public double probOfNegativePredict = 0.0;

	public int docIndex = 0; // Set from Documents.
	public String[] words_ori = null; // The words in the documents (before pre-processing.).
	public String[] words = null; // only valid words.
								// The invalid words are converted to empty string.

	// The features to represent this document. The features are generated in
	// FeatureGenerator.
	public Features featuresForNaiveBayes = null; // For Naive Bayes, the
													// feature value is not
													// used.
	public Set<Feature> featuresForSVM = null; // For SVM, each feature only
												// appear once with the feature
												// value (e.g., TF-IDF) used.

	/**
	 * For sentiment classification (review data).
	 * including data pre-processing, e.g., stop-words, and negation-words
	 * index	Domain	Label	Rating	RatingScore	Review
	 */
	public Document(String reviewID2, String domain2, String labelStr, int ratingScore2, String content2) {
		reviewId = reviewID2;
		domain = domain2;
		// convert label to +1 and -1
		label = Label.convertPostiveNegativeToPlusOneMinusOne(labelStr);
		ratingScore = ratingScore2;

		// productId = productId2;
		// title = title2;
		content = content2;

		// // Convert "-" and "/" to "".
		// text = content.replaceAll("[-/]", "").trim();
		// // Remove all the special characters including punctuations.
		// text = text.replaceAll("[^\\p{L}\\p{Nd}]+", " ").trim();

		StringBuilder sbTextPreprocessed = new StringBuilder();
		text = content.trim(); // eliminates leading and trailing whitespace
		words_ori = StringUtils.split(text); // default using whitespace as the separator
		words = new String[words_ori.length];
		boolean negationMode = false;
		for (int i = 0; i < words_ori.length; ++i) {

			// Consider negation (following Pang and Lee, 2002).
			// We added the tag NOT to every word between a negation word
			// ("not", "isn't", "didn't", etc.) and the first punctuation mark
			// following the negation word.
			if (NegationWordHelper.isNegationWord(words_ori[i])) {
				negationMode = true;
				words[i] = "";
				continue;
			}

			// replace punctuations ",;.!?" with ""
			if (words_ori[i].equals(",") || words_ori[i].equals(";")
					|| words_ori[i].equals(".") || words_ori[i].equals("!")
					|| words_ori[i].equals("?")) {
				negationMode = false;
				words[i] = "";
				continue;
			}

			// word validity, i.e., whether this word is a stop-word or digit
			if (WordValidityHelper.getInstance().isValid(words_ori[i], domain)) {
				if (negationMode) {
					words[i] = "not|" + words_ori[i];
					// System.out.println(words[i]);
				} else {
					words[i] = words_ori[i];
				}
				sbTextPreprocessed.append(words[i]).append(" ");
			} else {
				// Invalid word, convert it into an empty string.
				// System.out.println("Invalid word " + words_ori[i]);
				words[i] = "";
			}
		}
		// pre-processed reviews
		textPreprocessed = sbTextPreprocessed.toString().trim().toLowerCase();

		featuresForNaiveBayes = new Features();
	}

	/**
	 * 加入一个针对中文的重载
	 */
	public Document(String reviewID2, String domain2, String labelStr, int ratingScore2, String content2,String flag) {
		reviewId = reviewID2;
		domain = domain2;
		// convert label to +1 and -1
		label = Label.convertPostiveNegativeToPlusOneMinusOne(labelStr);
		ratingScore = ratingScore2;

		// productId = productId2;
		// title = title2;
		content = content2;

		// // Convert "-" and "/" to "".
		// text = content.replaceAll("[-/]", "").trim();
		// // Remove all the special characters including punctuations.
		// text = text.replaceAll("[^\\p{L}\\p{Nd}]+", " ").trim();

		StringBuilder sbTextPreprocessed = new StringBuilder();
		text = content.trim(); // eliminates leading and trailing whitespace
		words_ori = StringUtils.split(text); // default using whitespace as the separator
		words = new String[words_ori.length];
//		words_ori = new String[words_ori.length];
//		boolean negationMode = false;
		for (int i = 0; i < words_ori.length; ++i) {

			// Consider negation (following Pang and Lee, 2002).
			// We added the tag NOT to every word between a negation word
			// ("not", "isn't", "didn't", etc.) and the first punctuation mark
			// following the negation word.
//			if (NegationWordHelper.isNegationWord(words_ori[i])) {
//				negationMode = true;
//				words[i] = "";
//				continue;
//			}

//			// replace punctuations ",;.!?" with ""
			if (words_ori[i].equals("，") || words_ori[i].equals("；")
					|| words_ori[i].equals("。") || words_ori[i].equals("！")
					|| words_ori[i].equals("？")|| words_ori[i].equals(".")
					|| words_ori[i].equals("（")|| words_ori[i].equals("）")
					|| words_ori[i].equals("-")|| words_ori[i].equals("[")
					|| words_ori[i].equals("/")|| words_ori[i].equals("~")
					|| words_ori[i].equals("(")|| words_ori[i].equals(")")
					|| words_ori[i].equals(",")) {
				words[i] = "";
				continue;
			}

			// word validity, i.e., whether this word is a stop-word or digit
			if (WordValidityHelper.getInstance().isChineseValid(words_ori[i], domain)) {
				words[i] = words_ori[i];
				sbTextPreprocessed.append(words[i]).append(" ");
			} else {
				// Invalid word, convert it into an empty string.
				// System.out.println("Invalid word " + words_ori[i]);
				words[i] = "";
			}
		}
		// pre-processed reviews
		textPreprocessed = sbTextPreprocessed.toString().trim().toLowerCase();

		featuresForNaiveBayes = new Features();
	}

    /**
     * For sentiment classification (review data).
	 * including data pre-processing, e.g., stop-words, and negation-words
     */
    public Document(String domain2, String labelStr, int ratingScore2, String content2) {
        domain = domain2;
		// convert label to +1 and -1
        label = Label.convertPostiveNegativeToPlusOneMinusOne(labelStr);
        ratingScore = ratingScore2;
        // reviewId = reviewId2;
        // productId = productId2;
        // title = title2;
        content = content2;

        // // Convert "-" and "/" to "".
        // text = content.replaceAll("[-/]", "").trim();
        // // Remove all the special characters including punctuations.
        // text = text.replaceAll("[^\\p{L}\\p{Nd}]+", " ").trim();

        StringBuilder sbTextPreprocessed = new StringBuilder();
        text = content.trim(); // eliminates leading and trailing whitespace
        words_ori = StringUtils.split(text); // default using whitespace as the separator
        words = new String[words_ori.length];
        boolean negationMode = false;
        for (int i = 0; i < words_ori.length; ++i) {

            // Consider negation (following Pang and Lee, 2002).
            // We added the tag NOT to every word between a negation word
            // ("not", "isn't", "didn't", etc.) and the first punctuation mark
            // following the negation word.
            if (NegationWordHelper.isNegationWord(words_ori[i])) {
                negationMode = true;
                words[i] = "";
                continue;
            }

			// replace punctuations ",;.!?" with ""
            if (words_ori[i].equals(",") || words_ori[i].equals(";")
                    || words_ori[i].equals(".") || words_ori[i].equals("!")
                    || words_ori[i].equals("?")) {
                negationMode = false;
                words[i] = "";
                continue;
            }

			// word validity, i.e., whether this word is a stop-word or digit
            if (WordValidityHelper.getInstance().isValid(words_ori[i], domain)) {
                if (negationMode) {
                    words[i] = "not|" + words_ori[i];
                    // System.out.println(words[i]);
                } else {
                    words[i] = words_ori[i];
                }
                sbTextPreprocessed.append(words[i]).append(" ");
            } else {
                // Invalid word, convert it into an empty string.
                // System.out.println("Invalid word " + words_ori[i]);
                words[i] = "";
            }
        }
		// pre-processed reviews
        textPreprocessed = sbTextPreprocessed.toString().trim().toLowerCase();

        featuresForNaiveBayes = new Features();
    }

	/**
	 * For binary classification (20Newsgroup and Reuters). We do not conduct
	 * negation pre-processing here. Remain the words with punctuations. We only
	 * remove the standard stop-words.
	 */
	public Document(String domain2, int docIndex2, String content2) {
		domain = domain2;
		docIndex = docIndex2;
		content = content2;

		StringBuilder sbTextPreprocessed = new StringBuilder();
		text = content.trim();
		words_ori = StringUtils.split(text);
		words = new String[words_ori.length];
		for (int i = 0; i < words_ori.length; ++i) {
			if (!StopWordHelper.getInstance().isDomainIndepOriginalStopWord(
					words_ori[i])) {
				words[i] = words_ori[i];
				sbTextPreprocessed.append(words[i]).append(" ");
			} else {
				// Invalid word, convert it into an empty string.
				// System.out.println("Invalid word " + words_ori[i]);
				words[i] = "";
			}
		}
		textPreprocessed = sbTextPreprocessed.toString().trim();

		featuresForNaiveBayes = new Features();
	}

	@Override
	protected Object clone() throws CloneNotSupportedException {
		return (Document) super.clone();
	}

	public Document getDeepClone() {
		try {
			// Clone basic types.
			Document clone = (Document) this.clone();
			// Clone words_ori.
			clone.words_ori = new String[this.words_ori.length];
			for (int i = 0; i < this.words_ori.length; ++i) {
				clone.words_ori[i] = this.words_ori[i];
			}
			// Clone words.
			clone.words = new String[this.words.length];
			for (int i = 0; i < this.words.length; ++i) {
				clone.words[i] = this.words[i];
			}
			// Clone Features.
			clone.featuresForNaiveBayes = this.featuresForNaiveBayes
					.getDeepClone();
			// Clone featuresForSVM.
			clone.featuresForSVM = new HashSet<Feature>();
			if (this.featuresForSVM != null) {
				for (Feature feature : this.featuresForSVM) {
					clone.featuresForSVM.add(feature.getDeepClone());
				}
			}
			return clone;
		} catch (Exception ex) {
			ex.printStackTrace();
		}
		return null;
	}

	public boolean isPositive() {
		return label.startsWith("+");
	}

	public boolean isNegative() {
		return label.startsWith("-");
	}

	public String toAllFieldString() {
		// index (i.e., reviewID)	Domain	Label	Rating	Review (i.e., content)
		StringBuilder sbOutput = new StringBuilder();
		sbOutput.append(this.reviewId + "\t");
		sbOutput.append(this.domain + "\t");
		sbOutput.append(Label
				.convertPlusOneMinusOneToPostiveNegative(this.label) + "\t");
		sbOutput.append(this.ratingScore + "\t");

		sbOutput.append(this.content);
		return sbOutput.toString();
	}

	public String toAllFieldStringWithPreprocessedContent() {
		StringBuilder sbOutput = new StringBuilder();
		sbOutput.append(this.domain + "\t");
		sbOutput.append(Label
				.convertPlusOneMinusOneToPostiveNegative(this.label) + "\t");
		sbOutput.append(this.ratingScore + "\t");
		sbOutput.append(this.reviewId + "\t");
		sbOutput.append(this.productId + "\t");
		sbOutput.append(this.title + "\t");
		sbOutput.append(this.textPreprocessed);
		return sbOutput.toString();
	}

	/**
	 * Needs to revert NEU filtering and toLowerCase().
	 */
	public String toAllFieldStringWithDatasetSubmit() {
		StringBuilder sbOutput = new StringBuilder();
		sbOutput.append(this.domain + "\t");
		if (this.ratingScore == 3) {
			sbOutput.append("NEU" + "\t");
		} else if (this.ratingScore < 3) {
			sbOutput.append("NEG" + "\t");
		} else {
			sbOutput.append("POS" + "\t");
		}
		// sbOutput.append(Label
		// .convertPlusOneMinusOneToPostiveNegative(this.label) + "\t");
		sbOutput.append(this.ratingScore + "\t");
		// sbOutput.append(this.reviewId + "\t");
		// sbOutput.append(this.productId + "\t");
		// sbOutput.append(this.title + "\t");
		sbOutput.append(this.content);
		return sbOutput.toString();
	}

	@Override
	public String toString() {
		return this.text;
	}


//	public static String preprocessText(String text) {
//	    ArrayList<String> wordList = new ArrayList<String>();
//	    String[] parts = text.split("\\s+");
//	    for (String item : parts) {
//	        String lastStr = Character.toString(item.charAt(item.length() - 1));
//	        if (isPunctuation(lastStr)) {
//	            String tmpWord = item.replace(lastStr, "");
//	            wordList.add(tmpWord);
//	            wordList.add(lastStr);
//            } else {
//	            wordList.add(item);
//            }
//        }
//        String newline = "";
//	    for (String item : wordList) {
//	        newline = newline + item + " ";
//        }
//        newline = newline.trim();
//	    return newline;
//    }
//
//
//    public static boolean isPunctuation(String item) {
//	    HashSet<String> punctuationSet = new HashSet<String>();
//        String punctuationStr = ", . ; ! ? : ...";
//        String[] parts = punctuationStr.split("\\s+");
//
//        for (String elem : parts) {
//            punctuationSet.add(elem);
//        }
//
//        if (punctuationSet.contains(item)) {
//            return true;
//        } else {
//            return false;
//        }
//    }
//
//    public static void testPreprocess() {
//	    String text = "I love. Chicago? Like! it.";
//
//        System.out.println(preprocessText(text));
//    }


    public static void main(String[] args) {
        String text = "I see it in Brookstone a few day ago and be totally sell on the idea of `` self-setting '' . well , that be actually not very true . pro : large display with adjustable level . the blue display also look a lot better than the usual green or red . the alarm be easy to use . it `` should '' be able to automatically adjust for Daylight Savings Time . con : Brookstone be intentionally vague about what they mean by `` self-setting '' . no , it do not connect to the US atomic clock as some reviewer say . it be merely pre-set in the factory , and sustain by the battery . if you look at they manual -lrb- -lrb- ... -rrb- -rrb- you will understand why : if you `` reset '' the clock , it will not reset to the current time . I do not realize this until I be try out the feature and inadvertently change the original clock . I think by `` reset '' the clock it will adjust back automatically . nope . this be the feature that I think be worth the price tag , but now I just feel cheated . the clock do use a battery -lrb- pre-installed -rrb- . so it be not really `` self-setting '' . while it store the day\\/month\\/year info , there be no way to show they on the clock screen . and , it be a lot easier and more tempting to turn of the alarm than snooze : the button to turn it off be a lot taller than the snooze button . overall , it be not a bad one even if it be not really automatic . I just wish they be more straightforward about what it really be .";
        String preprocessed = "brookstone day ago totally sell idea well not|true pro large display adjustable level blue display lot better usual green red alarm easy automatically adjust daylight savings time con brookstone intentionally vague not|connect not|atomic not|clock not|reviewer factory sustain battery manual understand reset clock not|reset not|current not|time not|realize not|feature not|inadvertently not|change not|original not|clock reset clock adjust back automatically nope feature worth price tag feel cheated clock battery store info not|show not|clock not|screen lot easier tempting turn alarm snooze button turn lot taller snooze button not|bad not|automatic straightforward";

        Document doc = new Document("", "", 3, text);

        System.out.println(doc.textPreprocessed);
        System.out.println(preprocessed.equals(doc.textPreprocessed));
        // testPreprocess();

    }

}
