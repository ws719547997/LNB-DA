package nlp;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

// Negation word.
public class NegationWordHelper {
	public static Set<String> negationWords = new HashSet<String>(
			Arrays.asList("not", "never", "no", "isn't", "don't", "didn't",
					"doesn't", "n't"));

	public static boolean isNegationWord(String word) {
		return negationWords.contains(word); // return True or False
	}
}
