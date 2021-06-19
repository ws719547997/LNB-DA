package nlp;

import org.apache.commons.lang3.StringUtils;

/**
 * A word is not valid if it meets any of the following criteria:
 * 
 * 1. stopword (domain independent or domain dependent).
 * 
 * 2. contains digits.
 * 
 * 3. contains punctuations.
 * 
 */
public class WordValidityHelper {
	private static WordValidityHelper _instance = null;

	public static WordValidityHelper getInstance() {
		if (_instance == null) {
			_instance = new WordValidityHelper();
		}
		return _instance;
	}

	public boolean isValid(String word, String domain) {
		// Stop word.
		if (StopWordHelper.getInstance().isDomainIndepStopWord(word)
				|| StopWordHelper.getInstance().isDomainDepStopWord(word,
						domain)) {
			return false;
		}

		// contains digits.
		if (StringUtils.isAlpha(word)) {
			// Contains letter only.
			return true;
		} else {
			return false;
		}
	}

	/**
	 *
	 * @param word 一个string
	 * @param domain 暂时还没用 如果需要可以实现领域专属stopword
	 * @return 真和假
	 * todo 这里边只去掉了停用词 还没有实现数字的消除，或许不用消除？
	 */
	public boolean isChineseValid(String word, String domain) {
		// Stop word.
		if (StopWordHelper.getInstance().iscnStopword(word)) {
			return false;
		}
		// 去除数字在这里
//		if (StringUtils.isAlpha(word)) {
//			// Contains letter only.
//			return true;
//		} else {
//			return false;
//		}
		return true;
	}

	public boolean isStopword(String word, String domain) {
		// Stop word.
		if (StopWordHelper.getInstance().isDomainIndepStopWord(word)
				|| StopWordHelper.getInstance().isDomainDepStopWord(word,
						domain)) {
			return false;
		}
		return true;
	}
}
