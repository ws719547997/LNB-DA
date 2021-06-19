package nlp;

import java.io.File;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import main.Constant;
import utility.ExceptionUtility;
import utility.FileReaderAndWriter;
import utility.OSFilePathConvertor;

/**
 * Note that domain name is case-insensitive.
 */
public class StopWordHelper {
	// Inputs.
	private final String stopWordDomainIndepFilePath = ".\\resources\\nlp\\stopword\\StopWord_DomainIndep.txt";
	private final String stopWordDomainIndepOriginalFilePath = ".\\resources\\nlp\\stopword\\StopWord_DomainIndep_Original.txt";
	private final String stopWordDomainDepDirectory = ".\\resources\\nlp\\stopword\\DomainDependent\\";
	private final String cnStopWordFilePath = ".\\resources\\nlp\\stopword\\cn_stopword_simple.txt";

	private static StopWordHelper _instance = null;
	// Domain Independent stopwords.
	private Set<String> hsStopWordDomainIndep = null;
	// Domain Independent Original stopwords.
	private Set<String> hsStopWordDomainIndepOriginal = null;
	// Domain dependent stopwords.
	private Map<String, Set<String>> mpDomainToStopwords = null;
	private Set<String> cnStopWord = null;


	protected StopWordHelper() {
		// Read domain independent stop words.
		hsStopWordDomainIndep = readStopWords(stopWordDomainIndepFilePath);
		cnStopWord = readStopWords(cnStopWordFilePath);
		// Read domain independent original stop words.
		hsStopWordDomainIndepOriginal = readStopWords(stopWordDomainIndepOriginalFilePath);
		// Read domain dependent stop words.
		mpDomainToStopwords = new HashMap<String, Set<String>>();
		File[] domainFiles = new File(OSFilePathConvertor.convertOSFilePath(stopWordDomainDepDirectory)).listFiles();
		for (File domainFile : domainFiles) {
			// Get the domain name, i.e., file name without extension.
			String domain = domainFile.getName().replaceFirst("[.][^.]+$", "")
					.toLowerCase();
			mpDomainToStopwords.put(domain,
					readStopWords(domainFile.getAbsolutePath()));
		}
	}

	private Set<String> readStopWords(String filepath) {
		Set<String> set = new HashSet<String>();
		List<String> contentLines = FileReaderAndWriter
				.readFileAllLines(filepath);
		for (String line : contentLines) {
			String stopword = line.trim().toLowerCase();
			set.add(stopword);
		}
		return set;
	}

	public static StopWordHelper getInstance() {
		if (_instance == null) {
			_instance = new StopWordHelper();
		}
		return _instance;
	}

	// Domain Independent stopwords.
	public boolean isDomainIndepStopWord(String word) {
		word = word.toLowerCase();
		return hsStopWordDomainIndep.contains(word);
	}

	// Domain Independent Original stopwords.
	public boolean isDomainIndepOriginalStopWord(String word) {
		word = word.toLowerCase();
		return hsStopWordDomainIndepOriginal.contains(word);
	}

	// Domain dependent stopwords.
	public boolean isDomainDepStopWord(String word, String domain) {
		if (!Constant.USE_DOMAIN_DEPENDENT_STOPWORDS) {
			return false;
		}
		word = word.toLowerCase();
		// Note that the domain names in the stop word directory are without " "
		// and "_".
		domain = domain.toLowerCase();
		domain = domain.replaceAll(" ", "");
		// For same domain different products setting.
		// Get the root domain.
		// For general setting, rootDomain == domain.
		String rootDomain = domain;
		if (domain.matches(".*\\d.*")) {
			// Contain digit.
			int index = domain.lastIndexOf('_');
			if (index >= 0) {
				rootDomain = domain.substring(0, index);
			}
		}
		rootDomain = rootDomain.replaceAll("_", "");

		ExceptionUtility.assertAsException(
				mpDomainToStopwords.containsKey(rootDomain), "Domain "
						+ rootDomain
						+ " cannot be found in the stop word direcotry");
		if (mpDomainToStopwords.get(rootDomain).contains(word)) {
			return true;
		}
		return false;
	}

	public boolean iscnStopword(String word) {
		return cnStopWord.contains(word);
	}
}
