package nlp;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Map;
import java.util.TreeMap;

import org.apache.commons.lang3.StringUtils;

import utility.FileReaderAndWriter;

/**
 * A corpus contains two components:
 * 
 * 1. Documents where each document contains a list of word ids.
 * 
 * 2. Vocabulary: the mapping from word id to word.
 */
public class Corpus {
	public String domain = null; // Domain name.
	public Vocabulary vocab = null;
	public int[][] docs = null;
	public String[][] docsStr = null;
	// Added by Zhiyuan (Brett) Chen on 01/10/2015.
	public Documents documents = null;

	// Build the inverted index that is used to compute document
	// frequency and co-document frequency.
	private Map<String, HashSet<Integer>> wordstrToSetOfDocsMap = null;

	public Corpus(String domain2) {
		domain = domain2;
		wordstrToSetOfDocsMap = new TreeMap<String, HashSet<Integer>>();
	}

	public Corpus(Documents documents2) {
		documents = documents2;

		domain = documents.domain;
		vocab = new Vocabulary();
		int D = documents.size();
		docs = new int[D][];
		docsStr = new String[D][];
		wordstrToSetOfDocsMap = new TreeMap<String, HashSet<Integer>>();

		for (int d = 0; d < documents.size(); ++d) {
			Document document = documents.getDocument(d);
			String[] words = document.words;
			int docLength = words.length;
			// Count the number of non-empty strings as we only consider the
			// non-empty string in topic model.
			int noOfNonEmptyString = 0;
			for (String word : words) {
				if (word.length() > 0) {
					++noOfNonEmptyString;
				}
			}
			docs[d] = new int[noOfNonEmptyString];
			docsStr[d] = new String[noOfNonEmptyString];
			for (int i = 0, j = 0; i < docLength && j < noOfNonEmptyString; ++i) {
				String wordstr = words[i].trim();
				if (words[i].length() == 0) {
					// Invalid words.
					continue;
				}
				if (!vocab.containsWordstr(wordstr)) {
					vocab.addWordstrWithoutWordid(wordstr);
				}
				int wordid = vocab.getWordidByWordstr(wordstr);
				docs[d][j] = wordid;
				docsStr[d][j] = wordstr;
				++j;

				if (!wordstrToSetOfDocsMap.containsKey(wordstr)) {
					wordstrToSetOfDocsMap.put(wordstr, new HashSet<Integer>());
				}
				HashSet<Integer> setOfDocs = wordstrToSetOfDocsMap.get(wordstr);
				setOfDocs.add(d);
			}
		}
	}

	/**
	 * Read the corpus from the files (both docs and vocab).
	 */
	public static Corpus getCorpusFromFile(String domain, String docsFilepath,
			String vocabFilepath, String oriContextFilePath) {
		Corpus corpus = new Corpus(domain);

		// Read the vocab file.
		corpus.vocab = Vocabulary.getVocabularyFromFile(vocabFilepath);

		// Read the docs file.
		ArrayList<String> docsLines = FileReaderAndWriter
				.readFileAllLines(docsFilepath);
		// // Ignore the empty line at the end.
		// ArrayList<String> docsLines_nonEmpty = new ArrayList<String>();
		// for (String line : docsLines) {
		// if (line.trim().length() > 0) {
		// docsLines_nonEmpty.add(line);
		// }
		// }

		// Modified by Zhiyuan (Brett) Chen on 01/11/2015.
		// If a document (line) is empty, keep it in the corpus.
		ArrayList<String> docsLines_nonEmpty = docsLines;
		int size = docsLines_nonEmpty.size();
		corpus.docs = new int[size][];
		corpus.docsStr = new String[size][];
		for (int d = 0; d < size; ++d) {
			String docsLine = docsLines_nonEmpty.get(d);
			String[] splits = StringUtils.split(docsLine.trim());
			int length = splits.length;
			corpus.docs[d] = new int[length];
			corpus.docsStr[d] = new String[length];
			for (int n = 0; n < length; ++n) {
				int wordid = Integer.parseInt(splits[n]);
				corpus.docs[d][n] = wordid;
				corpus.docsStr[d][n] = corpus.vocab.getWordstrByWordid(wordid);
				// Update the inverted index.
				String wordstr = corpus.vocab.getWordstrByWordid(wordid);
				if (!corpus.wordstrToSetOfDocsMap.containsKey(wordstr)) {
					corpus.wordstrToSetOfDocsMap.put(wordstr,
							new HashSet<Integer>());
				}
				HashSet<Integer> setOfDocs = corpus.wordstrToSetOfDocsMap
						.get(wordstr);
				setOfDocs.add(d);
			}
		}

		return corpus;
	}

	/**
	 * Get the number of documents in the corpus.
	 */
	public int getNoofDocuments() {
		return docs == null ? 0 : docs.length;
	}

	/**
	 * Get the number of documents that contain this word.
	 */
	public int getDocumentFrequency(String wordstr) {
		if (!wordstrToSetOfDocsMap.containsKey(wordstr)) {
			return 0;
		}
		return wordstrToSetOfDocsMap.get(wordstr).size();
	}

	/**
	 * Get the co-document frequency which is the number of documents that both
	 * words appear.
	 */
	public int getCoDocumentFrequency(String wordstr1, String wordstr2) {
		if (!wordstrToSetOfDocsMap.containsKey(wordstr1)
				|| !wordstrToSetOfDocsMap.containsKey(wordstr2)) {
			return 0;
		}
		HashSet<Integer> setOfDocs1 = wordstrToSetOfDocsMap.get(wordstr1);
		HashSet<Integer> setOfDocs2 = wordstrToSetOfDocsMap.get(wordstr2);
		HashSet<Integer> intersection = new HashSet<Integer>(setOfDocs1);
		intersection.retainAll(setOfDocs2);
		return intersection.size();
	}
}
