package nlp;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

import feature.Feature;
import feature.FeatureSelection;
import main.Constant;
import utility.ExceptionUtility;
import utility.FileReaderAndWriter;
import utility.OSFilePathConvertor;

public class Documents implements Iterable<Document> {
	private List<Document> documentList = null;
	public String domain = null;

	public Map<String, Integer> mpWordFrequency = null;

	public Documents() {
		documentList = new ArrayList<Document>();
		mpWordFrequency = new HashMap<String, Integer>();
	}

	public Documents(String domain2) {
		documentList = new ArrayList<Document>();
		domain = domain2;
		mpWordFrequency = new HashMap<String, Integer>();
	}

	public Documents getDeepClone() {
		Documents clone = new Documents();
		// Clone documentList.
		for (Document document : this.documentList) {
			clone.addDocument(document.getDeepClone());
		}
		// Clone domain.
		clone.domain = this.domain;
		// Clone mpWordFrequency.
		clone.mpWordFrequency = new HashMap<String, Integer>();
		for (Map.Entry<String, Integer> entry : this.mpWordFrequency.entrySet()) {
			clone.mpWordFrequency.put(entry.getKey(), entry.getValue());
		}
		return clone;
	}

    /**
     * read documents from one file. Actually, each file refers to a domain.
     * Concretely, it reads quadruple items, i.e., {domain name, label, rating score, and review content}.
     * @param domain
     * @param filepath
     * @return documents
     */
    public static Documents readDocuments(String domain, String filepath) {
        Documents documents = new Documents();
        documents.domain = domain;
        try {
            List<String> lines = FileReaderAndWriter.readFileAllLines(filepath);
            for (String line : lines) {
                String[] nextLine = line.split("\t");
                String domainLine = nextLine[0].toLowerCase(); // index
                // skip the first line? why do this due to the contents in the file
                if (domainLine.equals("index")) {
                    // The header (first line).
                    continue;
                }
				String reviewId = nextLine[0];
				// String domain = nextLine[1];
                String label = nextLine[2]; // label
                if (!label.equals("POS") && !label.equals("NEG")) {
                    continue;
                }
                int ratingScore = Double.valueOf(nextLine[3]).intValue(); // rating score

                String reviewContent = "EMPTYSTRING"; // Use a special string to
                // represent empty string.
                if (nextLine.length >= 5) {
                    reviewContent = nextLine[4].toLowerCase(); // review content
                }

                Document document = new Document(reviewId, domain, label, ratingScore, reviewContent);
                // Document document = new Document(domain, label, ratingScore, reviewContent); // including data pre-processing
                document.docIndex = documents.size(); // document index
                documents.addDocument(document);
            } // have completely read the documents
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        // Remove the infrequent words in String[] words in each document.
        documents.mpWordFrequency = documents.getMpWordFrequency();
        for (Document document : documents) {
            for (int i = 0; i < document.words.length; ++i) {
                if (document.words[i].equals("")) {
                    continue;
                }
                String word = document.words[i];
                int frequency = documents.mpWordFrequency.get(word);
                if (frequency <= Constant.INFREQUENT_WORD_REMOVAL_THRESHOLD) {
                    // Note that this threshold is 0 for classification and 5 for topic model.
                    document.words[i] = "";
                }
            }
        }
        // Get the word frequency mapping.
        documents.mpWordFrequency = documents.getMpWordFrequency();
        return documents;
    }

	/**
	 王松：添加read chinese doc
	 */

	public static Documents readChineseDocuments(String domain, String filepath) {
		Documents documents = new Documents();
		documents.domain = domain; //ws: single domain
		try {
			List<String> lines = FileReaderAndWriter.readFileAllLines(filepath);
			for (String line : lines) {  // ws: line by line
				String[] nextLine = line.split("\t");  //ws split a line into string bu Tab
				String domainLine = nextLine[0].toLowerCase(); // index
				// skip the first line? why do this due to the contents in the file
				if (domainLine.equals("index")) {
					// The header (first line).
					continue;
				}
				String reviewId = nextLine[0];
				// String domain = nextLine[1];
				String label = nextLine[2]; // label
				if (!label.equals("POS") && !label.equals("NEG")) {
					continue;
				}
				int ratingScore = Double.valueOf(nextLine[3]).intValue(); // rating score

				String reviewContent = "EMPTYSTRING"; // Use a special string to
				// represent empty string.
				if (nextLine.length >= 2) { //原文是5 我改成4了 可能是因为散户说了几个四字成语
					reviewContent = nextLine[4]; // review content
				}
				//王松： 我加了一个参数 实现了重载
				Document document = new Document(reviewId, domain, label, ratingScore, reviewContent,"chinese");
				// Document document = new Document(domain, label, ratingScore, reviewContent); // including data pre-processing
				document.docIndex = documents.size(); // document index
				documents.addDocument(document);
			} // have completely read the documents
		} catch (Exception ex) {
			ex.printStackTrace();
		}

		// Remove the infrequent words in String[] words in each document.
		documents.mpWordFrequency = documents.getMpWordFrequency();
		for (Document document : documents) {
			for (int i = 0; i < document.words.length; ++i) {
				if (document.words[i].equals("")) {
					continue;
				}
				String word = document.words[i];
				int frequency = documents.mpWordFrequency.get(word);
				if (frequency <= Constant.INFREQUENT_WORD_REMOVAL_THRESHOLD) {
					// Note that this threshold is 0 for classification and 5 for topic model.
					document.words[i] = "";
				}
			}
		}
		// Get the word frequency mapping.
		documents.mpWordFrequency = documents.getMpWordFrequency();
		return documents;
	}


	public static Map<String, Integer> checkUnseenWords(FeatureSelection featureSelection, Documents documentsTest) {
		Map<String, Integer> mpWordFreqTest = documentsTest.getMpWordFrequency();
		Map<String, Integer> unSeen = new HashMap<>();

		for (Document docTest : documentsTest) {
			for (Feature feature : docTest.featuresForNaiveBayes) {
				String featureStr = feature.featureStr;
				if (featureSelection != null
						&& !featureSelection.isFeatureSelected(featureStr)) {
					if (featureStr == "") {
						continue;
					}
					unSeen.put(featureStr, mpWordFreqTest.get(featureStr));
				}
			}
		}
		return unSeen;
	}

    public static Map<String, Integer> checkUnseenWords(Documents documentsTrain, Documents documentsTest) {
	   Map<String, Integer> mpWordFreqTrain = documentsTrain.getMpWordFrequency();
	   Map<String, Integer> mpWordFreqTest = documentsTest.getMpWordFrequency();

	   Map<String, Integer> unSeen = new HashMap<>();

	   for (String testWord : mpWordFreqTest.keySet()) {
	       if (!mpWordFreqTrain.containsKey(testWord)) {
	           unSeen.put(testWord, mpWordFreqTest.get(testWord));
           }
       }
       return unSeen;
    }

    public static Map<String, Integer> checkFoundWord(List<Documents> otherDomains, Map<String, Integer> unseenWord) {
	    Documents wholeDocuments = new Documents();
	    for (Documents docs: otherDomains) {
	        wholeDocuments.addDocuments(docs);
        }

		Map<String, Integer> mpWordFreqOtherDomain = wholeDocuments.getMpWordFrequency();

		Map<String, Integer> foundWord = new HashMap<>();

		for (String word: unseenWord.keySet()) {
			if (mpWordFreqOtherDomain.containsKey(word)) {
				foundWord.put(word, mpWordFreqOtherDomain.get(word));
			}
		}
		return foundWord;
	}

	public static List<Documents> readListOfDocumentsFromChineseStockComments(
			String domainDirectory) {
		List<Documents> documentsList = new ArrayList<Documents>();
		File[] domainFiles = new File(OSFilePathConvertor.convertOSFilePath(domainDirectory)).listFiles();
		for (File domainFile : domainFiles) {
			// Get the domain name, i.e., file name without extension.
			if (domainFile.getName().contains(".DS_Store")) {
				continue;
			}
			String domain = domainFile.getName().replaceFirst("[.][^.]+$", "");
			Documents documents = Documents.readChineseDocuments(domain,
					domainFile.getAbsolutePath());
			documentsList.add(documents);
		}
		return documentsList;
	}

	public static List<Documents> readListOfDocumentsFromDifferentDomains(
			String domainDirectory) {
		List<Documents> documentsList = new ArrayList<Documents>();
		File[] domainFiles = new File(OSFilePathConvertor.convertOSFilePath(domainDirectory)).listFiles();
		for (File domainFile : domainFiles) {
			// Get the domain name, i.e., file name without extension.
			if (domainFile.getName().contains(".DS_Store")) {
			    continue;
            }
			String domain = domainFile.getName().replaceFirst("[.][^.]+$", "");
			Documents documents = Documents.readDocuments(domain,
					domainFile.getAbsolutePath());
			documentsList.add(documents);
		}
		return documentsList;
	}


    /**
     * only the domain names in domainList will participate in the computation
     * @param domainDirectory
     * @param domainList
     * @return
     */
    public static List<Documents> readListOfDocumentsFromDifferentDomains(
            String domainDirectory, List<String> domainList) {
        List<Documents> documentsList = new ArrayList<Documents>();
        File[] domainFiles = new File(OSFilePathConvertor.convertOSFilePath(domainDirectory))
                .listFiles(); // get all file paths in this domainDirectory
        for (File domainFile : domainFiles) {
            // Get the name of this file, i.e., file name without extension.
            if (domainFile.getName().contains(".DS_Store")) {
                continue;
				// if contains ".DS_Store", continue (It is due to Apple Mac.)
            }

            String domain = domainFile.getName().replaceFirst("[.][^.]+$", ""); // domain name
            if (! domainList.contains(domain)) {
                continue;
            }

            // start to read documents from this file (i.e., domainFile)
            Documents documents = Documents.readDocuments(domain,
                    domainFile.getAbsolutePath());
            documentsList.add(documents);
        }
        return documentsList;
    }

	public static Documents getMergedDocuments(Documents documents1,
			Documents documents2) {
		Documents mergedDocuments = new Documents();
		mergedDocuments.addDocuments(documents1);
		mergedDocuments.addDocuments(documents2);
		mergedDocuments.domain = documents1.domain;
		return mergedDocuments;
	}

	public void addDocument(Document document) {
		documentList.add(document);
	}

	public void addDocuments(Documents documents) {
		for (Document document : documents) {
			this.addDocument(document);
		}
	}

	public void assignLabel(String label) {
		for (Document document : this.documentList) {
			document.label = label;
		}
	}

	public Documents selectSubsetOfDocumentsByOrder(double percentage) {
		return selectSubsetOfDocumentsByOrder((int) (this.size() * percentage));
	}

	public Documents selectSubsetOfDocumentsByOrder(int count) {
		Documents documents = new Documents(this.domain);
		for (int i = 0; i < count && i < this.size(); ++i) {
			documents.addDocument(this.getDocument(i));
		}
		return documents;
	}

	public Documents selectSubsetOfDocumentsRandomly(int count) {
		Documents documents = new Documents(this.domain);
		ArrayList<Document> dList = new ArrayList<Document>();
		for (Document document : this.documentList) {
			dList.add(document);
		}
		Collections.shuffle(dList, new Random(Constant.RANDOMSEED));
		for (int i = 0; i < count && i < this.size(); ++i) {
			documents.addDocument(dList.get(i));
		}
		return documents;
	}

	public Documents selectSubsetOfDocumentsRandomly(double percentage) {
		return selectSubsetOfDocumentsRandomly((int) (this.size() * percentage));
	}

    // count the frequency (i.e., wordCount) of each word in one domain
	public Map<String, Integer> getMpWordFrequency() {
		Map<String, Integer> mpWordFrequency = new HashMap<String, Integer>();
		for (Document document : this.documentList) {
			for (String word : document.words) {
				if (!mpWordFrequency.containsKey(word)) {
					mpWordFrequency.put(word, 0);
				}
				mpWordFrequency.put(word, mpWordFrequency.get(word) + 1);
			}
		}
		return mpWordFrequency;
	}

	public int size() {
        return documentList.size();
	}

	public Document getDocument(int i) {
		ExceptionUtility.assertAsException(0 <= i && i < documentList.size());
		return documentList.get(i);
	}

	public String[] getLabels() {
		String[] labels = new String[this.size()];
		for (int i = 0; i < this.size(); ++i) {
			labels[i] = documentList.get(i).label;
		}
		return labels;
	}

	public int[] getLabelsAsIntegers() {
		int[] labels = new int[this.size()];
		for (int i = 0; i < this.size(); ++i) {
			labels[i] = Label
					.convertLabelStrToLabelInteger(documentList.get(i).label);
		}
		return labels;
	}

	public String[] getPredicts() {
		String[] predicts = new String[this.size()];
		for (int i = 0; i < this.size(); ++i) {
			predicts[i] = documentList.get(i).predict;
		}
		return predicts;
	}

	public Documents getPositiveDocuments() {
		Documents positiveDocs = new Documents();
		for (Document document : documentList) {
			if (document.isPositive()) {
				positiveDocs.addDocument(document);
			}
		}
		return positiveDocs;
	}

	public int getNoOfPositiveLabels() {
		int count = 0;
		for (Document document : documentList) {
			if (document.isPositive()) {
				++count;
			}
		}
		return count;
	}

	public Documents getNegativeDocuments() {
		Documents positiveDocs = new Documents();
		for (Document document : documentList) {
			if (document.isNegative()) {
				positiveDocs.addDocument(document);
			}
		}
		return positiveDocs;
	}

	public int getNoOfNegativeLabels() {
		int count = 0;
		for (Document document : documentList) {
			if (document.isNegative()) {
				++count;
			}
		}
		return count;
	}

	@Override
	public Iterator<Document> iterator() {
		return documentList.iterator();
	}

	public void printToFile(String filepath) {
		StringBuilder sbOutput = new StringBuilder();
		// Insert headers.
		sbOutput.append("index\tDomain\tLabel\tRating\tReview");
		sbOutput.append(System.lineSeparator());
		for (Document document : this.documentList) {
			sbOutput.append(document.toAllFieldString());
			sbOutput.append(System.lineSeparator());
		}
		FileReaderAndWriter.writeFile(filepath, sbOutput.toString());
	}

	public void printToFileWithPreprocessedContent(String filepath) {
		StringBuilder sbOutput = new StringBuilder();
		// Insert headers.
		sbOutput.append("Domain\tRatingScore\tReviewID\tProduceID\tTitle\tTextPreprocessed");
		sbOutput.append(System.lineSeparator());
		for (Document document : this.documentList) {
			sbOutput.append(document.toAllFieldStringWithPreprocessedContent());
			sbOutput.append(System.lineSeparator());
		}
		FileReaderAndWriter.writeFile(filepath, sbOutput.toString());
	}

	public void printToFileWithDatasetSubmit(String filepath) {
		StringBuilder sbOutput = new StringBuilder();
		// Insert headers.
		sbOutput.append("Domain\tLabel\tRating\tReview\r\n");
		for (Document document : this.documentList) {
			sbOutput.append(document.toAllFieldStringWithDatasetSubmit());
			sbOutput.append(System.lineSeparator());
		}
		FileReaderAndWriter.writeFile(filepath, sbOutput.toString());
	}

	/**
	 * In order to balance the data
	 * @param nToPRatio
	 */
	public void makeBinaryClassesEven(double nToPRatio) {
		int noOfP = (int) (this.getNoOfPositiveLabels() * nToPRatio);
		int noOfN = this.getNoOfNegativeLabels();
		// Only consider the case when the negative is more than positive.
		if (noOfP < noOfN) {
			Collections.shuffle(this.documentList, new Random(
					Constant.RANDOMSEED));
			int difference = noOfN - noOfP;
			for (int i = this.documentList.size() - 1; i >= 0 && difference > 0; --i) {
				if (this.documentList.get(i).isNegative()) {
					this.documentList.remove(i);
					--difference;
				}
			}
		}

		// if (noOfP > noOfN) {
		// Collections.shuffle(this.documentList, new Random(
		// Constant.RANDOMSEED));
		// int difference = noOfP - noOfN;
		// for (int i = this.documentList.size() - 1; i >= 0 && difference > 0;
		// --i) {
		// if (this.documentList.get(i).isPositive()) {
		// this.documentList.remove(i);
		// --difference;
		// }
		// }
		// } else
	}

}
