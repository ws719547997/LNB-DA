package classifier;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import utility.ExceptionUtility;
import utility.FileReaderAndWriter;
import main.CmdOption;

/**
 * This class only works for binary classification.
 * 
 * For multi-classification, we need to reimplement it.
 */
public class ClassificationKnowledge {
	// Freq(+) and Freq(-). -> total number of documents in POS and NEG category, i.e., N_{+} and N_{-}
	public double[] countDocsInPerClass = null;
	// Freq(f, +) and Freq(f, -). -> wordCount(w,c)in POS and NEG documents, i.e., N_{+,w}^KB and N_{-,w}^KB
	public Map<String, double[]> wordCountInPerClass = null;
	// sum_f{Freq(f, +)} and sum_f{Freq(f, -)}. -> The number of total words in POS and NEG category.
	public double[] countTotalWordsInPerClass = null;
	// Domain-level knowledge: M_{+,w}^KB and M_{-,w}^KB -> number of domains having this word as POS and NEG.
	public Map<String, double[]> countDomainsInPerClass = null;

	public ClassificationKnowledge() {
		countDocsInPerClass = new double[2];
		wordCountInPerClass = new HashMap<String, double[]>();
		countTotalWordsInPerClass = new double[2];
		countDomainsInPerClass = new HashMap<String, double[]>();
	}

	public ClassificationKnowledge getDeepClone() {
		ClassificationKnowledge clone = new ClassificationKnowledge();
		clone.addKnowledge(this);
		return clone;
	}

	/**
	 * read classification knowledge from files
	 * @param filepath
	 * @return
	 */
	public static ClassificationKnowledge readClassificationProbabilitiesFromFile(
			String filepath) {
		ClassificationKnowledge knowledge = new ClassificationKnowledge();
		List<String> lines = FileReaderAndWriter.readFileAllLines(filepath);
		for (String line : lines) {
			String[] strSplits = line.split("\t");
			ExceptionUtility.assertAsException(strSplits.length == 7);
			// Each line represents each feature.
			String featureStr = strSplits[0]; // one feature, i.e., one word
			// Freq(+) and Freq(-). -> count number of documents in each category, i.e., N_{+} and N_{-}
			// N_{+}
			if (knowledge.countDocsInPerClass[0] <= 0) {
				knowledge.countDocsInPerClass[0] = Double
						.parseDouble(strSplits[1]);
			}
			// N_{-}
			if (knowledge.countDocsInPerClass[1] <= 0) {
				knowledge.countDocsInPerClass[1] = Double
						.parseDouble(strSplits[2]);
			}
			// Freq(f, +) and Freq(f, -). -> wordCount(w,c)in POS and NEG documents, i.e., N_{+,w}^KB and N_{-,w}^KB
			double[] featureWithClassCount = new double[] {
					Double.parseDouble(strSplits[3]),
					Double.parseDouble(strSplits[4]) };
			knowledge.wordCountInPerClass.put(featureStr,
					featureWithClassCount);
			// sum_f{Freq(f, +)} and sum_f{Freq(f, -)}. -> The number of total words in POS and NEG category.
			knowledge.countTotalWordsInPerClass[0] += featureWithClassCount[0];
			knowledge.countTotalWordsInPerClass[1] += featureWithClassCount[1];
			// Domain-level knowledge: M_{+,w}^KB and M_{-,w}^KB -> number of domains having this word as POS and NEG.
			double[] featureWithDomainClassCount = new double[] {
					Double.parseDouble(strSplits[5]),
					Double.parseDouble(strSplits[6]) };
			knowledge.countDomainsInPerClass.put(featureStr,
					featureWithDomainClassCount);
		}
		return knowledge;
	}

	/**
	 * add classification knowledge
	 * @param knowledge
	 */
	public void addKnowledge(ClassificationKnowledge knowledge) {
		for (int i = 0; i < countDocsInPerClass.length; ++i) {
			this.countDocsInPerClass[i] += knowledge.countDocsInPerClass[i];
		}
		for (Map.Entry<String, double[]> entry : knowledge.wordCountInPerClass
				.entrySet()) {
			if (!this.wordCountInPerClass.containsKey(entry
					.getKey())) {
				this.wordCountInPerClass.put(entry.getKey(),
						new double[2]);
			}
			double[] array = this.wordCountInPerClass.get(entry
					.getKey());
			for (int i = 0; i < array.length; ++i) {
				array[i] += entry.getValue()[i];
			}

			String featureStr = entry.getKey();
			if (!this.countDomainsInPerClass
					.containsKey(entry.getKey())) {
				this.countDomainsInPerClass.put(entry.getKey(),
						new double[2]);
			}
			array = this.countDomainsInPerClass.get(entry.getKey());
			double[] domainCounts = knowledge.countDomainsInPerClass
					.get(featureStr);
			if (domainCounts == null) {
				continue;
			}
			for (int i = 0; i < array.length; ++i) {
				array[i] += domainCounts[i];
			}
		}
		for (int i = 0; i < countTotalWordsInPerClass.length; ++i) {
			this.countTotalWordsInPerClass[i] += knowledge.countTotalWordsInPerClass[i];
		}
	}

	/**
	 * print classification knowledge to files
	 * classification knowledge:
	 * 1. feature -> feature word
	 * 2. Document-level knowledge: N_{+,w}^KB and N_{-,w}^KB
	 * 3. Domain-level knowledge: M_{+,w}^KB and M_{-,w}^KB
	 * 4. total number of documents in POS and NEG category: Freq(+) and Freq(-)
	 * @param filepath
	 */
	public void printToFile(String filepath) {
		StringBuilder sbOutput = new StringBuilder();
		for (Map.Entry<String, double[]> entry : wordCountInPerClass
				.entrySet()) {
			StringBuilder sbOneLine = new StringBuilder();
			String featureStr = entry.getKey();
			// feature, i.e., feature word.
			sbOneLine.append(featureStr + "\t");
			// Freq(+) and Freq(-), total number of documents in POS and NEG category
			sbOneLine.append(countDocsInPerClass[0] + "\t");
			sbOneLine.append(countDocsInPerClass[1] + "\t");
			// Document-level knowledge: N_{+,w}^KB and N_{-,w}^KB.
			double[] featureWithClassCount = entry.getValue();
			sbOneLine.append(featureWithClassCount[0] + "\t");
			sbOneLine.append(featureWithClassCount[1] + "\t");
			// Domain-level knowledge: M_{+,w}^KB and M_{-,w}^KB.
			double[] featureWithDomainClassCount = countDomainsInPerClass
					.get(featureStr);
			if (featureWithDomainClassCount == null) {
				featureWithDomainClassCount = new double[2];
			}
			sbOneLine.append(featureWithDomainClassCount[0] + "\t");
			sbOneLine.append(featureWithDomainClassCount[1]);

			sbOutput.append(sbOneLine.toString());
			sbOutput.append(System.lineSeparator());
		}
		FileReaderAndWriter.writeFile(filepath, sbOutput.toString());
	}
}
