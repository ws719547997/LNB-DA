package knowledge;

import java.io.File;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.lang3.StringUtils;

import topicmodel.ModelPrinter;
import utility.FileReaderAndWriter;

public class SentimentSeeds {
	public Set<String> positiveSeeds = null;
	public Set<String> negativeSeeds = null;

	public static SentimentSeeds readSeedsFromDirectory(String domain,
			String seedSentimentDirectory) {
		SentimentSeeds seeds = new SentimentSeeds();

		seeds.positiveSeeds = new HashSet<String>();
		String positiveSeedFilePath = seedSentimentDirectory + File.separator
				+ domain + ModelPrinter.positiveSeedSuffix;
		List<String> positiveLlines = FileReaderAndWriter
				.readFileAllLines(positiveSeedFilePath);
		for (String line : positiveLlines) {
			String word = StringUtils.split(line.toLowerCase().trim())[0];
			seeds.positiveSeeds.add(word);
		}

		seeds.negativeSeeds = new HashSet<String>();
		String negativeSeedFilePath = seedSentimentDirectory + File.separator
				+ domain + ModelPrinter.negativeSeedSuffix;
		List<String> negativeLlines = FileReaderAndWriter
				.readFileAllLines(negativeSeedFilePath);
		for (String line : negativeLlines) {
			String word = StringUtils.split(line.toLowerCase().trim())[0];
			seeds.negativeSeeds.add(word);
		}
		return seeds;
	}

	public void printToFiles(String positiveSeedFilepath,
			String negativeSeedFilepath) {
		StringBuilder sbPositive = new StringBuilder();
		for (String seed : positiveSeeds) {
			sbPositive.append(seed);
			sbPositive.append(System.lineSeparator());
		}
		FileReaderAndWriter.writeFile(positiveSeedFilepath,
				sbPositive.toString());

		StringBuilder sbNegative = new StringBuilder();
		for (String seed : negativeSeeds) {
			sbNegative.append(seed);
			sbNegative.append(System.lineSeparator());
		}
		FileReaderAndWriter.writeFile(negativeSeedFilepath,
				sbNegative.toString());
	}

}
