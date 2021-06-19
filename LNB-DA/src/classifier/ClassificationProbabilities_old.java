package classifier;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import feature.InformationGain;
import utility.ExceptionUtility;
import utility.FileReaderAndWriter;

public class ClassificationProbabilities_old {
	public int positiveCount = -1; // Freq(+).
	public int negativeCount = -1; // Freq(-).
	public Map<String, Double> mpFeatureStrToProbOfFeatureGivenPositive = null; // P(f|+).
	public Map<String, Double> mpFeatureStrToProbOfFeatureGivenNegative = null; // P(f|-).
	public Map<String, Double> mpFeatureStrToProbOfPositiveGivenFeature = null; // P(+|f).
	public Map<String, Double> mpFeatureStrToProbOfNegativeGivenFeature = null; // P(-|f).
	public Map<String, Integer> mpFeatureStrToFeatureCount = null; // Freq(f).

	public ClassificationProbabilities_old() {
		mpFeatureStrToProbOfFeatureGivenPositive = new HashMap<String, Double>();
		mpFeatureStrToProbOfFeatureGivenNegative = new HashMap<String, Double>();
		mpFeatureStrToProbOfPositiveGivenFeature = new HashMap<String, Double>();
		mpFeatureStrToProbOfNegativeGivenFeature = new HashMap<String, Double>();
		mpFeatureStrToFeatureCount = new HashMap<String, Integer>();
	}

	public static ClassificationProbabilities_old readClassificationProbabilitiesFromFile(
			String filepath) {
		ClassificationProbabilities_old cp = new ClassificationProbabilities_old();
		List<String> lines = FileReaderAndWriter.readFileAllLines(filepath);
		for (String line : lines) {
			String[] strSplits = line.split("\t");
			ExceptionUtility.assertAsException(strSplits.length == 8);
			String featureStr = strSplits[0];
			if (cp.positiveCount < 0) {
				cp.positiveCount = Integer.parseInt(strSplits[1]);
			}
			if (cp.negativeCount < 0) {
				cp.negativeCount = Integer.parseInt(strSplits[2]);
			}
			// P(f|+).
			cp.mpFeatureStrToProbOfFeatureGivenPositive.put(featureStr,
					Double.parseDouble(strSplits[3]));
			// P(f|-).
			cp.mpFeatureStrToProbOfFeatureGivenNegative.put(featureStr,
					Double.parseDouble(strSplits[4]));
			// P(+|f).
			cp.mpFeatureStrToProbOfPositiveGivenFeature.put(featureStr,
					Double.parseDouble(strSplits[5]));
			// P(-|f).
			cp.mpFeatureStrToProbOfNegativeGivenFeature.put(featureStr,
					Double.parseDouble(strSplits[6]));
			// Freq(f).
			cp.mpFeatureStrToFeatureCount.put(featureStr,
					Integer.parseInt(strSplits[7]));
		}
		return cp;
	}

	public static ClassificationProbabilities_old readClassificationProbabilitiesFromClassifier(
			BaseClassifier classifier) {
		ClassificationProbabilities_old cp = new ClassificationProbabilities_old();
		InformationGain ig = classifier.featureSelection.informationGain;

		for (Map.Entry<String, Double> entry : ig.mpFeatureStrToFeatureGivenPositive
				.entrySet()) {
			String featureStr = entry.getKey();
			// P(f|+).
			double probOfFeatureGivenPositive = entry.getValue();
			cp.mpFeatureStrToProbOfFeatureGivenPositive.put(featureStr,
					probOfFeatureGivenPositive);
			// P(f|-).
			double probOfFeatureGivenNegative = ig.mpFeatureStrToFeatureGivenNegative
					.get(featureStr);
			cp.mpFeatureStrToProbOfFeatureGivenNegative.put(featureStr,
					probOfFeatureGivenNegative);
			// P(+|f).
			double probOfPositiveGivenFeature = ig.mpFeatureStrToPositiveGivenFeature
					.get(featureStr);
			cp.mpFeatureStrToProbOfPositiveGivenFeature.put(featureStr,
					probOfPositiveGivenFeature);
			// P(-|f).
			double probOfNegativeGivenFeature = 1.0 - probOfPositiveGivenFeature;
			cp.mpFeatureStrToProbOfNegativeGivenFeature.put(featureStr,
					probOfNegativeGivenFeature);
			// freq(+, f).
			int positiveWithFeatureCount = ig.mpFeatureStrToPositiveWithFeatureCount
					.get(featureStr);
			// freq(-, f).
			int negativeWithFeatureCount = ig.mpFeatureStrToNegativeWithFeatureCount
					.get(featureStr);
			// freq(+, ~f).
			int positiveWithoutFeatureCount = ig.mpFeatureStrToPositiveWithoutFeatureCount
					.get(featureStr);
			// freq(-, ~f).
			int negativeWithoutFeatureCount = ig.mpFeatureStrToNegativeWithoutFeatureCount
					.get(featureStr);
			// freq(f) = freq(+, f) + freq(-, f).
			int featureCount = positiveWithFeatureCount
					+ negativeWithFeatureCount;
			cp.mpFeatureStrToFeatureCount.put(featureStr, featureCount);
			// freq(+) = freq(+, f) + freq(+, ~f).
			int positiveCount = positiveWithFeatureCount
					+ positiveWithoutFeatureCount;
			cp.positiveCount = positiveCount;
			// freq(-) = freq(-, f) + freq(-, ~f).
			int negativeCount = negativeWithFeatureCount
					+ negativeWithoutFeatureCount;
			cp.negativeCount = negativeCount;
		}
		return cp;
	}

	public void printToFile(String filepath) {
		StringBuilder sbOutput = new StringBuilder();
		for (Map.Entry<String, Double> entry : mpFeatureStrToProbOfFeatureGivenPositive
				.entrySet()) {
			StringBuilder sbOneLine = new StringBuilder();
			String featureStr = entry.getKey();
			sbOneLine.append(featureStr + "\t");
			sbOneLine.append(positiveCount + "\t");
			sbOneLine.append(negativeCount + "\t");
			// P(f|+).
			double probOfFeatureGivenPositive = mpFeatureStrToProbOfFeatureGivenPositive
					.get(featureStr);
			sbOneLine.append(probOfFeatureGivenPositive + "\t");
			// P(f|-).
			double probOfFeatureGivenNegative = mpFeatureStrToProbOfFeatureGivenNegative
					.get(featureStr);
			sbOneLine.append(probOfFeatureGivenNegative + "\t");
			// P(+|f).
			double probOfPositiveGivenFeature = mpFeatureStrToProbOfPositiveGivenFeature
					.get(featureStr);
			sbOneLine.append(probOfPositiveGivenFeature + "\t");
			// P(-|f).
			double probOfNegativeGivenFeature = 1.0 - probOfPositiveGivenFeature;
			sbOneLine.append(probOfNegativeGivenFeature + "\t");
			// Freq(f).
			int featureCount = mpFeatureStrToFeatureCount.get(featureStr);
			sbOneLine.append(featureCount + "\t");
			sbOutput.append(sbOneLine.toString());
			sbOutput.append(System.lineSeparator());
		}
		FileReaderAndWriter.writeFile(filepath, sbOutput.toString());
	}

}
