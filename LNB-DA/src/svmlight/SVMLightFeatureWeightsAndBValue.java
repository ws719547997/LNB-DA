package svmlight;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import nlp.Document;
import nlp.Documents;
import feature.Feature;
import feature.FeatureSelection;
import utility.ExceptionUtility;
import utility.FileReaderAndWriter;
import utility.ItemWithValue;

public class SVMLightFeatureWeightsAndBValue {
	private Map<Integer, Double> mpFeatureIdToWeight = null;
	private double bValue = 0.0;

	public SVMLightFeatureWeightsAndBValue() {
		mpFeatureIdToWeight = new TreeMap<Integer, Double>();
	}

	public static SVMLightFeatureWeightsAndBValue readSVMLightFeatureWeightsAndBValue(
			String filepath) {
		SVMLightFeatureWeightsAndBValue fw = new SVMLightFeatureWeightsAndBValue();

		int bValueLine = 10;
		ArrayList<String> inputLines = FileReaderAndWriter
				.readFileAllLines(filepath);
		String lineB = inputLines.get(bValueLine);
		String[] strSplitsB = lineB.split(" ");
		fw.bValue = Double.parseDouble(strSplitsB[0]);

		Pattern pattern = Pattern.compile("(\\d+):(-?\\d+\\.?\\d*)");

		for (int i = bValueLine + 1; i < inputLines.size(); ++i) {
			String line = inputLines.get(i);
			String[] strSplits = line.split(" ");
			Double w = Double.parseDouble(strSplits[0]);
			for (int j = 1; j < strSplits.length - 1; ++j) {
				Matcher matcher = pattern.matcher(strSplits[j]);
				// Attention: matcher.find() must be called before finding
				// groups.
				if (matcher.find()) {
					int featureId = Integer.parseInt(matcher.group(1));
					double value = Double.parseDouble(matcher.group(2));
					// Console.WriteLine(m.Groups[1].ToString());
					if (fw.mpFeatureIdToWeight.containsKey(featureId)) {
						fw.mpFeatureIdToWeight.put(featureId,
								fw.mpFeatureIdToWeight.get(featureId) + w
										* value);
					} else {
						fw.mpFeatureIdToWeight.put(featureId, w * value);
					}
				}
			}
		}
		return fw;
	}

	/**
	 * Predict the class using feature weights and b value.
	 */
	public List<String> getPredictedClasses(Documents testingDocs,
			FeatureSelection featureSelection) {
		List<String> predictedClassList = new ArrayList<String>();

		for (Document document : testingDocs) {
			double sum = -this.getbValue();
			for (Feature feature : document.featuresForSVM) {
				String featureStr = feature.featureStr;
				if (!featureSelection.isFeatureSelected(featureStr)) {
					continue;
				}
				int featureId = featureSelection
						.getFeatureIdGivenFeatureStr(featureStr);
				double featureValue = feature.featureValue;
				if (this.containsFeatureId(featureId)) {
					sum += featureValue
							* this.getWeightGivenFeatureId(featureId);
				}
			}
			if (sum > 0) {
				predictedClassList.add("+1");
			} else {
				predictedClassList.add("-1");
			}
		}
		return predictedClassList;
	}

	public double getbValue() {
		return bValue;
	}

	public void setbValue(double bValue) {
		this.bValue = bValue;
	}

	public double getWeightGivenFeatureId(int featureId) {
		ExceptionUtility.assertAsException(this.containsFeatureId(featureId));
		return mpFeatureIdToWeight.get(featureId);
	}

	public boolean containsFeatureId(int featureId) {
		return mpFeatureIdToWeight.containsKey(featureId);
	}

	public void printToFile(String featureWeightsOutputFilePath) {
		StringBuilder sbContentForOutput = new StringBuilder();
		sbContentForOutput.append("b = " + bValue + "\n");
		ArrayList<Integer> sortedFeatureList = new ArrayList<Integer>();
		for (int featureId : mpFeatureIdToWeight.keySet()) {
			sortedFeatureList.add(featureId);
		}
		Collections.sort(sortedFeatureList);
		for (int featureId : sortedFeatureList) {
			sbContentForOutput.append(featureId + " : "
					+ mpFeatureIdToWeight.get(featureId) + "\n");
		}
		FileReaderAndWriter.writeFile(featureWeightsOutputFilePath,
				sbContentForOutput.toString());
	}

	public void printFeaturesRankedByWeights(FeatureSelection featureSelection,
			String featureWeightsOutputFilePath) {
		StringBuilder sbContentForOutput = new StringBuilder();
		sbContentForOutput.append("b = " + bValue + "\n");

		List<ItemWithValue> featureWithWeightList = new ArrayList<ItemWithValue>();
		for (Map.Entry<Integer, Double> entry : mpFeatureIdToWeight.entrySet()) {
			int featureId = entry.getKey();
			double weight = entry.getValue();
			featureWithWeightList.add(new ItemWithValue(featureId, weight));
		}
		Collections.sort(featureWithWeightList);

		for (ItemWithValue iwv : featureWithWeightList) {
			int featureId = (Integer) iwv.getItem();
			double weight = iwv.getValue();
			String featureStr = featureSelection.featureIndexer
					.getFeatureStrGivenFeatureId(featureId);
			sbContentForOutput.append(featureStr + " : " + weight + "\n");
		}
		FileReaderAndWriter.writeFile(featureWeightsOutputFilePath,
				sbContentForOutput.toString());
	}
}
