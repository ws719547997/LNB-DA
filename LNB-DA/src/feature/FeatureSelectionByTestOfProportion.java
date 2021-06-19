package feature;

import java.util.Map;

import statistics.SignificanceTest;
import nlp.Documents;

public class FeatureSelectionByTestOfProportion extends FeatureSelection {
	private SignificanceTest significanceTest = null;

	public FeatureSelectionByTestOfProportion(Documents trainingDocs,
			double significanceLevel) {
		significanceTest = new SignificanceTest();
		informationGain = new InformationGain(trainingDocs);

		for (Map.Entry<String, Double> entry : informationGain.mpFeatureStrToInformationGain
				.entrySet()) {
			String featureStr = entry.getKey();
			double probOfFeatureGivenPositive = informationGain
					.getProbOfFeatureGivenPositive(featureStr);
			double probOfFeatureGivenNegative = informationGain
					.getProbOfFeatureGivenNegative(featureStr);

			boolean isDifferent = significanceTest
					.isProportionSignificantDifferentTwoTailed(
							probOfFeatureGivenPositive,
							trainingDocs.getNoOfPositiveLabels(),
							probOfFeatureGivenNegative,
							trainingDocs.getNoOfNegativeLabels(),
							significanceLevel);

			if (isDifferent) {
				// The feature is selected.
				selectedFeatureStrs.add(featureStr);
				featureIndexer.addFeatureStrIfNotExist(featureStr);
				mpFeatureStrToSelected.put(featureStr, true);
			} else {
				mpFeatureStrToSelected.put(featureStr, false);
			}
		}
	}

}
