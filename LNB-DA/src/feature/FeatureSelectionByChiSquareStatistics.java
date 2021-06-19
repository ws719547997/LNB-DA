package feature;

import java.util.Map;

import statistics.SignificanceTest;
import nlp.Documents;

public class FeatureSelectionByChiSquareStatistics extends FeatureSelection {
	public FeatureSelectionByChiSquareStatistics(Documents trainingDocs,
			double significanceLevel) {
		informationGain = new InformationGain(trainingDocs);

		for (Map.Entry<String, Double> entry : informationGain.mpFeatureStrToInformationGain
				.entrySet()) {
			String featureStr = entry.getKey();
			int N = trainingDocs.size();
			// freq(+, f).
			int A = informationGain.mpFeatureStrToPositiveWithFeatureCount
					.get(featureStr);
			// freq(-, f).
			int B = informationGain.mpFeatureStrToNegativeWithFeatureCount
					.get(featureStr);
			// freq(+, ~f).
			int C = informationGain.mpFeatureStrToPositiveWithoutFeatureCount
					.get(featureStr);
			// freq(-, f).
			int D = informationGain.mpFeatureStrToNegativeWithoutFeatureCount
					.get(featureStr);

			SignificanceTest test = new SignificanceTest();
			boolean isSignificant = test.isChiSqurareSignificant(N, A, B, C, D,
					significanceLevel);
			if (isSignificant) {
				selectedFeatureStrs.add(featureStr);
				featureIndexer.addFeatureStrIfNotExist(featureStr);
				mpFeatureStrToSelected.put(featureStr, true);
			} else {
				mpFeatureStrToSelected.put(featureStr, false);
			}
		}
	}

}
