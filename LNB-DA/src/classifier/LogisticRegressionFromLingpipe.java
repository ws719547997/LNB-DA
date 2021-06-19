package classifier;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import nlp.Document;
import nlp.Documents;
import classificationevaluation.ClassificationEvaluation;

import com.aliasi.matrix.SparseFloatVector;
import com.aliasi.matrix.Vector;
import com.aliasi.stats.AnnealingSchedule;
import com.aliasi.stats.LogisticRegression;
import com.aliasi.stats.RegressionPrior;

import feature.Feature;
import feature.FeatureSelection;

/**
 * Call the Lingpipe Logistic Regression package.
 */
public class LogisticRegressionFromLingpipe extends BaseClassifier {
	public LogisticRegression regressionModel = null;

	public LogisticRegressionFromLingpipe(FeatureSelection featureSelection2,
			ClassifierParameters param2) {
		featureSelection = featureSelection2;
		param = param2;
	}

	@Override
	public void train(Documents trainingDocs) {
		int dataCount = trainingDocs.size();
		Vector[] inputs = new Vector[dataCount];
		int[] outputs = trainingDocs.getLabelsAsIntegers();

		for (int d = 0; d < dataCount; ++d) {
			Document document = trainingDocs.getDocument(d);

			// Sort features by feature ids.
			List<Integer> featureIds = new ArrayList<Integer>();
			Map<Integer, Double> mpFeatureIdToFeatureValue = new HashMap<Integer, Double>();
			for (Feature feature : document.featuresForSVM) {
				String featureStr = feature.featureStr;
				if (!featureSelection.isFeatureSelected(featureStr)) {
					continue;
				}
				int featureId = featureSelection
						.getFeatureIdGivenFeatureStr(featureStr);
				featureIds.add(featureId);
				mpFeatureIdToFeatureValue.put(featureId, feature.featureValue);
			}
			Collections.sort(featureIds);

			// Assign feature ids to x.
			int[] keys = new int[featureIds.size()];
			float[] values = new float[featureIds.size()];
			for (int i = 0; i < featureIds.size(); ++i) {
				int featureId = featureIds.get(i);
				double featureValue = mpFeatureIdToFeatureValue.get(featureId);

				keys[i] = featureId;
				values[i] = (float) featureValue;
			}
			inputs[d] = new SparseFloatVector(keys, values,
					featureSelection.sizeOfSelectedFeatures() + 1);
		}

		// regressionModel = LogisticRegression.estimate(inputs, outputs,
		// RegressionPrior.noninformative(),
		// AnnealingSchedule.inverse(.05, 100), null, // null reporter
		// 0.000000001, // min improve
		// 1, // min epochs
		// 10000); // max epochs
		regressionModel = LogisticRegression.estimate(inputs, outputs,
				RegressionPrior.noninformative(),
				AnnealingSchedule.inverse(.05, 100), null, // null reporter
				0.000000001, // min improve
				1, // min epochs
				10000); // max epochs

		// Vector[] weights = regressionModel.weightVectors();
		// for (Vector weight : weights) {
		// weight.toString();
		// }

		// double[] conditionalProbs = regressionModel.classify(new DenseVector(
		// new double[] { 1, 1, 0, 2, 0 }));
		// for (int i = 0; i < conditionalProbs.length; ++i) {
		// System.out.println(conditionalProbs[i]);
		// }
	}

	@Override
	public ClassificationEvaluation test(Documents testingDocs) {
		for (Document testingDoc : testingDocs) {
			// Sort features by feature ids.
			List<Integer> featureIds = new ArrayList<Integer>();
			Map<Integer, Double> mpFeatureIdToFeatureValue = new HashMap<Integer, Double>();
			for (Feature feature : testingDoc.featuresForSVM) {
				String featureStr = feature.featureStr;
				if (!featureSelection.isFeatureSelected(featureStr)) {
					continue;
				}
				int featureId = featureSelection
						.getFeatureIdGivenFeatureStr(featureStr);
				featureIds.add(featureId);
				mpFeatureIdToFeatureValue.put(featureId, feature.featureValue);
			}
			Collections.sort(featureIds);

			// Assign feature ids to x.
			int[] keys = new int[featureIds.size()];
			float[] values = new float[featureIds.size()];
			for (int i = 0; i < featureIds.size(); ++i) {
				int featureId = featureIds.get(i);
				double featureValue = mpFeatureIdToFeatureValue.get(featureId);

				keys[i] = featureId;
				values[i] = (float) featureValue;
			}
			Vector input = new SparseFloatVector(keys, values,
					featureSelection.sizeOfSelectedFeatures() + 1);

			double[] predict = regressionModel.classify(input);
			if (predict[1] >= predict[0]) {
				testingDoc.predict = "+1";
			} else {
				testingDoc.predict = "-1";
			}
		}
		ClassificationEvaluation evaluation = new ClassificationEvaluation(
				testingDocs.getLabels(), testingDocs.getPredicts(),
				param.domain);
		return evaluation;
	}

	@Override
	public void printMisclassifiedDocuments(Documents testingDocs, String misclassifiedDocumentsForOneCVFolderForOneDomainFilePath) {

	}

}
