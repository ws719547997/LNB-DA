package classifier;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import nlp.Document;
import nlp.Documents;
import classificationevaluation.ClassificationEvaluation;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;
import feature.Feature;
import feature.FeatureSelection;

/**
 * Call the API of LibSVM to do the training and testing.
 * 
 * One requirement:
 * 
 * 1. Features must be in increasing order.
 */
public class LibLinear extends BaseClassifier {
	private Parameter liblinearParameter = null;
	private Model liblinearModel = null;

	public LibLinear(FeatureSelection featureSelection2,
			ClassifierParameters param2) {
		featureSelection = featureSelection2;

		param = param2;

		SolverType solver = SolverType.L2R_L2LOSS_SVC; // -s 0
		double C = 1.0; // cost of constraints violation
		double eps = 0.01; // stopping criteria
		liblinearParameter = new Parameter(solver, C, eps);
	}

	@Override
	public void train(Documents trainingDocs) {
		// Convert the training features to problem.
		Problem problem = new Problem();
		int dataCount = trainingDocs.size();
		problem.l = dataCount; // number of training examples.
		problem.n = featureSelection.sizeOfSelectedFeatures(); // number of
																// features.

		int[] labelIntegers = trainingDocs.getLabelsAsIntegers();
		problem.y = new double[dataCount];
		for (int d = 0; d < dataCount; ++d) {
			problem.y[d] = 1.0 * labelIntegers[d];
		}

		problem.x = new FeatureNode[dataCount][];
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
			problem.x[d] = new FeatureNode[featureIds.size()];
			for (int i = 0; i < featureIds.size(); ++i) {
				int featureId = featureIds.get(i);
				double featureValue = mpFeatureIdToFeatureValue.get(featureId);

				de.bwaldvogel.liblinear.Feature node = new FeatureNode(
						featureId, featureValue);
				problem.x[d][i] = node;
			}
		}

		liblinearModel = Linear.train(problem, liblinearParameter);
		// File modelFile = new File("model");
		// model.save(modelFile);
		// // load model or use it directly
		// model = Model.load(modelFile);
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
			de.bwaldvogel.liblinear.Feature[] x = new de.bwaldvogel.liblinear.Feature[featureIds
					.size()];
			for (int i = 0; i < featureIds.size(); ++i) {
				int featureId = featureIds.get(i);
				double featureValue = mpFeatureIdToFeatureValue.get(featureId);

				de.bwaldvogel.liblinear.Feature node = new FeatureNode(
						featureId, featureValue);
				x[i] = node;
			}

			double predict = Linear.predict(liblinearModel, x);
			if (predict > 0) {
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
