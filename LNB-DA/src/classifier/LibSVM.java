package classifier;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;
import nlp.Document;
import nlp.Documents;
import classificationevaluation.ClassificationEvaluation;
import feature.Feature;
import feature.FeatureSelection;

/**
 * Call the API of LibSVM to do the training and testing.
 * 
 * One requirement:
 * 
 * 1. Features must be in increasing order.
 */
public class LibSVM extends BaseClassifier {
	private svm_model model = null;
	private svm_parameter svm_param = null;

	public LibSVM(FeatureSelection featureSelection2,
			ClassifierParameters param2) {
		featureSelection = featureSelection2;

		param = param2;

		svm_param = new svm_parameter();
		svm_param.probability = 0;
		svm_param.coef0 = 0.0;
		svm_param.degree = 3;
		svm_param.gamma = 0.0;
		svm_param.nu = 0.5;
		svm_param.C = 1;
		svm_param.svm_type = svm_parameter.C_SVC;
		svm_param.kernel_type = svm_parameter.LINEAR;
		svm_param.cache_size = 20000;
		svm_param.eps = 0.001;
	}

	public LibSVM(
			FeatureSelection featureSelection2,
			Map<String, Map<String, double[]>> mpSelectedFeatureStrToDomainToProbs2,
			ClassifierParameters param2) {
		featureSelection = featureSelection2;
		mpSelectedFeatureStrToDomainToProbs = mpSelectedFeatureStrToDomainToProbs2;

		param = param2;

		svm_param = new svm_parameter();
		svm_param.probability = 0;
		svm_param.coef0 = 0.0;
		svm_param.degree = 3;
		svm_param.gamma = 0.0;
		svm_param.nu = 0.5;
		svm_param.C = 1;
		svm_param.svm_type = svm_parameter.C_SVC;
		svm_param.kernel_type = svm_parameter.LINEAR;
		svm_param.cache_size = 20000;
		svm_param.eps = 0.001;
	}

	@Override
	public void train(Documents trainingDocs) {
		// Convert the training features to svm_problem.
		svm_problem problem = new svm_problem();
		int dataCount = trainingDocs.size();
		problem.l = dataCount;

		int[] labelIntegers = trainingDocs.getLabelsAsIntegers();
		problem.y = new double[dataCount];
		for (int d = 0; d < dataCount; ++d) {
			problem.y[d] = 1.0 * labelIntegers[d];
		}

		problem.x = new svm_node[dataCount][];
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
			problem.x[d] = new svm_node[featureIds.size()];
			for (int i = 0; i < featureIds.size(); ++i) {
				int featureId = featureIds.get(i);
				double featureValue = mpFeatureIdToFeatureValue.get(featureId);

				svm_node node = new svm_node();
				node.index = featureId;
				node.value = featureValue;
				problem.x[d][i] = node;
			}
		}

		model = svm.svm_train(problem, svm_param);
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
			svm_node[] x = new svm_node[featureIds.size()];
			for (int i = 0; i < featureIds.size(); ++i) {
				int featureId = featureIds.get(i);
				double featureValue = mpFeatureIdToFeatureValue.get(featureId);

				svm_node node = new svm_node();
				node.index = featureId;
				node.value = featureValue;
				x[i] = node;
			}

			double predict = svm.svm_predict(model, x);
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
