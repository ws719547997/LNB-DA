package classifier;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import utility.FileOneByOneLineWriter;
import main.Constant;
import nlp.Document;
import nlp.Documents;
import classificationevaluation.ClassificationEvaluation;
import feature.Feature;
import feature.FeatureSelection;

/**
 * Call the Lingpipe Logistic Regression package.
 */
public class MyLogisticRegression extends BaseClassifier {
	List<Instance> data = null;

	public double[] weights = null;
	private double learningRate = 0.0;

	public MyLogisticRegression(FeatureSelection featureSelection2,
			ClassifierParameters param2) {
		featureSelection = featureSelection2;
		param = param2;

		// +1 for w_0.
		weights = new double[featureSelection2.sizeOfSelectedFeatures() + 2];
		for (int i = 0; i < weights.length; ++i) {
			if (i % 2 == 0) {
				weights[i] = 0.001;
			} else {
				weights[i] = -0.001;
			}
		}
	}

	/***************** Stochastic Gradient Descent *****************/
	private void SGDEntry() {
		// Shuffle the data.
		Collections.shuffle(data, new Random(Constant.RANDOMSEED));
		double before = getObjectiveFunctionValueForInstancesOriginal(data);
		int iter = 0;
		while (true) {
			++iter;
			learningRate = param.learningRate
					/ (1 + param.learningRateChange * param.learningRate
							* param.regCoefficientAlpha * iter);
			// System.out.println("Iteration " + iter + " : " + learningRate +
			// " "
			// + before);
			if (param.maxSGDIterations >= 0 && iter > param.maxSGDIterations) {
				break;
			}
			for (int d = 0; d < data.size(); ++d) {
				Instance instance = data.get(d);
				if (instance.y > 0) {
					SGDForPositiveInstance(instance);
				} else {
					SGDForNegativeInstance(instance);
				}
			}
			double after = getObjectiveFunctionValueForInstancesOriginal(data);
			if (Math.abs(after - before) <= param.convergenceDifference) {
				break;
			}
			before = after;
		}
	}

	private void SGDForPositiveInstance(Instance instance) {
		// Computer sum_i {w_i * x_i}.
		double linearSum = 0.0;
		for (Map.Entry<Integer, Double> entry : instance.mpFeatureToFeatureValue
				.entrySet()) {
			int featureId = entry.getKey();
			double featureValue = entry.getValue();
			linearSum += weights[featureId] * featureValue;
		}
		double expSum = Math.exp(linearSum);
		// Compute the gradient of each feature for the positive class.
		Map<Integer, Double> gradientOfFeature = new HashMap<Integer, Double>();
		for (Map.Entry<Integer, Double> entry : instance.mpFeatureToFeatureValue
				.entrySet()) {
			int featureId = entry.getKey();
			double featureValue = entry.getValue();
			double gradient = linearSum * featureValue / (expSum + 1)
					- linearSum * featureValue * (expSum - 1) / (expSum + 1)
					/ (expSum + 1);
			if (Double.isNaN(gradient)) {
				gradient = 0;
			}
			gradientOfFeature.put(featureId, gradient);
		}

		// Note that we need to compute all of the gradients first and then
		// update the counts.
		updateWeights(instance, gradientOfFeature);
	}

	private void SGDForNegativeInstance(Instance instance) {
		// Computer sum_i {w_i * x_i}.
		double linearSum = 0.0;
		for (Map.Entry<Integer, Double> entry : instance.mpFeatureToFeatureValue
				.entrySet()) {
			int featureId = entry.getKey();
			double featureValue = entry.getValue();
			linearSum += weights[featureId] * featureValue;
		}
		double expSum = Math.exp(linearSum);
		// Compute the gradient of each feature for the positive class.
		Map<Integer, Double> gradientOfFeature = new HashMap<Integer, Double>();
		for (Map.Entry<Integer, Double> entry : instance.mpFeatureToFeatureValue
				.entrySet()) {
			int featureId = entry.getKey();
			double featureValue = entry.getValue();
			double gradient = -linearSum * featureValue / (expSum + 1)
					- linearSum * featureValue * (1 - expSum) / (expSum + 1)
					/ (expSum + 1);
			if (Double.isNaN(gradient)) {
				gradient = 0;
			}
			gradientOfFeature.put(featureId, gradient);
		}

		// Note that we need to compute all of the gradients first and then
		// update the counts.
		updateWeights(instance, gradientOfFeature);
	}

	private double getObjectiveFunctionValueForInstancesOriginal(
			List<Instance> data2) {
		double sum = 0.0;
		for (int d = 0; d < data.size(); ++d) {
			Instance instance = data.get(d);

			double linearSum = 0.0;
			for (Map.Entry<Integer, Double> entry : instance.mpFeatureToFeatureValue
					.entrySet()) {
				int featureId = entry.getKey();
				double featureValue = entry.getValue();
				linearSum += weights[featureId] * featureValue;
			}
			double expSum = Math.exp(linearSum);
			if (Double.isInfinite(expSum)) {
				continue;
			}

			if (instance.y > 0) {
				sum += (expSum - 1) / (expSum + 1);
			} else {
				sum += (1 - expSum) / (expSum + 1);
			}
		}
		return sum;
	}

	/***************** Update Xs in SGD *****************/
	public void updateWeights(Instance instance,
			Map<Integer, Double> mpFeatureToDelta) {
		for (Map.Entry<Integer, Double> entry : instance.mpFeatureToFeatureValue
				.entrySet()) {
			int featureId = entry.getKey();
			double delta = mpFeatureToDelta.get(featureId);
			weights[featureId] -= learningRate * delta;
		}
	}

	@Override
	public void train(Documents trainingDocs) {
		// Convert the training documents to data.
		data = convertDocumentsToData(trainingDocs);
		if (param.convergenceDifference != Double.MAX_VALUE) {
			// Stochastic gradient descent.
			SGDEntry();
		}
	}

	private List<Instance> convertDocumentsToData(Documents trainingDocs) {
		List<Instance> data = new ArrayList<Instance>();
		for (Document trainingDoc : trainingDocs) {
			Instance instance = new Instance();
			instance.mpFeatureToFeatureValue.put(0, 1.0); // for w_0.

			for (Feature feature : trainingDoc.featuresForSVM) {
				String featureStr = feature.featureStr;
				if (featureSelection != null
						&& !featureSelection.isFeatureSelected(featureStr)) {
					// The feature is not selected.
					continue;
				}
				// +1 for w_0.
				int featureId = featureSelection
						.getFeatureIdGivenFeatureStr(featureStr) + 1;
				double featurevalue = feature.featureValue;
				if (!instance.mpFeatureToFeatureValue.containsKey(featureId)) {
					instance.mpFeatureToFeatureValue.put(featureId,
							featurevalue);
				}
			}
			instance.y = Integer.parseInt(trainingDoc.label);
			data.add(instance);
		}
		return data;
	}

	@Override
	public ClassificationEvaluation test(Documents testingDocs) {
		String filepath = param.nbRootDirectory + param.domain + ".txt";
		FileOneByOneLineWriter writer = new FileOneByOneLineWriter(filepath);
		for (Document testingDoc : testingDocs) {
			testingDoc.predict = this.getBestClassByClassification(
					testingDoc.featuresForSVM, writer);
		}
		writer.close();
		ClassificationEvaluation evaluation = new ClassificationEvaluation(
				testingDocs.getLabels(), testingDocs.getPredicts(),
				param.domain);
		return evaluation;
	}

	@Override
	public void printMisclassifiedDocuments(Documents testingDocs, String misclassifiedDocumentsForOneCVFolderForOneDomainFilePath) {

	}

	/**
	 * Classify the content and return best category with highest probability.
	 */
	public String getBestClassByClassification(Set<Feature> featuresForSVM,
			FileOneByOneLineWriter writer) {
		double sum = weights[0];
		for (Feature feature : featuresForSVM) {
			String featureStr = feature.featureStr;
			if (!featureSelection.mpFeatureStrToSelected
					.containsKey(featureStr)) {
				continue;
			}
			int featureId = featureSelection
					.getFeatureIdGivenFeatureStr(featureStr) + 1;
			double featureValue = feature.featureValue;
			sum += weights[featureId] * featureValue;
		}
		if (sum >= 0) {
			return "+1";
		} else {
			return "-1";
		}
	}

}
