package classifier;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import nlp.Document;
import nlp.Documents;
import cc.mallet.classify.ClassifierTrainer;
import cc.mallet.classify.MaxEntTrainer;
import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.InstanceList;
import cc.mallet.types.LabelAlphabet;
import cc.mallet.types.Labeling;
import classificationevaluation.ClassificationEvaluation;

import feature.Feature;
import feature.FeatureSelection;

/**
 * Call the Mallet Logistic Regression package.
 */
public class LogisticRegressionFromMallet extends BaseClassifier {
	private ClassifierTrainer trainer = null;
	private cc.mallet.classify.Classifier classifier = null;
	private Alphabet alphabet = null;
	private LabelAlphabet classes = null;

	public LogisticRegressionFromMallet(FeatureSelection featureSelection2,
			ClassifierParameters param2) {
		featureSelection = featureSelection2;
		param = param2;
	}

	@Override
	public void train(Documents trainingDocs) {
		List<String> selectedFeatureStrs = featureSelection.selectedFeatureStrs;
		selectedFeatureStrs.add(0, "EMPTYFORINDEX0");
		String[] selectedFeatures = selectedFeatureStrs
				.toArray(new String[selectedFeatureStrs.size()]);
		alphabet = new Alphabet(selectedFeatures); 
		classes = new LabelAlphabet();

		InstanceList trainingInstances = new InstanceList(alphabet, classes);

		int dataCount = trainingDocs.size();
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
			double[] values = new double[featureIds.size()];
			for (int i = 0; i < featureIds.size(); ++i) {
				int featureId = featureIds.get(i);
				double featureValue = mpFeatureIdToFeatureValue.get(featureId);

				keys[i] = featureId;
				values[i] = (float) featureValue;
			}

			FeatureVector featureVector = new FeatureVector(alphabet, keys,
					values);
			cc.mallet.types.Instance instance = new cc.mallet.types.Instance(
					featureVector, classes.lookupLabel(document.label), null,
					null);
			trainingInstances.add(instance);
		}

		trainer = new MaxEntTrainer();
		classifier = trainer.train(trainingInstances);
	}

	@Override
	public ClassificationEvaluation test(Documents testingDocs) {
		int dataCount = testingDocs.size();
		for (int d = 0; d < dataCount; ++d) {
			Document testingDoc = testingDocs.getDocument(d);

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
			double[] values = new double[featureIds.size()];
			for (int i = 0; i < featureIds.size(); ++i) {
				int featureId = featureIds.get(i);
				double featureValue = mpFeatureIdToFeatureValue.get(featureId);

				keys[i] = featureId;
				values[i] = (float) featureValue;
			}

			FeatureVector featureVector = new FeatureVector(alphabet, keys,
					values);
			cc.mallet.types.Instance instance = new cc.mallet.types.Instance(
					featureVector, classes.lookupLabel(testingDoc.label), null,
					null);
			Labeling labeling = classifier.classify(instance).getLabeling();
			testingDoc.predict = labeling.getBestLabel().toString();
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
