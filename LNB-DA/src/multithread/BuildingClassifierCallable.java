package multithread;

import java.util.concurrent.Callable;

import classifier.BaseClassifier;
import classifier.ClassifierParameters;
import feature.FeatureGenerator;
import feature.FeatureSelection;
import nlp.Documents;

public class BuildingClassifierCallable implements Callable<BaseClassifier> {
	private Documents trainingDocs = null;
	private ClassifierParameters param = null;

	public BuildingClassifierCallable(Documents trainingDocs2,
			ClassifierParameters param2) {
		trainingDocs = trainingDocs2;
		param = param2;
	}

	@Override
	/**
	 * Run the topic model in a domain and print it into the disk.
	 */
	public BaseClassifier call() throws Exception {
		try {
			System.out.println("\"" + param.domain + "\" <"
					+ param.classifierName + "> Starts...");

			// Build the classifier.
			// Feature generation.
			FeatureGenerator featureGenerator = new FeatureGenerator(param);
			featureGenerator
					.generateAndAssignFeaturesToTrainingAndTestingDocuments(
							trainingDocs, new Documents(), null);
			// Feature selection.
			FeatureSelection featureSelection = FeatureSelection
					.selectFeatureSelection(trainingDocs, param);

			// Build the classifier.
			BaseClassifier classifier = BaseClassifier.selectClassifier(
					param.classifierName, featureSelection, null, param);
			classifier.train(trainingDocs);

			System.out.println("\"" + param.domain + "\" <"
					+ param.classifierName + "> Ends...");

			return classifier;
		} catch (Exception ex) {
			ex.printStackTrace();
		}
		return null;
	}

}
