package svmlight;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import utility.FileOneByOneLineWriter;
import utility.FileReaderAndWriter;
import utility.StreamGobbler;
import feature.Feature;
import feature.FeatureSelection;
import main.Constant;
import nlp.Document;
import nlp.Documents;

/**
 * Updated by Zhiyuan (Brett) Chen on 2015.01.17.
 */
public class SVMLightHelper {
	/**
	 * Print the documents to the file with the format required by SVMLight.
	 * 
	 * Two requirements:
	 * 
	 * 1. Feature id starts with 1.
	 * 
	 * 2. Features must be in increasing order.
	 */
	public void printDocumentsToFile(Documents documents,
			FeatureSelection featureSelection, String outputFilepath) {
		FileOneByOneLineWriter writer = new FileOneByOneLineWriter(
				outputFilepath);
		for (Document document : documents) {
			StringBuilder sbOneLine = new StringBuilder();
			// Print label.
			sbOneLine.append(document.label);

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

			// Print features.
			for (int featureId : featureIds) {
				double featureValue = mpFeatureIdToFeatureValue.get(featureId);
				sbOneLine.append(" ");
				sbOneLine.append(featureId);
				sbOneLine.append(":");
				sbOneLine.append(featureValue);
			}
			writer.writeLine(sbOneLine.toString());
		}
		writer.close();
	}

	/**
	 * Learn the svm light model using svm_learn.exe.
	 */
	public void learnSVMLightModel(String trainingDocsFilepath,
			String svmLearningModelFilepath, Double cost_factor) {
		// Create the file.
		FileReaderAndWriter.writeFile(svmLearningModelFilepath, "");

		try {
			String svmLightLearnFilePath = Constant.SVM_LIGHT_LEARN_PATH;
			String commandLine = svmLightLearnFilePath + " -z c -j "
					+ cost_factor + " " + trainingDocsFilepath + " "
					+ svmLearningModelFilepath;

			Runtime rt = Runtime.getRuntime();
			Process proc = rt.exec(commandLine);

			// any error message.
			StreamGobbler errorGobbler = new StreamGobbler(
					proc.getErrorStream(), "ERROR");

			// any output.
			StreamGobbler outputGobbler = new StreamGobbler(
					proc.getInputStream(), "OUTPUT");

			// kick them off
			errorGobbler.start();
			outputGobbler.start();

			// any error.
			proc.waitFor();
			// int exitVal = proc.waitFor(); // Call waitFor() to wait the
			// process;
			// System.out.println("SVM Light Training ExitValue: " + exitVal);
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}

	/**
	 * Test the svm light model using svm_classify.exe.
	 */
	public List<String> getPredictedClasses(String testingDocsFilepath,
			String svmLearningModelFilepath, String testingResultsFilepath) {
		try {
			String svmLightTestFilePath = Constant.SVM_LIGHT_CLASSIFY_PATH;
			String commandLine = svmLightTestFilePath + " "
					+ testingDocsFilepath + " " + svmLearningModelFilepath
					+ " " + testingResultsFilepath;

			Runtime rt = Runtime.getRuntime();
			Process proc = rt.exec(commandLine);

			// any error message.
			StreamGobbler errorGobbler = new StreamGobbler(
					proc.getErrorStream(), "ERROR");

			// any output.
			StreamGobbler outputGobbler = new StreamGobbler(
					proc.getInputStream(), "OUTPUT");

			// kick them off
			errorGobbler.start();
			outputGobbler.start();

			proc.waitFor();
			// int exitVal = proc.waitFor(); // Call waitFor() to wait the
			// process;

			List<String> predictedClasses = new ArrayList<String>();
			// Read the testing results file.
			List<String> lines = FileReaderAndWriter
					.readFileAllLines(testingResultsFilepath);
			for (String line : lines) {
				double sum = Double.parseDouble(line);
				if (sum > 0) {
					predictedClasses.add("+1");
				} else {
					predictedClasses.add("-1");
				}
			}
			return predictedClasses;
		} catch (Exception ex) {
			ex.printStackTrace();
		}
		return null;
	}
}
