package multithread;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import classifier.BaseClassifier;
import classifier.ClassifierParameters;
import nlp.Documents;

/**
 * This implements multithreading pool which is able to return a handle of
 * classifier.
 */
public class BuildingClassifierThreadPool {
	private int numberOfThreads = 1;
	private ExecutorService executor = null;
	private List<Future<BaseClassifier>> futureList = new ArrayList<Future<BaseClassifier>>();

	public Map<String, BaseClassifier> mpDomainToClassificationEvaluation = null;

	public BuildingClassifierThreadPool(int numberOfThreads2) {
		numberOfThreads = numberOfThreads2;
		executor = Executors.newFixedThreadPool(numberOfThreads);
	}

	public void addTask(Documents trainingDocs, ClassifierParameters param) {
		try {
			Callable<BaseClassifier> callable = new BuildingClassifierCallable(
					trainingDocs, param);
			Future<BaseClassifier> future = executor.submit(callable);
			futureList.add(future);
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}

	public void awaitTermination() {
		try {
			executor.shutdown();
			executor.awaitTermination(60, TimeUnit.DAYS);

			mpDomainToClassificationEvaluation = new TreeMap<String, BaseClassifier>();
			// Get all the classifier. Note that they are sorted according to
			// the domain name alphabetically.
			for (Future<BaseClassifier> future : futureList) {
				BaseClassifier classifier = future.get();
				String domain = classifier.param.domain;
				mpDomainToClassificationEvaluation.put(domain, classifier);
			}
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}

}
