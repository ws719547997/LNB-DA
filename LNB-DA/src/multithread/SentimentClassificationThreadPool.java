package multithread;

import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import classificationevaluation.ClassificationEvaluation;
import classifier.ClassificationKnowledge;
import classifier.ClassifierParameters;
import nlp.Documents;

/**
 * This implements multithreading pool which is able to return a list of
 * sentiment classification results (for each domain).
 */
public class SentimentClassificationThreadPool {
	private int numberOfThreads = 1;
	private ExecutorService executor = null;
	private List<Future<ClassificationEvaluation>> futureList = new ArrayList<Future<ClassificationEvaluation>>();

	public Map<String, ClassificationEvaluation> mpClassificationEvaluation = null;

	/**
	 * set the number of threads
	 * @param numberOfThreads2
	 */
	public SentimentClassificationThreadPool(int numberOfThreads2) {
		numberOfThreads = numberOfThreads2;
		executor = Executors.newFixedThreadPool(numberOfThreads);
	}

	/**
	 * Only target documents, no knowledge
	 * @param documents
	 * @param param
	 */
	public void addTask(Documents documents, ClassifierParameters param) {
		try {
			Callable<ClassificationEvaluation> callable = new SentimentClassificationCallable(
					documents, param);
			Future<ClassificationEvaluation> future = executor.submit(callable);
			futureList.add(future);
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}

	/**
	 * Target documents, source documents, no knowledge
	 * @param documents
	 * @param documentsOfOtherDomains
	 * @param param
	 */
	public void addTask(Documents documents,
			List<Documents> documentsOfOtherDomains, ClassifierParameters param) {
		try {
			Callable<ClassificationEvaluation> callable = new SentimentClassificationCallable(
					documents, documentsOfOtherDomains, param);
			Future<ClassificationEvaluation> future = executor.submit(callable);
			futureList.add(future);
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}

	/**
	 * Target documents, knowledge
	 * @param documents
	 * @param knowledge
	 * @param param
	 */
	public void addTask(Documents documents, ClassificationKnowledge knowledge,
			ClassifierParameters param) {
		try {
			Callable<ClassificationEvaluation> callable = new SentimentClassificationCallable(
					documents, knowledge, param);
			Future<ClassificationEvaluation> future = executor.submit(callable);
			futureList.add(future);
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}

	/**
	 * Target documents, source documents, knowledge
	 * @param documents
	 * @param documentsOfOtherDomains
	 * @param knowledge
	 * @param param
	 */
	public void addTask(Documents documents, List<Documents> documentsOfOtherDomains,
						ClassificationKnowledge knowledge, ClassifierParameters param) {
		try {
			Callable<ClassificationEvaluation> callable = new SentimentClassificationCallable(documents,
					documentsOfOtherDomains, knowledge, param);
			Future<ClassificationEvaluation> future = executor.submit(callable);
			futureList.add(future);
		} catch (Exception ex) {
			ex.printStackTrace();
		}

	}

	/**
	 * Target documents, pastKnowledgeList
	 * @param documents
	 * @param knowledge
	 * @param param
	 */
	public void addTask(Documents documents, Map<String, ClassificationKnowledge> knowledge, ClassifierParameters param) {
		try {
			Callable<ClassificationEvaluation> callable =
					new SentimentClassificationCallable(documents, knowledge, param);
			Future<ClassificationEvaluation> future = executor.submit(callable);
			futureList.add(future);
		} catch (Exception ex) {
			ex.printStackTrace();
		}

	}

	/**
	 * Training documents, testing documents
	 * @param trainingDocs
	 * @param testingDocs
	 * @param param
	 */
	public void addTask(Documents trainingDocs, Documents testingDocs, ClassifierParameters param) {
		try {
			Callable<ClassificationEvaluation> callable =
					new SentimentClassificationCallable(trainingDocs, testingDocs, param);
			Future<ClassificationEvaluation> future = executor.submit(callable);
			futureList.add(future);
		} catch (Exception ex) {
			ex.printStackTrace();
		}

	}

	/**
	 * Getting classification evaluation (results)
	 */
	public void awaitTermination() {
		try {
			executor.shutdown();
			executor.awaitTermination(Long.MAX_VALUE, TimeUnit.DAYS);

			mpClassificationEvaluation = new TreeMap<String, ClassificationEvaluation>();
			// Get all the topic models. Note that they are sorted according to
			// the domain name alphabetically.
			for (Future<ClassificationEvaluation> future : futureList) {
				ClassificationEvaluation evaluation = future.get();
				String domain = evaluation.domain;
				mpClassificationEvaluation.put(domain, evaluation);
			}
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}
}
