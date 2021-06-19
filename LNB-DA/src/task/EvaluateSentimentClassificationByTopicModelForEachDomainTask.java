package task;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import classificationevaluation.ClassificationEvaluation;
import nlp.Document;
import nlp.Documents;
import topicmodel.TopicModel;

public class EvaluateSentimentClassificationByTopicModelForEachDomainTask {
	// Inputs.
	public String input100P100NReviewElectronicsDirectory = "..\\Data\\Input\\DifferentDomains\\100+100-\\50Electronics_100+100-\\";
	public String input100P100NReviewNonElectronicsDirectory = "..\\Data\\Input\\DifferentDomains\\100+100-\\50NonElectronics_100+100-\\";

	public List<TopicModel> topicModelList = null;

	// 100+100- reviews.
	public EvaluateSentimentClassificationByTopicModelForEachDomainTask(
			List<TopicModel> topicModelList2) {
		topicModelList = topicModelList2;
	}

	public void run() {
		// Classify each document in each domain.
		for (TopicModel topicModel : topicModelList) {
			Documents documentsOfThisDomain = topicModel.corpus.documents;
			double[][] dsdist = topicModel.getDocumentSentimentDistribution();
			// System.out.println(documentsOfThisDomain.size());
			// System.out.println(dsdist.length);
			for (int d = 0; d < documentsOfThisDomain.size(); ++d) {
				Document document = documentsOfThisDomain.getDocument(d);
				double probOfPositive = dsdist[d][0];
				double probOfNegative = dsdist[d][topicModel.param.S - 1];
				if (probOfPositive > probOfNegative) {
					document.predict = "+1";
				} else {
					document.predict = "-1";
				}
			}
		}
		// Evaluate all documents.
		// for (TopicModel topicModel : topicModelList) {
		// String domain = topicModel.corpus.domain;
		// Documents documentsOfThisDomain = topicModel.corpus.documents;
		// ClassificationEvaluation evaluation = new ClassificationEvaluation(
		// documentsOfThisDomain.getLabels(),
		// documentsOfThisDomain.getPredicts());
		// }

		// Evaluate only 100+100- documents.
		evaluate100P100NDocuments();
	}

	private void evaluate100P100NDocuments() {
		List<Documents> documentsOf100P100NOfAllDomains = new ArrayList<Documents>();
		documentsOf100P100NOfAllDomains
				.addAll(Documents
						.readListOfDocumentsFromDifferentDomains(input100P100NReviewElectronicsDirectory));
		documentsOf100P100NOfAllDomains
				.addAll(Documents
						.readListOfDocumentsFromDifferentDomains(input100P100NReviewNonElectronicsDirectory));

		// Map from domain to a set of review ids.
		Map<String, Set<String>> mpDomainToSetOfReviewIds = new HashMap<String, Set<String>>();
		for (Documents documentsOfThisDomain : documentsOf100P100NOfAllDomains) {
			String domain = documentsOfThisDomain.domain;
			mpDomainToSetOfReviewIds.put(domain, new HashSet<String>());
			Set<String> setOfReviewIds = mpDomainToSetOfReviewIds.get(domain);
			for (Document document : documentsOfThisDomain) {
				setOfReviewIds.add(document.reviewId);
			}
		}

		for (TopicModel topicModel : topicModelList) {
			String domain = topicModel.corpus.domain;
			if (!mpDomainToSetOfReviewIds.containsKey(domain)) {
				continue;
			}
			Documents documentsOfThisDomain = topicModel.corpus.documents;
			Documents documentsForEvaluation = new Documents();
			Set<String> setOfReviewIdsForEvaluation = mpDomainToSetOfReviewIds
					.get(domain);
			for (Document document : documentsOfThisDomain) {
				if (setOfReviewIdsForEvaluation.contains(document.reviewId)) {
					documentsForEvaluation.addDocument(document);
				}
			}
			ClassificationEvaluation evaluation = new ClassificationEvaluation(
					documentsForEvaluation.getLabels(),
					documentsForEvaluation.getPredicts(), domain);
			// System.out.println(domain);
			// System.out.println(evaluation.toString());
			System.out.println(domain + "\t" + evaluation.accuracy);
		}
	}
}
