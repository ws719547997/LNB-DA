package task;

import classificationevaluation.ClassificationEvaluation;
import classifier.ClassificationKnowledge;
import classifier.ClassifierParameters;
import classifier.NaiveBayes;
import feature.FeatureGenerator;
import feature.FeatureSelection;
import main.CmdOption;
import multithread.SentimentClassificationThreadPool;
import nlp.Documents;
import utility.ArraySumAndAverageAndMaxAndMin;
import utility.FileReaderAndWriter;

import java.util.*;

/**
 * Created by hao on 2/5/2018.
 * Knowledgeable naive Bayes learning,
 * where the training and testing data are from a single domain
 * with previous knowledge using fold cross validation.
 * The result of this method is comparable with Zhiyuan's result.
 */

public class KnowledgeableNaiveBayes {
    public CmdOption cmdOption = null;
    private List<String> domainsToEvaluate = null;

    public KnowledgeableNaiveBayes(CmdOption cmdOption2) {
        cmdOption = cmdOption2;

        // get evaluated domain name list
        if (cmdOption.inputListOfDomainsToEvaluate != null) {
            domainsToEvaluate = FileReaderAndWriter
                    .readFileAllLines(cmdOption.inputListOfDomainsToEvaluate);
        }
    }

    public Map<String, ClassificationEvaluation> run() {
        List<Documents> documentsOfAllDomains = readDocuments();

        // read or generate knowledge for each target domain
        Map<String, ClassificationKnowledge> mpDomainToKnowledge
                = readOrGenerateClassificationKnowledgeForEachTargetDomain(documentsOfAllDomains);

        // cmdOption.nthreads: the number of maximum threads in multithreading
        SentimentClassificationThreadPool threadPool = new SentimentClassificationThreadPool(cmdOption.nthreads);
        for (int domain_id = 0; domain_id < domainsToEvaluate.size(); ++domain_id) {
            String targetDomain = domainsToEvaluate.get(domain_id);
            // get documents of target domain -> targetDocs
            Documents documents = null;
            assert documentsOfAllDomains != null;
            for (int j = 0; j < documentsOfAllDomains.size(); ++j) {
                if (Objects.equals(targetDomain, documentsOfAllDomains.get(j).domain)) {
                    documents = documentsOfAllDomains.get(j).getDeepClone();
                    break;
                }
            }

            // only testing one domain
//			if (!domain.equals("CombAutomotive0-9")) {
//				continue;
//			}

            // classifier parameters
            ClassifierParameters param = new ClassifierParameters(documents, cmdOption);
            // Get the knowledge for this target domain.
            ClassificationKnowledge knowledgeForTargetDomain = mpDomainToKnowledge.get(targetDomain);

            // and use both target and source domains to train the model, but only test target domain.
            if (param.includeSourceDomainsLabeledDataForTraining) {
                // get documents of source domains
                System.out.println("also add source domains to train...");
                List<Documents> sourceDomainsDocs = new ArrayList<>();
                for (int j = 0; j < documentsOfAllDomains.size(); ++j) {
                    if (domain_id == j) {
                        continue;
                    }
                    sourceDomainsDocs.add(documentsOfAllDomains.get(j));
                }
                // use both target and source domains to train the model and test this target domain
                threadPool.addTask(documents, sourceDomainsDocs, knowledgeForTargetDomain, param);
            } else {
                // only use target domain to train the model and test this target domain
                threadPool.addTask(documents, knowledgeForTargetDomain, param);
            }
        }
        threadPool.awaitTermination();
        Map<String, ClassificationEvaluation> mpDomainToClassificationEvaluation = threadPool.mpClassificationEvaluation;
        /** print results to files */
        ClassifierParameters paramTemp = new ClassifierParameters();
        String resultName = paramTemp.classifierName + "_";
        if (paramTemp.includeTargetDomainLabeledDataForTraining
                && paramTemp.includeSourceDomainsLabeledDataForTraining) {
            resultName += "ST";
        } else if (paramTemp.includeTargetDomainLabeledDataForTraining) {
            resultName += "T";
        } else if (paramTemp.includeSourceDomainsLabeledDataForTraining) {
            resultName += "S";
        }
        // Accuracy.
        StringBuilder sbOutput = new StringBuilder();
        double[] accuracies = new double[mpDomainToClassificationEvaluation.size()];
        int i = 0;
        for (Map.Entry<String, ClassificationEvaluation> entry : mpDomainToClassificationEvaluation
                .entrySet()) {
            ClassificationEvaluation evaluation = entry.getValue();
            String domain = evaluation.domain;
            sbOutput.append(domain + "\t" + evaluation.accuracy);
//			sbOutput.append(evaluation.accuracy);
            sbOutput.append(System.lineSeparator());
            accuracies[i++] = evaluation.accuracy;
        }
        FileReaderAndWriter.writeFile(cmdOption.outputSentimentClassificationAccuracy
                        + "/" + resultName + "/" + "ACC_NB.txt",
                sbOutput.toString());
        System.out.println("Average Accuracy: "
                + ArraySumAndAverageAndMaxAndMin.getAverage(accuracies));
        // F1-score in both classes.
        sbOutput = new StringBuilder();
        double[] f1Scores = new double[mpDomainToClassificationEvaluation
                .size()];
        i = 0;
        for (Map.Entry<String, ClassificationEvaluation> entry : mpDomainToClassificationEvaluation
                .entrySet()) {
            ClassificationEvaluation evaluation = entry.getValue();
            // String domain = evaluation.domain;
            // sbOutput.append(domain + "\t" + evaluation.accuracy);
            sbOutput.append(evaluation.f1scoreBothClasses);
            sbOutput.append(System.lineSeparator());
            f1Scores[i++] = evaluation.f1scoreBothClasses;
        }
        FileReaderAndWriter.writeFile(cmdOption.outputSentimentClassificationF1Score
                + "/" + resultName + "/" + "F1_BothClasses_NB.txt", sbOutput.toString());
        System.out.println("Average F1-score (Both Classes): "
                + ArraySumAndAverageAndMaxAndMin.getAverage(f1Scores));

        // F1-score in the positive class.
        sbOutput = new StringBuilder();
        f1Scores = new double[mpDomainToClassificationEvaluation.size()];
        i = 0;
        for (Map.Entry<String, ClassificationEvaluation> entry : mpDomainToClassificationEvaluation
                .entrySet()) {
            ClassificationEvaluation evaluation = entry.getValue();
            // String domain = evaluation.domain;
            // sbOutput.append(domain + "\t" + evaluation.accuracy);
            sbOutput.append(evaluation.f1score);
            sbOutput.append(System.lineSeparator());
            f1Scores[i++] = evaluation.f1score;
        }
        FileReaderAndWriter
                .writeFile(cmdOption.outputSentimentClassificationF1Score
                        + "/" + resultName + "/" + "F1_Positive_NB.txt", sbOutput.toString());
        System.out.println("Average F1-score (Positive): "
                + ArraySumAndAverageAndMaxAndMin.getAverage(f1Scores));

        // F1-score in the negative class.
        sbOutput = new StringBuilder();
        f1Scores = new double[mpDomainToClassificationEvaluation.size()];
        i = 0;
        for (Map.Entry<String, ClassificationEvaluation> entry : mpDomainToClassificationEvaluation
                .entrySet()) {
            ClassificationEvaluation evaluation = entry.getValue();
            // String domain = evaluation.domain;
            // sbOutput.append(domain + "\t" + evaluation.accuracy);
            sbOutput.append(evaluation.f1scoreNegativeClass);
            sbOutput.append(System.lineSeparator());
            f1Scores[i++] = evaluation.f1scoreNegativeClass;
        }
        FileReaderAndWriter
                .writeFile(cmdOption.outputSentimentClassificationF1Score
                        + "/" + resultName + "/" + "F1_Negative_NB.txt", sbOutput.toString());
        System.out.println("Average F1-score (Negative): "
                + ArraySumAndAverageAndMaxAndMin.getAverage(f1Scores));

        return mpDomainToClassificationEvaluation;
    }

    /**
     * According to the configuration, we read documents from different
     * directories.
     */
    private List<Documents> readDocuments() {
        InputReaderTask task = new InputReaderTask(cmdOption);
        if (cmdOption.datasetName.equals("Reuters10")) {
            task.readReuters10domains();
        } else if (cmdOption.datasetName.equals("20Newgroup")) {
            return task.read20Newsgroup();
        } else if (cmdOption.datasetName.equals("PangAndLeeMovieReviews")) {
            return task.readDocumentsFromPangAndLeeMovieReview();
        } else if (cmdOption.datasetName.equals("100P100NDomains")) {
            return task.readDocumentsListFrom100P100NDomains();
        } else if (cmdOption.datasetName.equals("1KP1KNDomains")) {
            return task.readDocumentsListFrom1KP1KNDomains();
        } else if (cmdOption.datasetName
                .equals("1KReviewNaturalClassDistributionDomains")) {
            return task
                    .readDocumentsListFrom1KReviewsNaturalClassDistributionDomains();
        } else if (cmdOption.datasetName
                .equals("DifferentProductsOfSameDomain")) {
            return task.readDocumentsFromDifferentProductsOfSameDomain();
        } else if (cmdOption.datasetName
                .equals("BalancedWithMostNegativeReviews")) {
            return task.readDocumentsFromBalancedWithMostNegativeReviews();
        }
        return null;
    }

    /**
     * read or generate classification knowledge for each target domain
     *
     * @param documentsOfAllDomains
     * @return classification knowledge
     */
    private Map<String, ClassificationKnowledge> readOrGenerateClassificationKnowledgeForEachTargetDomain(
            List<Documents> documentsOfAllDomains) {
        Map<String, ClassificationKnowledge> mpDomainToKnowledge = new HashMap<String, ClassificationKnowledge>();

        // read or generate classification knowledge for each target domain in turns
        for (int i = 0; i < documentsOfAllDomains.size(); ++i) {
            Documents documentsOfTargetDomain = documentsOfAllDomains.get(i);
            String targetDomain = documentsOfTargetDomain.domain;
            if (domainsToEvaluate != null
                    && !domainsToEvaluate.contains(targetDomain)) {
                continue;
            }

            System.out.println("Obtain knowledge for target domain " + targetDomain);
            // Build the model from labeled data of source domains and
            // record the probs.
            Documents trainingDocsFromSourceDomains = new Documents();
            trainingDocsFromSourceDomains.domain = targetDomain;
            for (int j = 0; j < documentsOfAllDomains.size(); ++j) {
                if (j == i) {
                    // if current domain j is target domain i, continue...
                    continue;
                }
                if (domainsToEvaluate != null
                        && !domainsToEvaluate.contains(documentsOfAllDomains.get(j).domain)) {
                    // checking domain validity
                    continue;
                }
                trainingDocsFromSourceDomains.addDocuments(documentsOfAllDomains.get(j).getDeepClone());
            } // have got all source domains for current target domain

            // Extract classification knowledge: indexed by word (featured word)
            // 1. total number of documents in POS and NEG category: Freq(+) and Freq(-)
            // 2. Document-level knowledge: N_{+,w}^KB and N_{-,w}^KB
            // 3. total number of words in POS and NEG category: sum_f{Freq(f, +)} and sum_f{Freq(f, -)}
            // training all past task documents at once
            NaiveBayes nbClassifier = getKnowledgeBasedOnNBClassifier(trainingDocsFromSourceDomains);
            ClassificationKnowledge knowledge = nbClassifier.knowledge;

            // 4. Domain-level knowledge: M_{+,w}^KB and M_{-,w}^KB
            // ->
            int numberOfSourceDomains = 0;
            for (int j = 0; j < documentsOfAllDomains.size(); ++j) {
                if (j == i) {
                    continue;
                }
                if (domainsToEvaluate != null
                        && !domainsToEvaluate.contains(documentsOfAllDomains.get(j).domain)) {
                    continue;
                }

                // training each past task (domain) in turns
                nbClassifier = getKnowledgeBasedOnNBClassifier(documentsOfAllDomains.get(j).getDeepClone());
                for (Map.Entry<String, double[]> entry : nbClassifier.knowledge.wordCountInPerClass
                        .entrySet()) {
                    String featureStr = entry.getKey();
                    double[] tokenCounts = entry.getValue();
                    double probOfFeatureGivenPositive = nbClassifier
                            .probTokenByIndexArray(0, tokenCounts);
                    double probOfFeatureGivenNegative = nbClassifier
                            .probTokenByIndexArray(1, tokenCounts);
                    if (!knowledge.countDomainsInPerClass
                            .containsKey(featureStr)) {
                        knowledge.countDomainsInPerClass.put(
                                featureStr, new double[2]);
                    }
                    if (probOfFeatureGivenPositive >= probOfFeatureGivenNegative) {
                        // Positive feature in this domain.
                        knowledge.countDomainsInPerClass
                                .get(featureStr)[0]++;
                    } else {
                        // Negative feature in this domain.
                        knowledge.countDomainsInPerClass
                                .get(featureStr)[1]++;
                    }
                }

                // TODO: closed now
                // print knowledge of each target domain to file
                // knowledge.printToFile(domainFilePath);

                mpDomainToKnowledge.put(targetDomain, knowledge);

                ++numberOfSourceDomains;
                if (numberOfSourceDomains >= CmdOption.numberOfMaximumSourceDomains) {
                    break;
                }
            }

//            // debugging: print word information into file
//            printWordInformationOfEachDomain(documentsOfTargetDomain,
//                    ClassifierParameters.wordInformationOfEachDomainFilepath);
        }
        return mpDomainToKnowledge;
    }

    /**
     * extract classification knowledge based on Naive Bayes
     *
     * @param trainingData
     * @return nbClassifier.knowledge
     */
    private NaiveBayes getKnowledgeBasedOnNBClassifier(Documents trainingData) {
        ClassifierParameters param = new ClassifierParameters(trainingData,
                cmdOption);
        param.classifierName = "NaiveBayes";

        // Feature generation.
        // In practice, using 1-Gram features for documents. mainly to add new item "featuresForNaiveBayes"
        FeatureGenerator featureGenerator = new FeatureGenerator(param);
        featureGenerator.generateAndAssignFeaturesToTrainingAndTestingDocuments(
                trainingData, new Documents(), null);

        // Feature selection. // In practice, all features are selected.
        // All selected features are covered by all documents of one domain.
        // The selected features are used to verify the validity of input features (i.e., input words)
        FeatureSelection featureSelection = FeatureSelection
                .selectFeatureSelection(trainingData, param);

        // Build the classifier.
        NaiveBayes nbClassifier = new NaiveBayes(featureSelection, param);
        nbClassifier.train(trainingData);

        return nbClassifier;
    }
}
