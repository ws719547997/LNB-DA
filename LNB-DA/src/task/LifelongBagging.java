package task;

import java.io.File;
import java.util.*;

import classificationevaluation.ClassificationEvaluationAccumulator;
import classifier.*;
import feature.FeatureGenerator;
import feature.FeatureSelection;
import main.CmdOption;
import multithread.SentimentClassificationThreadPool;
import nlp.Document;
import nlp.Documents;
import topicmodel.TopicModel;
import utility.ArraySumAndAverageAndMaxAndMin;
import utility.CrossValidationOperatorMaintainingLabelDistribution;
import utility.FileReaderAndWriter;
import classificationevaluation.ClassificationEvaluation;
import utility.Pair;

/**
 * The training and testing data are all from a single domain using fold cross
 * validation.
 * LLV in the paper (EMNLP-2018)
 */
public class LifelongBagging {
    public CmdOption cmdOption = null;
    private List<String> domainsToEvaluate = null;

    public LifelongBagging(
            CmdOption cmdOption2) {
        cmdOption = cmdOption2;

        if (cmdOption.inputListOfDomainsToEvaluate != null) {
            domainsToEvaluate = FileReaderAndWriter
                    .readFileAllLines(cmdOption.inputListOfDomainsToEvaluate);
        }
    }

    public Map<String, ClassificationEvaluation> run() {
        List<Documents> documentsOfAllDomains = readDocuments();

        ClassifierParameters paramTemp = new ClassifierParameters();
        StringBuilder titleOutput = new StringBuilder();
        int lenStr = cmdOption.inputListOfDomainsToEvaluate.length();
        String fix = cmdOption.inputListOfDomainsToEvaluate.substring(lenStr-5, lenStr-4);
        String resultName = paramTemp.classifierName + fix;
        titleOutput.append(resultName);
        titleOutput.append(System.lineSeparator());
        for (int i = 0; i < domainsToEvaluate.size(); ++i) {
            String domain = domainsToEvaluate.get(i);
            FileReaderAndWriter.writeFile(cmdOption.outputSentimentClassificationAccuracy
                    + "/" + resultName + "/" + domain + "_Acc.txt", titleOutput.toString());
//            FileReaderAndWriter.writeFile(cmdOption.outputSentimentClassificationF1Score
//                    + "/" + resultName + "/" + domain + "_F1BothClasses.txt", titleOutput.toString());
//            FileReaderAndWriter.writeFile(cmdOption.outputSentimentClassificationF1Score
//                    + "/" + resultName + "/" + domain + "_F1Positive.txt", titleOutput.toString());
            FileReaderAndWriter.writeFile(cmdOption.outputSentimentClassificationF1Score
                    + "/" + resultName + "/" + domain + "_F1Negative.txt", titleOutput.toString());
        }

        for (int domain_id = 0; domain_id < domainsToEvaluate.size(); ++domain_id) {
            String targetDomain = domainsToEvaluate.get(domain_id);
//            if (!newDomain.equals("Baby")) {
//                continue;
//            }

            // TODO: main body, from here
            Documents trainingDocsFromTargetOrPast = new Documents();
            trainingDocsFromTargetOrPast.domain = targetDomain;
            Documents documents = new Documents();
            assert documentsOfAllDomains != null;
            for (int jj = 0; jj < documentsOfAllDomains.size(); ++jj) {
                if (Objects.equals(targetDomain, documentsOfAllDomains.get(jj).domain)) {
                    documents = documentsOfAllDomains.get(jj).getDeepClone();
                    break;
                }
            }

            // classifier parameters
            ClassifierParameters param = new ClassifierParameters(documents, cmdOption);
            int k = 0;
            param.K = k;

            // read training documents from target domain
            String trainingDocsFile = cmdOption.intermediateTrainingDocsDir + targetDomain + ".txt";
            Documents trainingDocs = Documents.readDocuments(targetDomain, trainingDocsFile);
            trainingDocsFromTargetOrPast.addDocuments(trainingDocs);
            // read testing documents
            String testingDocsFile = cmdOption.intermediateTestingDocsDir + targetDomain + ".txt";
            Documents testingDocs = Documents.readDocuments(targetDomain, testingDocsFile);

            double[][] classificationProInEachClass = new double[testingDocs.size()][2];
            double[][] classificationPro = getClassificationResult(trainingDocs, testingDocs, param);
            for (int docID = 0; docID < testingDocs.size(); ++docID) {
                for (int classID = 0; classID < 2; ++classID) {
                    classificationProInEachClass[docID][classID] += classificationPro[docID][classID];
                }
            }

            // read new training documents for this target domain
            for (int j = 0; j < domainsToEvaluate.size(); ++j) {
                if (domain_id == j) {
                    continue;
                }
                String addDomain = domainsToEvaluate.get(j);
                String newTrainingDocsFile = cmdOption.intermediateTrainingDocsDir + addDomain + ".txt";
                Documents newTrainingDocs = Documents.readDocuments(addDomain, newTrainingDocsFile);
                trainingDocsFromTargetOrPast.addDocuments(newTrainingDocs);
                classificationPro = getClassificationResult(newTrainingDocs, testingDocs, param);
                for (int docID = 0; docID < testingDocs.size(); ++docID) {
                    for (int classID = 0; classID < 2; ++classID) {
                        classificationProInEachClass[docID][classID] += classificationPro[docID][classID];
                    }
                }
            }

            for (int docID = 0; docID < testingDocs.size(); ++docID) {
                for (int classID = 0; classID < 2; ++classID) {
                    classificationProInEachClass[docID][classID] =
                            classificationProInEachClass[docID][classID] / domainsToEvaluate.size();
                }
            }

            // get the best category with highest probability
            double[] predict = new double[testingDocs.size()];
            String[] mCategories = { "+1", "-1" };
            int docID = 0;
            for (Document testingDoc : testingDocs) {
                double maximumProb = -Double.MAX_VALUE;
                int maximumIndex = -1;
                for (int i = 0; i < classificationProInEachClass[docID].length; ++i) {
                    if (maximumProb < classificationProInEachClass[docID][i]) {
                        maximumProb = classificationProInEachClass[docID][i];
                        maximumIndex = i;
                    }
                }
                testingDoc.predict = mCategories[maximumIndex];
                docID += 1;
            }
            ClassificationEvaluation evaluationValue = new ClassificationEvaluation(
                    testingDocs.getLabels(), testingDocs.getPredicts(), param.domain);

            Map<String, ClassificationEvaluation> mpDomainToClassificationEvaluation = new TreeMap<String, ClassificationEvaluation>();
            mpDomainToClassificationEvaluation.put(targetDomain, evaluationValue);

            System.out.println("Done for target domain " + targetDomain);

            /** print results to files */
            // Accuracy.
            StringBuilder sbOutput = new StringBuilder();
            double[] accuracies = new double[mpDomainToClassificationEvaluation.size()];
            int nfold = 0;
            for (Map.Entry<String, ClassificationEvaluation> entry : mpDomainToClassificationEvaluation
                    .entrySet()) {
                ClassificationEvaluation evaluation = entry.getValue();
                String domainToEvaluate = evaluation.domain;
                sbOutput.append(evaluation.accuracy);
                sbOutput.append(System.lineSeparator()); // line separator (i.e., '\n')
                accuracies[nfold++] = evaluation.accuracy;
                FileReaderAndWriter.addWriteFile(cmdOption.outputSentimentClassificationAccuracy
                        + "/" + resultName + "/" + domainToEvaluate + "_Acc.txt", sbOutput.toString());
            }

//            // F1-score in both classes.
//            sbOutput = new StringBuilder();
            double[] f1Scores = new double[mpDomainToClassificationEvaluation.size()];
//            nfold = 0;
//            for (Map.Entry<String, ClassificationEvaluation> entry : mpDomainToClassificationEvaluation
//                    .entrySet()) {
//                ClassificationEvaluation evaluation = entry.getValue();
//                String domainToEvaluate = evaluation.domain;
//                sbOutput.append(evaluation.f1scoreBothClasses);
//                sbOutput.append(System.lineSeparator());
//                f1Scores[nfold++] = evaluation.f1scoreBothClasses;
//                FileReaderAndWriter.addWriteFile(cmdOption.outputSentimentClassificationF1Score
//                        + "/" + resultName + "/" + domainToEvaluate + "_F1BothClasses.txt", sbOutput.toString());
//            }
//
//            // F1-score in the positive class.
//            sbOutput = new StringBuilder();
//            f1Scores = new double[mpDomainToClassificationEvaluation.size()];
//            nfold = 0;
//            for (Map.Entry<String, ClassificationEvaluation> entry : mpDomainToClassificationEvaluation
//                    .entrySet()) {
//                ClassificationEvaluation evaluation = entry.getValue();
//                String domainToEvaluate = evaluation.domain;
//                sbOutput.append(evaluation.f1score);
//                sbOutput.append(System.lineSeparator());
//                f1Scores[nfold++] = evaluation.f1score;
//                FileReaderAndWriter.addWriteFile(cmdOption.outputSentimentClassificationF1Score
//                        + "/" + "NaiveBayesWithAddingPastTrainingData" + "/" + domainToEvaluate + "_F1Positive.txt", sbOutput.toString());
//            }

            // F1-score in the negative class.
            sbOutput = new StringBuilder();
            f1Scores = new double[mpDomainToClassificationEvaluation.size()];
            nfold = 0;
            for (Map.Entry<String, ClassificationEvaluation> entry : mpDomainToClassificationEvaluation
                    .entrySet()) {
                ClassificationEvaluation evaluation = entry.getValue();
                String domainToEvaluate = evaluation.domain;
                sbOutput.append(evaluation.f1scoreNegativeClass);
                sbOutput.append(System.lineSeparator());
                f1Scores[nfold++] = evaluation.f1scoreNegativeClass;
                FileReaderAndWriter.addWriteFile(cmdOption.outputSentimentClassificationF1Score
                        + "/" + resultName + "/" + domainToEvaluate + "_F1Negative.txt", sbOutput.toString());
            }
        }
        return null;
    }

    /**
     * According to the configuration, we read documents from different
     * directories.
     */
    private List<Documents> readDocuments() {
        InputReaderTask task = new InputReaderTask(cmdOption);
        switch (cmdOption.datasetName) {
            case "100P100NDomains":
                return task.readDocumentsListFrom100P100NDomains();
            case "Reuters10":
                return task.readDocumentsFromstock();
            case "20Newgroup":
                return task.read20Newsgroup();
            case "PangAndLeeMovieReviews":
                return task.readDocumentsFromPangAndLeeMovieReview();
            case "1KP1KNDomains":
                return task.readDocumentsListFrom1KP1KNDomains();
            case "1KReviewNaturalClassDistributionDomains":
                return task.readDocumentsListFrom1KReviewsNaturalClassDistributionDomains();
            case "DifferentProductsOfSameDomain":
                return task.readDocumentsFromDifferentProductsOfSameDomain();
            case "BalancedWithMostNegativeReviews":
                return task.readDocumentsFromBalancedWithMostNegativeReviews();
        }
        return null;
    }

    /**
     * The proposed Lifelong Sentiment Classification (no tricks to tune the optimal parameters)
     *
     * @return
     */
    public double[][] getClassificationResult(
            Documents trainingDocs, Documents testingDocs, ClassifierParameters param) {

        // Classification evaluation: (Cross validation results, i.e., 5-folds have 5-results)
        ClassificationEvaluationAccumulator evaluationAccumulator
                = new ClassificationEvaluationAccumulator();

        CmdOption cmdOption = new CmdOption();
        String domain = param.domain;

        // Feature generation.
        // mainly generate feature for naive bayes, i.e., item "featuresForNaiveBayes"
        TopicModel topicModelForThisDomain = null;
        FeatureGenerator featureGenerator = new FeatureGenerator(param);
        featureGenerator
                .generateAndAssignFeaturesToTrainingAndTestingDocuments(
                        trainingDocs, testingDocs, topicModelForThisDomain);

        // Feature selection.
        // All selected features are covered by all documents of one domain.
        // The selected features are used to verify the validity of input feature (i.e., input word)
        // That is to say,
        // selected features are real valid features among target domain.
        FeatureSelection featureSelection = FeatureSelection
                .selectFeatureSelection(trainingDocs, param);

        // Print out top features.
//            featureSelection
//                    .printSelectedFeaturesToFile(param.outputTopFeaturesFilePath
//                            + param.domain + ".txt");

        // Build the classifier.
        NaiveBayes nbClassifier = new NaiveBayes(featureSelection, param);
        nbClassifier.train(trainingDocs);
        double[][] classificationProbability = nbClassifier.classificationPro(testingDocs);
        return classificationProbability;
    }

}
