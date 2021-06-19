package task;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import main.CmdOption;
import multithread.SentimentClassificationThreadPool;
import nlp.Document;
import nlp.Documents;
import utility.ArraySumAndAverageAndMaxAndMin;
import utility.FileReaderAndWriter;
import classificationevaluation.ClassificationEvaluation;
import classifier.ClassifierParameters;

/**
 * call LibLinear form https://www.csie.ntu.edu.tw/~cjlin/liblinear/
 */
public class LibLinearSequence {
    public CmdOption cmdOption = null;
    private List<String> domainsToEvaluate = null;

    public LibLinearSequence(
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
        String resultName = paramTemp.classifierName + "_";
        if (paramTemp.includeTargetDomainLabeledDataForTraining
                && paramTemp.includeSourceDomainsLabeledDataForTraining) {
            resultName += "ST";
        } else if (paramTemp.includeTargetDomainLabeledDataForTraining) {
            resultName += "T";
        } else if (paramTemp.includeSourceDomainsLabeledDataForTraining) {
            resultName += "S";
        }
        titleOutput.append(resultName);
        titleOutput.append(System.lineSeparator());
        for (int i = 0; i < domainsToEvaluate.size(); ++i) {
            String domain = domainsToEvaluate.get(i);
            FileReaderAndWriter.writeFile(cmdOption.outputSentimentClassificationAccuracy
                    + "/" + resultName + "/" + domain + "_Acc.txt", titleOutput.toString());
            FileReaderAndWriter.writeFile(cmdOption.outputSentimentClassificationF1Score
                    + "/" + resultName + "/" + domain + "_F1BothClasses.txt", titleOutput.toString());
            FileReaderAndWriter.writeFile(cmdOption.outputSentimentClassificationF1Score
                    + "/" + resultName + "/" + domain + "_F1Positive.txt", titleOutput.toString());
            FileReaderAndWriter.writeFile(cmdOption.outputSentimentClassificationF1Score
                    + "/" + resultName + "/" + domain + "_F1Negative.txt", titleOutput.toString());
        }

        for (int domain_id = 0; domain_id < domainsToEvaluate.size(); ++domain_id) {
            String targetDomain = domainsToEvaluate.get(domain_id);
//            if (!targetDomain.equals("Baby")) {
//                continue;
//            }

            // TODO: main body, from here
            // cmdOption.nthreads: the number of maximum threads in multithreading
            SentimentClassificationThreadPool threadPool = new SentimentClassificationThreadPool(cmdOption.nthreads);
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

            // read training documents from target domain
            if (param.includeTargetDomainLabeledDataForTraining) {
                String trainingDocsFile = cmdOption.intermediateTrainingDocsDir + targetDomain + ".txt";
                Documents trainingDocs = Documents.readDocuments(targetDomain, trainingDocsFile);
                trainingDocsFromTargetOrPast.addDocuments(trainingDocs);

            }
            // read testing documents
            String testingDocsFile = cmdOption.intermediateTestingDocsDir + targetDomain + ".txt";
            Documents testingDocs = Documents.readDocuments(targetDomain, testingDocsFile);

            int k = 0;
            param.K = k;
            threadPool.addTask(trainingDocsFromTargetOrPast, testingDocs, param);

            threadPool.awaitTermination(); // Getting classification evaluation (results)
            Map<String, ClassificationEvaluation> mpDomainToClassificationEvaluation = threadPool.mpClassificationEvaluation;

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

            // F1-score in both classes.
            sbOutput = new StringBuilder();
            double[] f1Scores = new double[mpDomainToClassificationEvaluation.size()];
            nfold = 0;
            for (Map.Entry<String, ClassificationEvaluation> entry : mpDomainToClassificationEvaluation
                    .entrySet()) {
                ClassificationEvaluation evaluation = entry.getValue();
                String domainToEvaluate = evaluation.domain;
                sbOutput.append(evaluation.f1scoreBothClasses);
                sbOutput.append(System.lineSeparator());
                f1Scores[nfold++] = evaluation.f1scoreBothClasses;
                FileReaderAndWriter.addWriteFile(cmdOption.outputSentimentClassificationF1Score
                        + "/" + resultName + "/" + domainToEvaluate + "_F1BothClasses.txt", sbOutput.toString());
            }

            // F1-score in the positive class.
            sbOutput = new StringBuilder();
            f1Scores = new double[mpDomainToClassificationEvaluation.size()];
            nfold = 0;
            for (Map.Entry<String, ClassificationEvaluation> entry : mpDomainToClassificationEvaluation
                    .entrySet()) {
                ClassificationEvaluation evaluation = entry.getValue();
                String domainToEvaluate = evaluation.domain;
                sbOutput.append(evaluation.f1score);
                sbOutput.append(System.lineSeparator());
                f1Scores[nfold++] = evaluation.f1score;
                FileReaderAndWriter.addWriteFile(cmdOption.outputSentimentClassificationF1Score
                        + "/" + resultName + "/" + domainToEvaluate + "_F1Positive.txt", sbOutput.toString());
            }

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

            // add past domain training data sequentially
            for (int j = 0; j < domainsToEvaluate.size(); ++j) {
                if (domain_id == j) {
                    continue;
                }
                k += 1;

                // read training documents from past domain
                String addDomain = domainsToEvaluate.get(j);
                if (param.includeSourceDomainsLabeledDataForTraining) {
                    String trainingDocsFile = cmdOption.intermediateTrainingDocsDir + addDomain + ".txt";
                    Documents trainingDocs = Documents.readDocuments(addDomain, trainingDocsFile);
                    trainingDocsFromTargetOrPast.addDocuments(trainingDocs);
                }

                param.K = k;
                // new thread
                threadPool = new SentimentClassificationThreadPool(cmdOption.nthreads);
                threadPool.addTask(trainingDocsFromTargetOrPast, testingDocs, param);
                threadPool.awaitTermination(); // Getting classification evaluation (results)
                mpDomainToClassificationEvaluation = threadPool.mpClassificationEvaluation;

                /** print results to files */
                // Accuracy.
                sbOutput = new StringBuilder();
                accuracies = new double[mpDomainToClassificationEvaluation.size()];
                nfold = 0;
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

                // F1-score in both classes.
                sbOutput = new StringBuilder();
                f1Scores = new double[mpDomainToClassificationEvaluation.size()];
                nfold = 0;
                for (Map.Entry<String, ClassificationEvaluation> entry : mpDomainToClassificationEvaluation
                        .entrySet()) {
                    ClassificationEvaluation evaluation = entry.getValue();
                    String domainToEvaluate = evaluation.domain;
                    sbOutput.append(evaluation.f1scoreBothClasses);
                    sbOutput.append(System.lineSeparator());
                    f1Scores[nfold++] = evaluation.f1scoreBothClasses;
                    FileReaderAndWriter.addWriteFile(cmdOption.outputSentimentClassificationF1Score
                            + "/" + resultName + "/" + domainToEvaluate + "_F1BothClasses.txt", sbOutput.toString());
                }

                // F1-score in the positive class.
                sbOutput = new StringBuilder();
                f1Scores = new double[mpDomainToClassificationEvaluation.size()];
                nfold = 0;
                for (Map.Entry<String, ClassificationEvaluation> entry : mpDomainToClassificationEvaluation
                        .entrySet()) {
                    ClassificationEvaluation evaluation = entry.getValue();
                    String domainToEvaluate = evaluation.domain;
                    sbOutput.append(evaluation.f1score);
                    sbOutput.append(System.lineSeparator());
                    f1Scores[nfold++] = evaluation.f1score;
                    FileReaderAndWriter.addWriteFile(cmdOption.outputSentimentClassificationF1Score
                            + "/" + resultName + "/" + domainToEvaluate + "_F1Positive.txt", sbOutput.toString());
                }

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
        }
        return null;
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

}
