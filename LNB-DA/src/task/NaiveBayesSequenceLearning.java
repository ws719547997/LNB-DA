package task;

import classificationevaluation.ClassificationEvaluation;
import classifier.ClassificationKnowledge;
import classifier.ClassifierParameters;
import classifier.NaiveBayes;
import feature.FeatureGenerator;
import feature.FeatureIndexer;
import feature.FeatureSelection;
import main.CmdOption;
import multithread.SentimentClassificationThreadPool;
import nlp.Documents;
import utility.ArraySumAndAverageAndMaxAndMin;
import utility.FileReaderAndWriter;

import java.io.File;
import java.util.*;

/**
 * Tasks come sequentially.
 * Naive Bayes sequence learning focus on sequence learning, not focus on go back problem.
 * where knowledge is generated from all data (training and testing) of each domain.
 */
public class NaiveBayesSequenceLearning {
    public CmdOption cmdOption = null;
    private List<String> domainList = null;

    // construction method: the name is same to the class name
    public NaiveBayesSequenceLearning(CmdOption cmdOption2) {
        cmdOption = cmdOption2;

        if (cmdOption.inputListOfDomainsToEvaluate != null) {
            domainList = FileReaderAndWriter.readFileAllLines(cmdOption.inputListOfDomainsToEvaluate);
        }
    }

    public Map<String, ClassificationEvaluation> run() {
        List<Documents> documentsOfAllDomains = readDocuments();

//		// Print the documents with preprocessed content to the directory
//		 printListOfPreprocessedDocumentsToDirectory(documentsOfAllDomains,
//		 "../Data/Output/PreprocessedDocuments/");

        // TODO: debugging, print word information
        // generate and print knowledge for each domain
//         printWordInformationOfEachDomain(documentsOfAllDomains);

        ClassifierParameters paramTemp = new ClassifierParameters();
        StringBuilder titleOutput = new StringBuilder();
        String resultName = paramTemp.classifierName;
        titleOutput.append(resultName);
        titleOutput.append(System.lineSeparator());
        for (int i = 0; i < domainList.size(); ++i) {
            String domain = domainList.get(i);
            FileReaderAndWriter.writeFile(cmdOption.outputSentimentClassificationAccuracy
                    + "/" + resultName + "/" + domain + "_Acc.txt", titleOutput.toString());
            FileReaderAndWriter.writeFile(cmdOption.outputSentimentClassificationF1Score
                    + "/" + resultName + "/" + domain +  "_F1BothClasses.txt", titleOutput.toString());
            FileReaderAndWriter.writeFile(cmdOption.outputSentimentClassificationF1Score
                    + "/" + resultName + "/" + domain +  "_F1Positive.txt", titleOutput.toString());
            FileReaderAndWriter.writeFile(cmdOption.outputSentimentClassificationF1Score
                    + "/" + resultName + "/" + domain +  "_F1Negative.txt", titleOutput.toString());
        }

        // Start...
        for (int domain_id = 0; domain_id < domainList.size(); ++domain_id) {
            String targetDomain = domainList.get(domain_id);
            // get submitted documents including target domain (i.e., domain_id)
            List<Documents> submittedDocuments = new ArrayList<Documents>();

            Documents targetDocuments = new Documents();
            assert documentsOfAllDomains != null;
            for (int i = 0; i < documentsOfAllDomains.size(); ++i) {
                if (Objects.equals(targetDomain, documentsOfAllDomains.get(i).domain)) {
                    targetDocuments = documentsOfAllDomains.get(i).getDeepClone();
                    break;
                }
            }
            submittedDocuments.add(targetDocuments.getDeepClone());
            // cmdOption.nthreads: the number of maximum threads in multithreading
            SentimentClassificationThreadPool threadPool = new SentimentClassificationThreadPool(
                    cmdOption.nthreads);
            // classifier parameters
            ClassifierParameters param = new ClassifierParameters(targetDocuments, cmdOption);
            int k = 0;
            param.K = k;
            // Get the knowledge for this target domain.
            ClassificationKnowledge knowledgeForTargetDomain = new ClassificationKnowledge();
            threadPool.addTask(targetDocuments, knowledgeForTargetDomain, param);
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
                        + "/" + resultName + "/" + domainToEvaluate +  "_Acc.txt", sbOutput.toString());
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

            // add task one by one
            Documents addDocuments = new Documents();
            for (int j = 0; j < domainList.size(); ++j) {
                String addDomain = domainList.get(j);
                if (domain_id == j) {
                    continue;
                }
                k += 1;
                param.K = k;
                // cmdOption.nthreads: the number of maximum threads in multithreading
                threadPool = new SentimentClassificationThreadPool(cmdOption.nthreads);
                for (int jj = 0; jj < documentsOfAllDomains.size(); ++jj) {
                    if (Objects.equals(addDomain, documentsOfAllDomains.get(jj).domain)) {
                        addDocuments = documentsOfAllDomains.get(jj).getDeepClone();
                        break;
                    }
                }
                submittedDocuments.add(addDocuments.getDeepClone());
                // read or generate knowledge for each target domain
                Map<String, ClassificationKnowledge> mpDomainToKnowledge
                        = readOrGenerateClassificationKnowledgeForEachTargetDomain(submittedDocuments);
                knowledgeForTargetDomain = mpDomainToKnowledge.get(targetDomain).getDeepClone();
                threadPool.addTask(targetDocuments, knowledgeForTargetDomain, param);
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
                            + "/" + resultName + "/" + domainToEvaluate +  "_Acc.txt", sbOutput.toString());
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
        return null; // classification evaluation results
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
                return task.readReuters10domains();
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
     * read or generate classification knowledge for each target domain
     *
     * @param documentsOfAllDomains
     * @return classification knowledge
     */
    private Map<String, ClassificationKnowledge> readOrGenerateClassificationKnowledgeForEachTargetDomain(
            List<Documents> documentsOfAllDomains) {
        Map<String, ClassificationKnowledge> mpDomainToKnowledge = new HashMap<String, ClassificationKnowledge>();
        String knowledgeDirectory = ClassifierParameters.KnowledgeFromSourceDomainsDirectory;

        // read or generate classification knowledge for each target domain in turns
        for (int i = 0; i < 1; ++i) {
            Documents documentsOfTargetDomain = documentsOfAllDomains.get(i);
            String targetDomain = documentsOfTargetDomain.domain;
            if (domainList != null
                    && !domainList.contains(targetDomain)) {
                continue;
            }

//			String domainFilePath = knowledgeDirectory + targetDomain + ".txt";
//			if (new File(domainFilePath).exists()) {
//			// If the classification knowledge file already exists,
//			// -> load...
//			System.out.println("Loaded knowledge for target domain " + targetDomain);
//			ClassificationKnowledge knowledge = ClassificationKnowledge
//					.readClassificationProbabilitiesFromFile(domainFilePath);
//			mpDomainToKnowledge.put(targetDomain, knowledge);
//			} else {
            // generate...
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
                if (domainList != null
                        && !domainList.contains(documentsOfAllDomains.get(j).domain)) {
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
                if (domainList != null
                        && !domainList.contains(documentsOfAllDomains.get(j).domain)) {
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
//			}

                // TODO: closed now
                // print knowledge of each target domain to file
                // knowledge.printToFile(domainFilePath);

                mpDomainToKnowledge.put(targetDomain, knowledge);

                ++numberOfSourceDomains;
                if (numberOfSourceDomains >= cmdOption.numberOfMaximumSourceDomains) {
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

    /**
     * print word information of each domain into file (i.e., ../domain_name.txt)
     * where word information include,
     * "Feature", "#Domains(Pr(w|+)>Pr(w|-)):#Domains(Pr(w|+)<Pr(w|-))",
     * "Sum_Freq(+,w):Sum_Freq(-,w)", and "DomainName: Freq(+,w):Freq(-,w)"
     *
     * @param documentsOfAllDomains
     */
    private void printWordInformationOfEachDomain(List<Documents> documentsOfAllDomains) {
        Map<String, ClassificationKnowledge> mpDomainToKnowledge = new HashMap<String, ClassificationKnowledge>();
        String file_path = ClassifierParameters.wordInformationFilepath;

        FeatureIndexer featuresInAllDomains = new FeatureIndexer();

        // generate and print classification knowledge for each domain one by one
        for (int i = 0; i < documentsOfAllDomains.size(); ++i) {
            // take out domain_name and its own documents
            Documents documentsOfOneDomain = documentsOfAllDomains.get(i);
            String domain_name = documentsOfOneDomain.domain;

            // Extract classification knowledge: indexed by word (i.e., featured word)
            NaiveBayes nbClassifierForEachDomain = getKnowledgeBasedOnNBClassifier(documentsOfOneDomain);
            ClassificationKnowledge knowledgeForEachDomain = nbClassifierForEachDomain.knowledge;

            // take out all featured words from all domains
            for (String featureStr : knowledgeForEachDomain.wordCountInPerClass.keySet()) {
                featuresInAllDomains
                        .addFeatureStrIfNotExistStartingFrom0(featureStr);
            }

            // Domain-level knowledge: M_{+,w}^KB and M_{-,w}^KB
            for (int j = 0; j < documentsOfAllDomains.size(); ++j) {
                // Note: no need to do if (j==i) {continue} as we want to collect word information from all domains
                if (domainList != null
                        && !domainList.contains(documentsOfAllDomains.get(j).domain)) {
                    continue;
                }
                nbClassifierForEachDomain = getKnowledgeBasedOnNBClassifier(documentsOfAllDomains.get(j).getDeepClone());
                for (Map.Entry<String, double[]> entry : nbClassifierForEachDomain.knowledge.wordCountInPerClass
                        .entrySet()) {
                    String featureStr = entry.getKey();
                    double[] tokenCounts = entry.getValue();
                    double probOfFeatureGivenPositive = nbClassifierForEachDomain
                            .probTokenByIndexArray(0, tokenCounts);
                    double probOfFeatureGivenNegative = nbClassifierForEachDomain
                            .probTokenByIndexArray(1, tokenCounts);
                    if (!knowledgeForEachDomain.countDomainsInPerClass
                            .containsKey(featureStr)) {
                        knowledgeForEachDomain.countDomainsInPerClass.put(
                                featureStr, new double[2]);
                    }
                    if (probOfFeatureGivenPositive >= probOfFeatureGivenNegative) {
                        // Positive feature in this domain.
                        knowledgeForEachDomain.countDomainsInPerClass
                                .get(featureStr)[0]++;
                    } else {
                        // Negative feature in this domain.
                        knowledgeForEachDomain.countDomainsInPerClass
                                .get(featureStr)[1]++;
                    }
                }
            }

            // print knowledge (i.e., word information) of each domain to file
            knowledgeForEachDomain.printToFile(file_path + cmdOption.numberOfMaximumSourceDomains
                    + File.separator
                    + domain_name + ".txt");
            // add knowledge to Map
            mpDomainToKnowledge.put(domain_name, knowledgeForEachDomain);
        }

        // print all featured word information (i.e., featured word from all domains)
        String file_dir = file_path
                + cmdOption.numberOfMaximumSourceDomains
                + File.separator
                + "allFeaturedWordInformation_" + cmdOption.numberOfMaximumSourceDomains + ".txt";
        StringBuilder sbOutput = new StringBuilder();
        sbOutput.append("Feature" + "\t" + "#Domains(Pr(w|+)>Pr(w|-)):#Domains(Pr(w|+)<Pr(w|-))" + "\t"
                + "Sum_Freq(+,w):Sum_Freq(-,w)" + "\t"
                + "DomainName: Freq(+,w):Freq(-,w)");
        sbOutput.append(System.lineSeparator());
        for (int featureID = 0; featureID < featuresInAllDomains.mpFeatureIdToFeatureStr.size(); ++featureID) {
            String featureStr = featuresInAllDomains.getFeatureStrGivenFeatureId(featureID);
            sbOutput.append(featureStr + "\t");
            double[] wordInfo = new double[4];
            StringBuilder sbOutput2 = new StringBuilder();
            for (int domainID = 0; domainID < documentsOfAllDomains.size(); ++domainID) {
                Documents documents = documentsOfAllDomains.get(domainID).getDeepClone();
                String domain = documents.domain;
                if (domainList != null
                        && !domainList.contains(domain)) {
                    continue;
                }
                ClassificationKnowledge currentKnowledge = mpDomainToKnowledge.get(domain);
                Map<String, double[]> wordCount = currentKnowledge.wordCountInPerClass;
                if (wordCount.containsKey(featureStr)) {
                    sbOutput2.append("\t" + domain + " " + (int) wordCount.get(featureStr)[0]
                            + ":" + (int) wordCount.get(featureStr)[1]);
                    wordInfo[0] = currentKnowledge.countDomainsInPerClass.get(featureStr)[0];
                    wordInfo[1] = currentKnowledge.countDomainsInPerClass.get(featureStr)[1];
                    wordInfo[2] += wordCount.get(featureStr)[0];
                    wordInfo[3] += wordCount.get(featureStr)[1];
                }
            }
            sbOutput.append((int) wordInfo[0] + ":" + (int) wordInfo[1] + "\t");
            sbOutput.append((int) wordInfo[2] + ":" + (int) wordInfo[3]);
            sbOutput.append(sbOutput2.toString());
            sbOutput.append(System.lineSeparator());
        }
        // print information into file
        if (file_path != null) {
            FileReaderAndWriter.writeFile(file_dir, sbOutput.toString());
        }
    }

    /**
     * print word information into file, where word information include,
     * featureStr, wordCount(+,w), wordCount(-,w)
     *
     * @param documentsOfOneDomain
     * @param file_path
     */
    private void printWordInformationOfEachDomain(Documents documentsOfOneDomain,
                                                  String file_path) {
        NaiveBayes nbClassifierForEachDomain = getKnowledgeBasedOnNBClassifier(documentsOfOneDomain);
        ClassificationKnowledge knowledgeForEachDomain = nbClassifierForEachDomain.knowledge;

        String file_dir = file_path + documentsOfOneDomain.domain + ".txt";
        StringBuilder sbOutput = new StringBuilder();
        sbOutput.append("Feature: " + "\twordCount(+,w)" + "\twordCount(-,w)");
        sbOutput.append(System.lineSeparator());
        sbOutput.append("=============================================");
        sbOutput.append(System.lineSeparator());
        for (Map.Entry<String, double[]> entry : knowledgeForEachDomain.wordCountInPerClass
                .entrySet()) {
            String featureStr = entry.getKey();
            double[] tokenCounts = entry.getValue();
            sbOutput.append(featureStr + ": " + (int) tokenCounts[0] + "(+), " + (int) tokenCounts[1] + "(-)");
            sbOutput.append(System.lineSeparator());
        }
        // print information into file
        if (file_path != null) {
            FileReaderAndWriter.writeFile(file_dir, sbOutput.toString());
        }
    }

    /**
     * print preprocessed documents of each domain into file (i.e., ../domain_name.txt)
     *
     * @param documentsOfAllDomains
     * @param directory
     */
    private void printListOfPreprocessedDocumentsToDirectory(
            List<Documents> documentsOfAllDomains, String directory) {
        for (int i = 0; i < documentsOfAllDomains.size(); ++i) {
            Documents documents = documentsOfAllDomains.get(i).getDeepClone();
            String domain = documents.domain;
            if (domainList != null
                    && !domainList.contains(domain)) {
                continue;
            }
            documents.printToFileWithPreprocessedContent(directory
                    + File.separator + domain + ".txt");
        }
    }
}
