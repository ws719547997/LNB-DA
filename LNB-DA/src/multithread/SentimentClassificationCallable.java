package multithread;

import classificationevaluation.ClassificationEvaluation;
import classificationevaluation.ClassificationEvaluationAccumulator;
import classifier.*;
import main.CmdOption;
import feature.Feature;
import feature.FeatureGenerator;
import feature.FeatureSelection;
import nlp.Document;
import nlp.Documents;
import task.TrainingCrossValidationToFindOptimalParametersTask;
import topicmodel.ModelLoader;
import topicmodel.TopicModel;
import utility.*;

import java.io.File;
import java.util.*;
import java.util.concurrent.Callable;

public class SentimentClassificationCallable implements
        Callable<ClassificationEvaluation> {
    private Documents documents = null;
    private Documents trainingDocuments = null;
    private Documents testingDocuments = null;
    private List<Documents> documentsOfOtherDomains = null;
    private ClassifierParameters param = null;
    private ClassificationKnowledge knowledge = null;
    private Map<String, ClassificationKnowledge> mPastknowledge = new HashMap<String, ClassificationKnowledge>();

    /**
     * Only target documents, no knowledge
     */
    public SentimentClassificationCallable(Documents documents2,
                                           ClassifierParameters param2) {
        documents = documents2;
        documentsOfOtherDomains = null;
        param = param2;
    }

    /**
     * Target documents, source documents, no knowledge
     */
    public SentimentClassificationCallable(Documents documents2,
                                           List<Documents> documentsOfOtherDomains2,
                                           ClassifierParameters param2) {
        documents = documents2;
        documentsOfOtherDomains = documentsOfOtherDomains2;
        param = param2;
    }

    /**
     * Target documents, knowledge
     */
    public SentimentClassificationCallable(Documents documents2,
                                           ClassificationKnowledge knowledge2, ClassifierParameters param2) {
        documents = documents2;
        documentsOfOtherDomains = null;
        knowledge = knowledge2;
        param = param2;
    }

    /**
     * Target documents, source documents, knowledge
     */
    public SentimentClassificationCallable(Documents documents2, List<Documents> documentsOfOtherDomains2,
                                           ClassificationKnowledge knowledge2, ClassifierParameters param2) {
        this.documents = documents2;
        this.documentsOfOtherDomains = documentsOfOtherDomains2;
        this.knowledge = knowledge2;
        this.param = param2;
    }

    /**
     * Target documents, pastKnowledgeList
     */
    public SentimentClassificationCallable(Documents documents2,
                                           Map<String, ClassificationKnowledge> mPastknowledge2,
                                           ClassifierParameters param2) {
        this.documents = documents2;
        this.mPastknowledge = mPastknowledge2;
        this.param = param2;
    }

    /**
     * Training documents, testing documents
     */
    public SentimentClassificationCallable(Documents trainingDocs2, Documents testingDocs2,
                                           ClassifierParameters param2) {
        this.trainingDocuments = trainingDocs2;
        this.testingDocuments = testingDocs2;
        this.param = param2;
    }


    /**
     * TODO: Multi-thread entrance...
     * Run the topic model or the proposed method in a domain and print it into the disk.
     */
    @Override
    public ClassificationEvaluation call() throws Exception {
        try {
            System.out.println("\"" + param.domain + "\" <"
                    + param.classifierName + "_" + param.K + "> Starts...");

            // if useTopicModelFeatures == true, run... (Currently, false)
            TopicModel topicModelForThisDomain = null;
            if (param.useTopicModelFeatures) {
                topicModelForThisDomain = readTopicModelForThisDomain(
                        param.domain, param.topicModelNameForFeatureGeneration,
                        param.topicModelSettingNameForFeatureGeneration,
                        param.outputTopicModelMultiDomainFilepath);
            }

            // Choose one: TODO
            // getClassificationEvaluation_good
            // getClassificationEvaluation (default)
//            ClassificationEvaluation evaluation = getClassificationEvaluation(
//                    documents, documentsOfOtherDomains,
//                    topicModelForThisDomain, knowledge, param);
            if (Objects.equals(param.classifierName, "NaiveBayes_Sequence_GoBack")) {
                ClassificationEvaluation evaluation = getClassificationEvaluation(documents,
                        topicModelForThisDomain, mPastknowledge, param);
                System.out.println("\"" + param.domain + "\" <"
                        + param.classifierName + ": " + evaluation.accuracy + " "
                        + evaluation.f1scoreBothClasses + "> Ends...");
                return evaluation; // which is "Future<ClassificationEvaluation> future" will get.
            } else if (Objects.equals(param.classifierName, "NaiveBayes_AddPastDomain")) {
                ClassificationEvaluation evaluation = getClassificationEvaluation(trainingDocuments,
                        topicModelForThisDomain, testingDocuments, param);
                System.out.println("\"" + param.domain + "\" <"
                        + param.classifierName + ": " + evaluation.accuracy + " "
                        + evaluation.f1scoreBothClasses + "> Ends...");
                return evaluation; // which is "Future<ClassificationEvaluation> future" will get.
            } else if ((Objects.equals(param.classifierName, "LibSVM"))
                    || (Objects.equals(param.classifierName, "LibLinear"))) {
                ClassificationEvaluation evaluation = getClassificationEvaluation(trainingDocuments,
                        topicModelForThisDomain, testingDocuments, param);
                System.out.println("\"" + param.domain + "\" <"
                        + param.classifierName + ": " + evaluation.accuracy + " "
                        + evaluation.f1scoreBothClasses + "> Ends...");
                return evaluation; // which is "Future<ClassificationEvaluation> future" will get.
            } else {
                ClassificationEvaluation evaluation = getClassificationEvaluation(documents, documentsOfOtherDomains,
                    topicModelForThisDomain, knowledge, param);
                System.out.println("\"" + param.domain + "\" <"
                        + param.classifierName + ": " + evaluation.accuracy + " "
                        + evaluation.f1scoreBothClasses + "> Ends...");
                return evaluation; // which is "Future<ClassificationEvaluation> future" will get.
            }
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        return null;
    }

    /**
     * Topic model
     *
     * @return
     */
    public TopicModel readTopicModelForThisDomain(String domain,
                                                  String topicModelNameForFeatureGeneration,
                                                  String topicModelSettingNameForFeatureGeneration,
                                                  String outputTopicModelMultiDomainFilepath) {
        // Read topic model if exists.
        TopicModel topicModelForThisDomain = null;
        String topicModelDirectory = outputTopicModelMultiDomainFilepath
                + topicModelSettingNameForFeatureGeneration + File.separator
                + "DomainModels" + File.separator + domain + File.separator;
        if (new File(topicModelDirectory).exists()) {
            ModelLoader modelLoader = new ModelLoader();
            topicModelForThisDomain = modelLoader.loadModel(
                    topicModelNameForFeatureGeneration, domain,
                    topicModelDirectory);
        }
        return topicModelForThisDomain;
    }

    /**
     * The proposed naive Bayes sequance with going back
     *
     * @return
     */
    public ClassificationEvaluation getClassificationEvaluation(Documents documents,
                                                                TopicModel topicModelForThisDomain,
                                                                Map<String, ClassificationKnowledge> pastKnowledgeList,
                                                                ClassifierParameters param) {
        // Classification evaluation: (Cross validation results, i.e., 5-folds have 5-results)
        ClassificationEvaluationAccumulator evaluationAccumulator
                = new ClassificationEvaluationAccumulator();

        // Classification begins ...
        CmdOption cmdOption = new CmdOption();
        String domain = param.domain;

        // read training documents
        String trainingDocsFile = cmdOption.intermediateTrainingDocsDir + domain + ".txt";
        Documents trainingDocs = Documents.readDocuments(domain, trainingDocsFile);
        // read testing documents
        String testingDocsFile = cmdOption.intermediateTestingDocsDir + domain + ".txt";
        Documents testingDocs = Documents.readDocuments(domain, testingDocsFile);

        // Feature generation.
        // mainly generate feature for naive bayes, i.e., item "featuresForNaiveBayes"
        FeatureGenerator featureGenerator = new FeatureGenerator(param);
        featureGenerator
                .generateAndAssignFeaturesToTrainingAndTestingDocuments(trainingDocs, testingDocs, topicModelForThisDomain);

        // Feature selection.
        // All selected features are covered by all documents of one domain.
        // The selected features are used to verify the validity of input feature (i.e., input word)
        // That is to say,
        // selected features are real valid features among target domain.
        FeatureSelection featureSelection = FeatureSelection
                .selectFeatureSelection(trainingDocs, param);

        // Build the classifier. // param.classifierName = "NaiveBayes_Sequence_GoBack";
        BaseClassifier classifier = BaseClassifier.selectGobackClassifier(
                param.classifierName, featureSelection, pastKnowledgeList, param);
        assert classifier != null; // in order to avoid classifier is null
        classifier.train(trainingDocs);
        ClassificationEvaluation evaluation = classifier.test(testingDocs);

        evaluationAccumulator.addClassificationEvaluation(evaluation);

        return evaluationAccumulator.getAverageClassificationEvaluation();
    }


    /**
     * Baseline, Naive Bayes with adding past domain training
     *
     * @return
     */
    public ClassificationEvaluation getClassificationEvaluation(Documents trainingDocs,
                                                                TopicModel topicModelForThisDomain,
                                                                Documents testingDocs,
                                                                ClassifierParameters param) {
        // Assign directory for SVM Light.
        int folderIndex = 0;
        param.svmLightCVFoldDirectory = param.svmLightRootDirectory
                + param.domain + File.separator + "CV" + folderIndex
                + File.separator;
        // Classification evaluation
        ClassificationEvaluationAccumulator evaluationAccumulator
                = new ClassificationEvaluationAccumulator();

        // Classification begins ...

        // Feature generation.
        // mainly generate feature for naive bayes, i.e., item "featuresForNaiveBayes"
        FeatureGenerator featureGenerator = new FeatureGenerator(param);
        featureGenerator
                .generateAndAssignFeaturesToTrainingAndTestingDocuments(trainingDocs, testingDocs, topicModelForThisDomain);

        // Feature selection.
        // All selected features are covered by all documents of one domain.
        // The selected features are used to verify the validity of input feature (i.e., input word)
        // That is to say,
        // selected features are real valid features among target domain.
        FeatureSelection featureSelection = FeatureSelection
                .selectFeatureSelection(trainingDocs, param);

        // Build the classifier. // param.classifierName = "NaiveBayes_Sequence_GoBack";
        Map<String, ClassificationKnowledge> pastKnowledgeList = new HashMap<>();
        BaseClassifier classifier = BaseClassifier.selectGobackClassifier(
                param.classifierName, featureSelection, pastKnowledgeList, param);
        assert classifier != null; // in order to avoid classifier is null
        classifier.train(trainingDocs);
        ClassificationEvaluation evaluation = classifier.test(testingDocs);

        evaluationAccumulator.addClassificationEvaluation(evaluation);
        return evaluationAccumulator.getAverageClassificationEvaluation();
    }

    /**
     * The proposed Lifelong Sentiment Classification (no tricks to tune the optimal parameters)
     *
     * @return
     */
    public ClassificationEvaluation getClassificationEvaluation(
            Documents documents, List<Documents> documentsOfOtherDomains,
            TopicModel topicModelForThisDomain,
            ClassificationKnowledge knowledge, ClassifierParameters param) {

        // Classification evaluation: (Cross validation results, i.e., 5-folds have 5-results)
        ClassificationEvaluationAccumulator evaluationAccumulator
                = new ClassificationEvaluationAccumulator();

        if (param.lifelongSequenceSwitch) {
            CmdOption cmdOption = new CmdOption();
            String domain = param.domain;
            // read training documents
            String trainingDocsFile = cmdOption.intermediateTrainingDocsDir + domain + ".txt";
            Documents trainingDocs = Documents.readDocuments(domain, trainingDocsFile);
            // read testing documents
            String testingDocsFile = cmdOption.intermediateTestingDocsDir + domain + ".txt";
            Documents testingDocs = Documents.readDocuments(domain, testingDocsFile);

            // Feature generation.
            // mainly generate feature for naive bayes, i.e., item "featuresForNaiveBayes"
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
            BaseClassifier classifier = BaseClassifier.selectNewClassifier(
                    param.classifierName, featureSelection, knowledge,
                    param);
            assert classifier != null; // in order to avoid classifier is null
            classifier.train(trainingDocs);

            ClassificationEvaluation evaluation = new ClassificationEvaluation();
            evaluation = classifier.test(testingDocs);
            evaluationAccumulator.addClassificationEvaluation(evaluation);

            return evaluationAccumulator.getAverageClassificationEvaluation();
        }
        // Cross validation.
        CrossValidationOperatorMaintainingLabelDistribution cvo
                = new CrossValidationOperatorMaintainingLabelDistribution(documents, param.noOfCrossValidationFolders);

        // Classification begins ...
        for (int folderIndex = 0; folderIndex < param.noOfCrossValidationFolders; ++folderIndex) {
            // System.out.println("Folder No " + folderIndex);
            // Assign directory for SVM Light.
            param.svmLightCVFoldDirectory = param.svmLightRootDirectory
                    + param.domain + File.separator + "CV" + folderIndex
                    + File.separator;

            // get training and testing documents
            Pair<Documents, Documents> pair = cvo.getTrainingAndTestingDocuments(folderIndex,
                    param.noOfCrossValidationFolders);

            // 1. get training documents
            // add target domain, if true
            Documents trainingDocs = new Documents();
            if (param.includeTargetDomainLabeledDataForTraining) {
                trainingDocs.addDocuments(pair.t); // for debugging over-fitting, use pair.u if fold is 1.
            }

            // also add source domains to train the model, if true
            if (param.includeSourceDomainsLabeledDataForTraining
                    && documentsOfOtherDomains != null) {
                for (Documents documentsForOneDomain : documentsOfOtherDomains) {
                    trainingDocs.addDocuments(documentsForOneDomain);
                }
            }
            // Only consider the case when the negative is more than positive. i.e., balance the data
            if (param.trainingNegativeVSPositiveRatio > 0) {
                trainingDocs
                        .makeBinaryClassesEven(param.trainingNegativeVSPositiveRatio);
            }
            // 2. get testing documents
            Documents testingDocs = pair.u; // // for debugging over-fitting, use pair.u if fold is not 1.

            // ratio of the number of POS and NEG in Training and testing
//            System.out.println("Training ratio: "
//                    + trainingDocs.getNoOfPositiveLabels() + " : " + trainingDocs.getNoOfNegativeLabels());
//            System.out.println("Testing ratio: "
//                    + testingDocs.getNoOfPositiveLabels() + " : " + testingDocs.getNoOfNegativeLabels());

            // Feature generation.
            // mainly generate feature for naive bayes, i.e., item "featuresForNaiveBayes"
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
            BaseClassifier classifier = BaseClassifier.selectNewClassifier(
                    param.classifierName, featureSelection, knowledge,
                    param);
            assert classifier != null; // in order to avoid classifier is null
            classifier.train(trainingDocs);

            ClassificationEvaluation evaluation = new ClassificationEvaluation();
            if (param.discardUnseenWords) {
                System.out.println("Discard unseen words...");

                // discard unseen words, i.e., no response for unseen words.
                evaluation = ((NaiveBayes_SGD_Lifelong) classifier)
                        .test_with_only_traing_vocab(testingDocs, trainingDocs);
            } else {
                // ACL2015 test
                evaluation = classifier.test(testingDocs);
            }

            // Debugging ...
            if (param.misclassifiedDocumentsFilePath != null) {
                String misclassifiedDocumentsForOneCVFolderForOneDomainFilePath = param.misclassifiedDocumentsFilePath
                        + param.classifierName
                        + File.separator
                        + param.domain
                        + File.separator + "CV" + folderIndex + ".txt";
//                printMisclassifiedDocuments(testingDocs,
//                        misclassifiedDocumentsForOneCVFolderForOneDomainFilePath);
                classifier.printMisclassifiedDocuments(testingDocs,
                        misclassifiedDocumentsForOneCVFolderForOneDomainFilePath);
            }
            if (param.classificationDetailsFilePath != null) {
                String classificationDetailsForOneCVFolderFilePath = param.classificationDetailsFilePath
                        + param.classifierName
                        + File.separator
                        + param.domain
                        + File.separator + "CV" + folderIndex + ".txt";
                printClassificationDetails(testingDocs, classifier,
                        classificationDetailsForOneCVFolderFilePath);
            }

            evaluationAccumulator.addClassificationEvaluation(evaluation);
        }
        return evaluationAccumulator.getAverageClassificationEvaluation();
    }

    /**
     * For testing documents
     * print unseen words which not emerging in training data
     * print documents, true label, and predict label to files
     */
    private void printUnseenWords(Map<String, Integer> mpUnseenWords,
                                  String unseenWordsForOneCVFolderForOneDomainFilePath,
                                  int sizeOfSourceKnowledge, int sizeOfTargetKnowledge,
                                  int sizeOfSourceAndTargetKnowledge ) {
        StringBuilder sbOutput = new StringBuilder();
        sbOutput.append("Source Knowledge Size: " + String.valueOf(sizeOfSourceKnowledge));
        sbOutput.append(System.lineSeparator());
        sbOutput.append("Target Knowledge Size: " + String.valueOf(sizeOfTargetKnowledge));
        sbOutput.append(System.lineSeparator());
        sbOutput.append("Source and Target Knowledge Size: " + String.valueOf(sizeOfSourceAndTargetKnowledge));
        sbOutput.append(System.lineSeparator());
        sbOutput.append("Unseen words are following,");
        sbOutput.append(System.lineSeparator());

        for (String unSeenWord : mpUnseenWords.keySet()) {
            if (!knowledge.wordCountInPerClass.containsKey(unSeenWord)) {
                sbOutput.append(unSeenWord + "\t::also not in knowledge base");
            } else {
                sbOutput.append(unSeenWord);
            }
            sbOutput.append(System.lineSeparator());
        }
        FileReaderAndWriter.writeFile(
                unseenWordsForOneCVFolderForOneDomainFilePath,
                sbOutput.toString());
    }

    /**
     * For testing documents
     * print documents, true label, and predict label to files
     */
    private void printMisclassifiedDocuments(Documents testingDocs,
                                             String misclassifiedDocumentsForOneCVFolderForOneDomainFilePath) {
        StringBuilder sbOutput = new StringBuilder();
        sbOutput.append("Document\tLabel\tPredict");
        sbOutput.append(System.lineSeparator());
        for (Document testingDoc : testingDocs) {
            if (!testingDoc.label.equals(testingDoc.predict)) {
                // Misclassified Document.
                sbOutput.append(testingDoc.text + "\t" + testingDoc.label
                        + "\t" + testingDoc.predict);
                sbOutput.append(System.lineSeparator());
            }
        }
        FileReaderAndWriter.writeFile(
                misclassifiedDocumentsForOneCVFolderForOneDomainFilePath,
                sbOutput.toString());
    }

    /**
     * For testing documents (each single domain)
     * print No., documents, true label, predict label, correct or wrong
     * and print P(w|+) and P(w|-)
     */
    private void printClassificationDetails(Documents testingDocs,
                                            BaseClassifier classifier,
                                            String classificationDetailsForOneCVFolderFilePath) {
        StringBuilder sbOutput = new StringBuilder();
        sbOutput.append("No.\tDocument\tLabel\tPredict\tCorrectOrWrong");
        sbOutput.append(System.lineSeparator());
        int index = 0;
        for (Document testingDoc : testingDocs) {
            if (testingDoc.label.equals(testingDoc.predict)) {
                sbOutput.append((index++) + "\t" + testingDoc.text + "\t"
                        + testingDoc.label + "\t" + testingDoc.predict
                        + "\tCorrect");
            } else {
                sbOutput.append((index++) + "\t" + testingDoc.text + "\t"
                        + testingDoc.label + "\t" + testingDoc.predict
                        + "\tWrong");
            }
            sbOutput.append(System.lineSeparator());

            List<ItemWithValue> featuresWithRatios = classifier
                    .getFeaturesByRatio(testingDoc);
            if (featuresWithRatios == null) {
                continue;
            }
            if (testingDoc.isPositive()) {
                // if testingDoc is positive, descending order
                Collections
                        .sort(featuresWithRatios);
            } else {
                // ascending order
                Collections.sort(featuresWithRatios, Collections.reverseOrder());
            }
            for (ItemWithValue iwv : featuresWithRatios) {
                String featureStr = iwv.getItem().toString();
                Double ratio = iwv.getValue();
                double[] countsOfClasses = classifier
                        .getCountsOfClasses(featureStr);
                sbOutput.append(featureStr + "\t" + ratio + "\t"
                        + countsOfClasses[0] + ":" + countsOfClasses[1]);
                sbOutput.append(System.lineSeparator());
            }
        }
        FileReaderAndWriter.writeFile(
                classificationDetailsForOneCVFolderFilePath,
                sbOutput.toString());
    }

}