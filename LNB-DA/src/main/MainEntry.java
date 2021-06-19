package main;

/** command line arguments parser (Enhancing the parsing)*/
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;

import task.*; // import package task, i.e., import all the classes in task
import classifier.ClassifierParameters;
import task.SVM_Sequence;
import utility.ExceptionUtility;

public class MainEntry {
    public static void main(String[] args) {
        // start to parse the command line parameters
        CmdOption cmdOption = new CmdOption();
        CmdLineParser parser = new CmdLineParser(cmdOption);
        ClassifierParameters parm = new ClassifierParameters();

        try {
            /** try...catch: handling of exceptions in the block of try*/
            long startTime = System.currentTimeMillis();

            // Parse the arguments.
            parser.parseArgument(args);

            /**
             * user own codes
             */

            cmdOption.ngram = 1;
            cmdOption.attantionMode = "att";
            //"att" "att_max" "att_percent"
            cmdOption.vtMode = "none"; // add
            cmdOption.vkbMode = "none"; //ds
            cmdOption.gammaThreshold = 6;
            cmdOption.positiveRatioThreshold = 5;
            cmdOption.domainNumLavege = 13;

            switch (parm.classifierName) {
                case "NaiveBayes_SGD_Lifelong":  // 这是goback
                    if (parm.lifelongSequenceSwitch) {
                        // NaiveBayes_SGD_Lifelong_Sequence, baseline in terms of going back
                        // We use the classifier built when the past domain was the new domain at that time)
                        LifelongSGDNaiveBayesSequenceLearning
                                lifelongSequence = new LifelongSGDNaiveBayesSequenceLearning(cmdOption);
                        lifelongSequence.run();
                    } else {
                        // NaiveBayes_SGD_Lifelong, Zhiyuan's method, improving future tasks
                        LifelongWithKnowledgeTrainingAndTestingFoldCrossValidationTask
                                lifelongTask = new LifelongWithKnowledgeTrainingAndTestingFoldCrossValidationTask(cmdOption);
                        lifelongTask.run();
                    }
                    break;
                case "LibSVM":
                    // SVM-T, SVM-S, SVM-ST in EMNLP-2018
                    SVM_Sequence LibSVM_task = new SVM_Sequence(cmdOption);
                    LibSVM_task.run();
                    break;
                case "NaiveBayes_Sequence_GoBack":  // 训练集和测试集事先分开
                    // 2018ACL_LNB_ShortPaper
                    // Training data and testing data of each domain are separated before.
                    // LNB in EMNLP-2018: focus on the past learning
                    NaiveBayesSequenceLearningGoBack
                            naiveBayesSequenceGoBack = new NaiveBayesSequenceLearningGoBack(cmdOption);
                    naiveBayesSequenceGoBack.run();
                    break;
                case "NaiveBayes_AddPastDomain":  // 看起来这是baseline
                    // no accumulated knowledge
                    // Training data and testing data of each domain are separated before.
                    // NB-T, NB-S, NB-ST in the paper (EMNLP-2018)
                    NaiveBayesWithAddingPastTrainingData naiveBayesWithAdding =
                            new NaiveBayesWithAddingPastTrainingData(cmdOption);
                    naiveBayesWithAdding.run();
                    break;
                case "LifelongBagging":
                    // Xia et al., 2017, Distantly Supervised Lifelong Learning for Large-Scale Social Media Sentiment Analysis
                    // LLV in EMNLP-2018
                    // Both for future task or previous tasks are here, which need to change one place in LifelongBaging().
                    LifelongBagging lifelongBagging = new LifelongBagging(cmdOption);
                    lifelongBagging.run();
                    break;

                // Below: Training data and test data of each domain are not separated before.
                case "NaiveBayes":  //没分开
                    // multinomial Naive Bayes
                    // Training data and testing data of each domain are NOT separated before.
                    MergeLabeledDataFromMultipleDomainIncludingTargetDomainForTrainingCrossValidationTask
                            naiveBayesTask = new
                            MergeLabeledDataFromMultipleDomainIncludingTargetDomainForTrainingCrossValidationTask(cmdOption);
                    naiveBayesTask.run();
                    break;
                // below are not going back
                case "NaiveBayes_Sequence":
                    // focus on sequence learning, not go back
                    // training data from target domain and past domains
                    // Training data and testing data of each domain are NOT separated before.
                    NaiveBayesSequenceLearning naiveBayesSequence = new NaiveBayesSequenceLearning(cmdOption);
                    naiveBayesSequence.run();
                    break;
                case "KnowledgeableNB":
                    // using the accumulated knowledge from past domains, no SGD optimization
                    // Training data and testing data of each domain are NOT separated before.
                    // The setting is the same to Zhiyuan, and the result is comparable with Zhiyuan's result.
                    KnowledgeableNaiveBayes knowledgeableNB = new KnowledgeableNaiveBayes(cmdOption);
                    knowledgeableNB.run();
                    break;
                default:
                    ExceptionUtility
                            .throwAndCatchException("The classifier guagua " +
                                    "is not recognizable!");
                    break;
            }

            /************************** Single domain cross validation ****************************/
            // Reuters data.
            // One-vs-Rest.
//             cmdOption.datasetName = "Reuters10";
//             cmdOption.dataVsSetting = "One-vs-Rest";
//             // cmdOption.outputSentimentClassificationAccuracy =
//             // "..\\Data\\Output\\SentimentClassificaton\\Accuracy_"
//             // + cmdOption.dataVsSetting
//             // + "_"
//             // + cmdOption.datasetName + ".txt";
//             cmdOption.outputSentimentClassificationF1Score =
//             "..\\Data\\Output\\SentimentClassificaton\\F1Score_"
//             + cmdOption.dataVsSetting
//             + "_"
//             + cmdOption.datasetName
//             + ".txt";
//             cmdOption.smoothingPriorForFeatureInNaiveBayes = 0.1;
//             SingleDomainTrainingAndTestingFoldCrossValidationTask task1 = new
//             SingleDomainTrainingAndTestingFoldCrossValidationTask(
//             cmdOption);
//             task1.run();

            /************************** Merge labeled data from other domains domain simply for training ****************************/
//             MergeLabeledDataFromMultipleDomainForTrainingCrossValidationTask
//             task = new
//             MergeLabeledDataFromMultipleDomainForTrainingCrossValidationTask(
//             cmdOption);
//             task.run();

            // PariwiseDomainTransferLearningTask task = new
            // PariwiseDomainTransferLearningTask();
            // task.run();

            // SequentialDomainTransferLearningTask task = new
            // SequentialDomainTransferLearningTask();
            // task.run();

            // ElectronicsAndNonElectronicsTransferLearningTask task = new
            // ElectronicsAndNonElectronicsTransferLearningTask();
            // task.run();

//             SentimentClassificationDifferentDomainsTask task = new
//             SentimentClassificationDifferentDomainsTask();
//             task.run();

            // SentimentClassificationSameDomainDifferentProductsTask task = new
            // SentimentClassificationSameDomainDifferentProductsTask();
            // task.run();

            // TopicModelMultiDomainRunningTask task = new
            // TopicModelMultiDomainRunningTask();
            // task.run(cmdOption);

            // SentimentWordsPolarityExtractionTask task = new
            // SentimentWordsPolarityExtractionTask();
            // task.run();

            System.out.println("Program Ends.");
            long endTime = System.currentTimeMillis();
            showRunningTime(endTime - startTime);
        } catch (CmdLineException cle) {
            // handling of error
            System.out.println("Command line error: " + cle.getMessage());
            showCommandLineHelp(parser);
            return;
        } catch (Exception e) {
            // handling of wrong arguments
            System.out.println("Error in program: " + e.getMessage());
            e.printStackTrace();
            return;
        }
    }

    private static void showCommandLineHelp(CmdLineParser parser) {
        System.out.println("java [options ...] [arguments...]");
        parser.printUsage(System.out);
    }

    private static void showRunningTime(long time) {
        System.out.println("Elapsed time: "
                + String.format("%.3f", (time) / 1000.0) + " seconds");
        System.out.println("Elapsed time: "
                + String.format("%.3f", (time) / 1000.0 / 60.0) + " minutes");
        System.out.println("Elapsed time: "
                + String.format("%.3f", (time) / 1000.0 / 3600.0) + " hours");
    }
}
