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
import utility.*;

import java.io.File;
import java.util.*;

/**
 * In this setting, the problem can be defined as follows:
 * <p>
 * Our tasks come sequentially, T1, T2, �, Tn, �.
 * When each new task Tn comes to the system, it is accompanied with its training data Dn.
 * After the classifier Cn for task Tn is learned from Dn (and possibly probability information saved from the previous tasks),
 * its data is forgotten and gone.
 * Only the probabilities of P(w|c) and P(c) for task Tn are saved, which are the classifier Cn.
 * That is, all the training data for T(n-k) are forgotten.
 * Our goal is to improve all classifiers Ti (I = 1, �, n).
 * <p>
 * Note that this is different from Zhiyuan�s setting, Zhiyuan�s system only improves Cn, but not Cn-k (k = 1, �., n-1).
 */
public class NaiveBayesSequenceLearningGoBack {
    public CmdOption cmdOption = null;
    private List<String> domainListOrg = null; // domain name list
    private List<String> domainList = null; // domain name list

    // construction method: the name is same to the class name
    public NaiveBayesSequenceLearningGoBack(CmdOption cmdOption2) {
        cmdOption = cmdOption2;

        // get evaluated domain name list
        if (cmdOption.inputListOfDomainsToEvaluate != null) {
            domainList = FileReaderAndWriter.readFileAllLines(cmdOption.inputListOfDomainsToEvaluate);
        }
    }

    public Map<String, ClassificationEvaluation> run() {
        List<Documents> documentsOfAllDomains = readDocuments();
        // TODO: generate training documents and testing documents
        // generateTrainingDocsAndTestingDocs(documentsOfAllDomains);

        // if need randomly shuffle the sequence of domain list, use the following
//        Collections.shuffle(domainListOrg);
//        StringBuilder domainOutput = new StringBuilder();
//        for (int i = 0; i < domainListOrg.size(); ++i) {
//            domainOutput.append(domainListOrg.get(i));
//            domainOutput.append(System.lineSeparator());
//        }
//        String outputDomainPath = cmdOption.outputListOfDomainsToEvaluate + "shuffle10.txt";
//        FileReaderAndWriter.writeFile(outputDomainPath, domainOutput.toString());
//        domainList = FileReaderAndWriter.readFileAllLines(outputDomainPath);

        // read training documents one by one
        List<Documents> trainingDocumentsList = new ArrayList<Documents>();
        for (int k = 0; k < domainList.size(); ++k) {
            String knowledgeDomain = domainList.get(k);
            String trainingDocsFile = cmdOption.intermediateTrainingDocsDir + knowledgeDomain + ".txt";
            trainingDocumentsList.add(Documents.readDocuments(knowledgeDomain, trainingDocsFile));
        }

        // generate knowledge for training documents of this domain
        // Here, we also save trainingDocs, testingDocs and knowledge for each domain.
        Map<String, ClassificationKnowledge> mpDomainToKnowledge
                = generateClassificationKnowledgeForTrainingDocs(trainingDocumentsList);

//        Map<String, double[]> mapDomainSimilarity = new HashMap<String, double[]>();
//        for (int i = 0; i < domainList.size(); ++i) {
//            double[] pairSimilarity = new double[domainList.size()];
//            for (int j = 0; j < domainList.size(); ++j) {
//                if (j == i) {
//                    continue;
//                }
//                ClassificationKnowledge comKnowledge1 = mpDomainToKnowledge.get(domainList.get(i));
//                ClassificationKnowledge comKnowledge2 = mpDomainToKnowledge.get(domainList.get(j));
//                DomainSimilarity domainSimilarity = new DomainSimilarity(comKnowledge1, comKnowledge2);
//                double similarity = domainSimilarity.domainSentimentSimilarity();
//                pairSimilarity[j] = similarity;
//            }
//            mapDomainSimilarity.put(domainList.get(i),pairSimilarity);
//        }
//        Map<String, double[]> mapNormalDomainSimilarity = new HashMap<String, double[]>();
//        double[] pairSimilarityTemp = new double[domainList.size()];
//        for (Map.Entry<String, double[]> entry : mapDomainSimilarity.entrySet()) {
//            double maxSimilarity = 0;
//            for (int i = 0; i < entry.getValue().length; ++i) {
//                if (maxSimilarity < entry.getValue()[i]) {
//                    maxSimilarity = entry.getValue()[i];
//                }
//            }
//            for (int j = 0; j< entry.getValue().length; ++j) {
//                pairSimilarityTemp[j] = entry.getValue()[j] / maxSimilarity;
//            }
//            mapNormalDomainSimilarity.put(entry.getKey(), pairSimilarityTemp);
//        }

        // TODO: debugging, print word information
        // generate and print knowledge for each domain
//         printWordInformationOfEachDomain(documentsOfAllDomains);


        /** From here,
         *
         * Multi-tread: training and testing every target domain
         * Every domain is set as the target domain in turns
         */
        ClassifierParameters paramTemp = new ClassifierParameters();
        StringBuilder titleOutput = new StringBuilder();
        int lenStr = cmdOption.inputListOfDomainsToEvaluate.length();
        String fix = cmdOption.inputListOfDomainsToEvaluate.substring(lenStr-5, lenStr-4);
        //wangsong add
        String resultName = paramTemp.classifierName + "_" +
                cmdOption.attantionMode+"_" +
                cmdOption.gammaThreshold + "_" +
                cmdOption.positiveRatioThreshold+"_" +
                cmdOption.domainNumLavege;
        titleOutput.append(resultName);
        titleOutput.append(System.lineSeparator());
        for (int i = 0; i <= domainList.size() -1; ++i) {
            String domain = domainList.get(i);
            FileReaderAndWriter.writeFile(cmdOption.outputSentimentClassificationAccuracy
                    + "/" + resultName + "/" + domain + "_Acc.txt", titleOutput.toString());
            FileReaderAndWriter.writeFile(cmdOption.outputSentimentClassificationF1Score
                    + "/" + resultName + "/" + domain + "_F1Negative.txt", titleOutput.toString());
        }

        // Collections.shuffle(domainList);
//        for (int domain_id = 0; domain_id < domainList.size() - 1; ++domain_id) {
        for (int domain_id = 0; domain_id <= domainList.size()-1 ; ++domain_id) {
            String targetDomain = domainList.get(domain_id);
//            if (!newDomain.equals("Baby")) {
//                continue;
//            }

            // TODO: main body, from here
            // cmdOption.nthreads: the number of maximum threads in multithreading
            SentimentClassificationThreadPool threadPool = new SentimentClassificationThreadPool(
                    cmdOption.nthreads);
            Documents documents = new Documents();
            assert documentsOfAllDomains != null;
            for (int jj = 0; jj < documentsOfAllDomains.size(); ++jj) {
                if (Objects.equals(targetDomain, documentsOfAllDomains.get(jj).domain)) {
                    documents = documentsOfAllDomains.get(jj).getDeepClone();
                    break;
                }
            }
            // Get all past knowledge for this target domain.
            //wangsong add
            Map<String, ClassificationKnowledge> mPastKnowledgeForTargetDomain = new HashMap<String, ClassificationKnowledge>();
            ClassificationKnowledge comKnowledge1 = mpDomainToKnowledge.get(domainList.get(domain_id));
            double similaritySum = 0;
            double tempsim = 0;
            double[] similarity_row = new double[domainList.size()];
            double maxsim = 0;
            for (int j = 0; j < domainList.size(); ++j) {
                if (j == domain_id) {
                    // ws:if self then sim equ zero.
                    similarity_row[j] = 0;
                    continue;
                }
                ClassificationKnowledge comKnowledge2 = mpDomainToKnowledge.get(domainList.get(j));
                DomainSimilarity domainSimilarity = new DomainSimilarity(comKnowledge1, comKnowledge2);
                //ws: culculate sim
                tempsim = domainSimilarity.domainSentimentSimilarity();
                if (tempsim>maxsim){ // get the max to norm.
                    maxsim = tempsim;
                }
//                System.out.println(domainList.get(j)+':'+tempsim);
                // 记录每个值
                similarity_row[j] = tempsim;
                // sum as you see
                similaritySum += tempsim;
            }
            //domainSimilarity 这个参数是平均值
//            cmdOption.domainSimilarity = similaritySum/(domainList.size()-1);
//            cmdOption.domainNumLavege = Math.floor(domainList.size()/2)+Math.floor(domainList.size()*cmdOption.domainSimilarity/2);

//            System.out.println("domain similiarity:"+ cmdOption.domainSimilarity);
//            System.out.println("domain supportNumber:"+ cmdOption.domainNumLavege);

            // Get all past knowledge for this target domain.
//            Map<String, ClassificationKnowledge> mPastKnowledgeForTargetDomain = new HashMap<String, ClassificationKnowledge>();
//            // take out knowledge sequentially
            for (int j = 0; j < domainList.size(); ++j) {
                if (domain_id == j) {
                    continue;
                }

                switch (cmdOption.attantionMode){
                    case "none":{
                        mPastKnowledgeForTargetDomain.put(domainList.get(j), mpDomainToKnowledge.get(domainList.get(j)));
                        break;
                    }
                    case "att":{
                        mPastKnowledgeForTargetDomain.put(domainList.get(j),
                        addDomainSimlarityToPastKonwledge(domainList.get(j),
                                mpDomainToKnowledge.get(domainList.get(j)),
                                similarity_row[j]));
                        break;
                    }
                    case "att_max":{
                        mPastKnowledgeForTargetDomain.put(domainList.get(j),
                                                addDomainSimlarityToPastKonwledge(domainList.get(j),
                                                                                mpDomainToKnowledge.get(domainList.get(j)),
                                                                        similarity_row[j]/maxsim));
                        break;
                    }
                    case "att_percent":{
                        mPastKnowledgeForTargetDomain.put(domainList.get(j),
                        addDomainSimlarityToPastKonwledge(domainList.get(j),
                                mpDomainToKnowledge.get(domainList.get(j)),
                                (similarity_row[j]/similaritySum)*(domainList.size()-1)));
                        break;
                    }
                    default:
                        ExceptionUtility
                                .throwAndCatchException("cmdoption attmode is nothing!");
                        break;
                }
            }


            // classifier parameters
            ClassifierParameters param = new ClassifierParameters(documents, cmdOption);
            int k = 0;
            param.K = k;
            threadPool.addTask(documents, mPastKnowledgeForTargetDomain, param);

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
//                        + "/" + resultName + "/" + domainToEvaluate + "_F1Positive.txt", sbOutput.toString());
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
     * generate classification knowledge for training documents
     *
     * @param documentsOfAllDomains
     * @return classification knowledge
     */
    private void generateTrainingDocsAndTestingDocs(List<Documents> documentsOfAllDomains) {
        for (int i = 0; i < documentsOfAllDomains.size(); ++i) {
            Documents documents = documentsOfAllDomains.get(i).getDeepClone();
            String domain = documents.domain;
            ClassifierParameters param = new ClassifierParameters(documents, cmdOption);
            // Cross validation.
            CrossValidationOperatorMaintainingLabelDistribution cvo
                    = new CrossValidationOperatorMaintainingLabelDistribution(documents, param.noOfCrossValidationFolders);

            // get training and testing documents
            for (int folderIndex = 0; folderIndex < param.noOfCrossValidationFolders; ++folderIndex) {
                if (folderIndex != 0) {
                    continue;
                }
                // System.out.println("Folder No " + folderIndex);

                Pair<Documents, Documents> pair = cvo.getTrainingAndTestingDocuments(folderIndex,
                        param.noOfCrossValidationFolders);

                // 1. get training documents
                Documents trainingDocs = new Documents();
                trainingDocs.addDocuments(pair.t);
                // save training documents
                String trainingDocsPath = cmdOption.intermediateTrainingDocsDir + domain + ".txt";
                trainingDocs.printToFile(trainingDocsPath);
                // 2. get testing documents
                Documents testingDocs = new Documents();
                testingDocs.addDocuments(pair.u);
                // save testing documents
                String testingDocsPath = cmdOption.intermediateTestingDocsDir + domain + ".txt";
                testingDocs.printToFile(testingDocsPath);
            }
        }
    }

    /**
     * generate classification knowledge for training documents
     *
     * @param trainingDocumentsList
     * @return classification knowledge
     */
    private Map<String, ClassificationKnowledge> generateClassificationKnowledgeForTrainingDocs(
            List<Documents> trainingDocumentsList) {
        Map<String, ClassificationKnowledge> mpDomainToKnowledge = new HashMap<String, ClassificationKnowledge>();
        ClassifierParameters param = new ClassifierParameters();

        for (int i = 0; i < trainingDocumentsList.size(); ++i) {
            Documents trainingDocs = trainingDocumentsList.get(i).getDeepClone();
            String domain = trainingDocs.domain;
            String knowledgePath = cmdOption.intermediateKnowledgeDir + domain + ".txt";

//            if (new File(knowledgePath).exists()) {
//                // If the classification knowledge file already exists,
//                // -> load...
//                System.out.println("Loaded knowledge for target domain " + domain);
//                ClassificationKnowledge knowledge = ClassificationKnowledge
//                        .readClassificationProbabilitiesFromFile(knowledgePath);
//                mpDomainToKnowledge.put(domain, knowledge);
//            } else {
//
            System.out.println("LOL！ Obtain knowledge for target domain " + domain);
            // Extract classification knowledge: indexed by word (featured word)
            // 1. total number of documents in POS and NEG category: Freq(+) and Freq(-)
            // 2. Document-level knowledge: N_{+,w}^KB and N_{-,w}^KB
            // 3. total number of words in POS and NEG category: sum_f{Freq(f, +)} and sum_f{Freq(f, -)}
            // training all past task documents at once
            NaiveBayes nbClassifier = getKnowledgeBasedOnNBClassifier(trainingDocs);
            ClassificationKnowledge knowledge = nbClassifier.knowledge;

            // 4. Domain-level knowledge: M_{+,w}^KB and M_{-,w}^KB
            for (Map.Entry<String, double[]> entry : knowledge.wordCountInPerClass
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
//                if (probOfFeatureGivenPositive >= probOfFeatureGivenNegative) {
//                    // Positive feature in this domain.
//                    knowledge.countDomainsInPerClass
//                            .get(featureStr)[0]++;
//                } else {
//                    // Negative feature in this domain.
//                    knowledge.countDomainsInPerClass
//                            .get(featureStr)[1]++;
//                }
                // below is new version
                if ((probOfFeatureGivenPositive / probOfFeatureGivenNegative) >= param.gammaThreshold) {
                    // Positive feature in this domain.
                    knowledge.countDomainsInPerClass
                            .get(featureStr)[0]++;
                }
                if ((probOfFeatureGivenNegative / probOfFeatureGivenPositive) >= param.gammaThreshold) {
                    // Yes, it is param.positiveRatioThreshold
                    // as here I use "(probOfFeatureGivenNegative / probOfFeatureGivenPositive)"
                    // Negative feature in this domain.
                    knowledge.countDomainsInPerClass
                            .get(featureStr)[1]++;
                }
            }
            // print knowledge of each target domain to file

            knowledge.printToFile(knowledgePath);

            mpDomainToKnowledge.put(domain, knowledge);
        }
//    }
        return mpDomainToKnowledge;
    }

    private ClassificationKnowledge addDomainSimlarityToPastKonwledge(String domain,ClassificationKnowledge knowledge,double domainSim){
        System.out.println(domain+":"+domainSim);
        ClassificationKnowledge tempKnowledge = knowledge.getDeepClone();
        for (Map.Entry<String, double[]> entry : tempKnowledge.wordCountInPerClass.entrySet()){
//            System.out.println("before changed:"+entry.getKey() + entry.getValue()[0]);
            double[] wordCountInPerClasTemp = new double[2];
            wordCountInPerClasTemp = entry.getValue();
            wordCountInPerClasTemp[0] *= domainSim;
            wordCountInPerClasTemp[1] *= domainSim;
            entry.setValue(wordCountInPerClasTemp);
//            System.out.println("after modified:" + entry.getValue()[0]);
        }
        double[] allCountInPerClasTemp = tempKnowledge.countTotalWordsInPerClass;
        allCountInPerClasTemp[0] *= domainSim;
        allCountInPerClasTemp[1] *= domainSim;
        tempKnowledge.countTotalWordsInPerClass = allCountInPerClasTemp;
        return tempKnowledge;
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

        // get all featured word information (i.e., featured word from all domains)
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
