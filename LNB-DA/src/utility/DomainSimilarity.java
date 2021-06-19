package utility;

import classifier.ClassificationKnowledge;

import java.util.*;

/**
 * Created by hao on 2/8/2018.
 */
public class DomainSimilarity {
    // Freq(+) and Freq(-). -> total number of documents in POS and NEG category, i.e., N_{+} and N_{-}
    private double[] countDocsInPerClass1 = null;
    private double[] countDocsInPerClass2 = null;
    // Freq(f, +) and Freq(f, -). -> wordCount(w,c)in POS and NEG documents, i.e., N_{+,w}^KB and N_{-,w}^KB
    private Map<String, double[]> wordCountInPerClass1 = null;
    private Map<String, double[]> wordCountInPerClass2 = null;

    private HashSet<String> featureIndexerBothTwoDomains = null;

    public DomainSimilarity (ClassificationKnowledge knowledge1, ClassificationKnowledge knowledge2) {
        // total number of documents in POS and NEG category
        countDocsInPerClass1 = knowledge1.countDocsInPerClass;
        countDocsInPerClass2 = knowledge2.countDocsInPerClass;
        // wordCount(w,c)in POS and NEG documents
        wordCountInPerClass1 = knowledge1.wordCountInPerClass;
        wordCountInPerClass2 = knowledge2.wordCountInPerClass;

        featureIndexerBothTwoDomains = new  HashSet<String>();
    }

    public double domainSentimentSimilarity() {
        double similarity = 0;
        // take out all featured words from the source domains (i.e., knowledge base)
        int positiveOrNegativeFreqThreshold = 5;
        int i = 0;
        int j = 0;
        for (String featureStr : wordCountInPerClass1.keySet()) {
            if ((wordCountInPerClass1.get(featureStr)[0] > positiveOrNegativeFreqThreshold)
                    || (wordCountInPerClass1.get(featureStr)[1] > positiveOrNegativeFreqThreshold)) {
                featureIndexerBothTwoDomains.add(featureStr);
            }
        }
        for (String featureStr : wordCountInPerClass2.keySet()) {
            if ((wordCountInPerClass2.get(featureStr)[0] > positiveOrNegativeFreqThreshold)
                    || (wordCountInPerClass2.get(featureStr)[1] > positiveOrNegativeFreqThreshold)) {
                featureIndexerBothTwoDomains.add(featureStr);
            }
        }

        double[] docsCountInPerClass1 = countDocsInPerClass1;
        double[] docsCountInPerClass2 = countDocsInPerClass2;
        SortedMap<String, Double> wordSentimentDistribution1 =
                pointwiseMutualInformation(wordCountInPerClass1, docsCountInPerClass1);
        SortedMap<String, Double> wordSentimentDistribution2 =
                pointwiseMutualInformation(wordCountInPerClass2, docsCountInPerClass2);
        if (wordSentimentDistribution1.size() != wordSentimentDistribution2.size()) {
            ExceptionUtility
                    .throwAndCatchException("The size of two domain is not same!");
        }
        // calculate domain similarity based on KL-divergence
        // can not work as wordSentimentDistribution contains negative value
//        for (String featureStr : featureIndexerBothTwoDomains) {
//            if ((wordSentimentDistribution1.get(featureStr) == 0.0)
//                    || (wordSentimentDistribution2.get(featureStr) == 0.0)) {
//                continue;
//            }
//            similarity += wordSentimentDistribution1.get(featureStr)
//                    * Math.log(wordSentimentDistribution1.get(featureStr)/wordSentimentDistribution2.get(featureStr));
//        }

        // calculate domain similarity based on KL-divergence
        double vectorProduct = 0;
        double vector1Modulo = 0;
        double vector2Modulo = 0;
        for (String featureStr : featureIndexerBothTwoDomains) {
            vectorProduct += wordSentimentDistribution1.get(featureStr)*wordSentimentDistribution2.get(featureStr);
            vector1Modulo += wordSentimentDistribution1.get(featureStr)*wordSentimentDistribution1.get(featureStr);
            vector2Modulo += wordSentimentDistribution2.get(featureStr)*wordSentimentDistribution2.get(featureStr);
        }
        vector1Modulo = Math.sqrt(vector1Modulo);
        vector2Modulo = Math.sqrt(vector2Modulo);
        similarity = vectorProduct/(vector1Modulo*vector2Modulo);

        if (similarity < 0) {
            similarity = 0;
        }
        return similarity;
    }

    public SortedMap<String, Double> pointwiseMutualInformation(Map<String, double[]> wordCountInPerClass,
                                                                 double[] docsCountInPerClass) {
        SortedMap<String, Double> pmiTable = new TreeMap<>();
        for (String featureStr : featureIndexerBothTwoDomains) {
            double wordFreqInPOS = 0;
            double wordFreqInNEG = 0;
            if (wordCountInPerClass.containsKey(featureStr)) {
                wordFreqInPOS = wordCountInPerClass.get(featureStr)[0];
                wordFreqInNEG = wordCountInPerClass.get(featureStr)[1];
            }
            wordFreqInPOS += 0.001;
            wordFreqInNEG += 0.001;
            double tempV = (wordFreqInPOS * docsCountInPerClass[1]) / (wordFreqInNEG*docsCountInPerClass[0]);
            double wordPMI = Math.log(tempV);
            pmiTable.put(featureStr, wordPMI);
        }
        return pmiTable;
    }
}
