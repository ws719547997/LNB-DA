package classifier;

import classificationevaluation.ClassificationEvaluation;
import feature.Feature;
import feature.FeatureIndexer;
import feature.FeatureSelection;
import feature.Features;
import main.CmdOption;
import nlp.Document;
import nlp.Documents;
import utility.ExceptionUtility;
import utility.ItemWithValue;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Note: This class can provide knowledge for Lifelong, and is also a Naive Bayes framework.
 * <p>
 * Implement the naive Bayes, copying some from Lingpipe naive Bayes implementation.
 * <p>
 * // Not as good as implementation as Lingpipe.
 */
public class NaiveBayes_Sequence extends BaseClassifier {
    // The array of category, used in smoothing.
    private final String[] mCategories = {"+1", "-1"};

    // Freq(f, +) and Freq(f, -). -> wordCount(w,c)in POS and NEG documents, i.e., N_{+,w}^KB and N_{-,w}^KB
    private Map<String, double[]> mWordCountInPerClass = new HashMap<String, double[]>();
    private double[] mCountTotalWordsInPerClass = new double[2];// SUM_w {wordCount(w,c)} // indexed by c.
    // the total number of words in positive and negative category
    private double[] mCountDocsInPerClass = new double[2]; // countDocs(c). -> count number of Docs in each category
    private double mTotalDocsInAllClasses; // SUM_c countDocs(c). -> count the total number of Docs in all categories

    private FeatureIndexer featureIndexerAllDomains = null;

    private ClassificationKnowledge allKnowledge = null;

    public NaiveBayes_Sequence() {
    }

    public NaiveBayes_Sequence(FeatureSelection featureSelection2,
                                      ClassificationKnowledge knowledge2, ClassifierParameters param2) {
        featureSelection = featureSelection2;
        if (knowledge2 != null) {
            knowledge = knowledge2.getDeepClone();
            allKnowledge = knowledge2.getDeepClone();
        } else {
            knowledge = new ClassificationKnowledge();
            allKnowledge = new ClassificationKnowledge();
        }
        targetKnowledge = new ClassificationKnowledge();
        param = param2;
        featureIndexerAllDomains = new FeatureIndexer();
    }

    /**
     * Train the naive Bayes model. We assume that the word (feature) is
     * separated by blank character (e.g., ' ').
     *
     * @param trainingDocs
     */
    @Override
    public void train(Documents trainingDocs) {
        // Expand knowledge-base with the knowledge from the classic
        // naive Bayes classifier for this target domain
        NaiveBayes nbClassifier = new NaiveBayes(featureSelection, param);
        // get knowledge for this training documents
        nbClassifier.train(trainingDocs);

        // store current target knowledge
        targetKnowledge.addKnowledge(nbClassifier.knowledge);

        // Update classification knowledge.
        allKnowledge.addKnowledge(targetKnowledge.getDeepClone());

        // take out all featured words from the source domains (i.e., knowledge base)
        for (String featureStr : allKnowledge.wordCountInPerClass.keySet()) {
            featureIndexerAllDomains.addFeatureStrIfNotExistStartingFrom0(featureStr);
        }

        /** initialize */
        // total number of documents in POS and NEG category -> N_{+} and N_{-}
        mCountDocsInPerClass = allKnowledge.countDocsInPerClass;
        // Freq(f, +) and Freq(f, -). -> wordCount(w,c)in POS and NEG documents, i.e., N_{+,w}^KB and N_{-,w}^KB
        mWordCountInPerClass = allKnowledge.wordCountInPerClass;
        // sum_f{Freq(f, +)} and sum_f{Freq(f, -)}. -> number of total words in POS and NEG category
        mCountTotalWordsInPerClass = allKnowledge.countTotalWordsInPerClass;

        mTotalDocsInAllClasses = mCountDocsInPerClass[0] + mCountDocsInPerClass[1];
    }

    /**
     * Train the naive Bayes model.
     *
     * @param testingDocs
     * @return
     */
    @Override
    public ClassificationEvaluation test(Documents testingDocs) {

        for (Document testingDoc : testingDocs) {
            testingDoc.predict = this.getBestCategoryByClassification(
                    testingDoc.featuresForNaiveBayes);
        }
        ClassificationEvaluation evaluation = new ClassificationEvaluation(
                testingDocs.getLabels(), testingDocs.getPredicts(),
                param.domain);
        return evaluation;
    }

    /**
     * Classify the content and return the probability of each category. i.e., P(+|d) and P(-|d)
     */
    public double[] getCategoryProbByClassification(Features features) {

        // logps will add all P(w|c) and P(c) for each class
        // Why do this, why transfer to log?
        // The reason is that,
        // in this way, multiplication (\prod{p(w_i|c)}) will transfer to accumulation (\sum{log2(p(w_i|c))}).
        double[] logps = new double[2];
        for (Feature feature : features) {
            String featureStr = feature.featureStr;
            if (featureIndexerAllDomains != null
                    && !featureIndexerAllDomains.containsFeatureStr(featureStr)) {
                // The feature is not selected.
                continue;
            }

            mCountDocsInPerClass = allKnowledge.countDocsInPerClass;
            int totalWordSize = allKnowledge.wordCountInPerClass.size();
            double[] totalWordsInPerClass = allKnowledge.countTotalWordsInPerClass;
            // Penalty term 1, i.e., distinguishable target domain dependent words
            int knowledgeVocabularySize = allKnowledge.wordCountInPerClass.size();
            if (targetKnowledge.wordCountInPerClass.containsKey(featureStr)) {
                int wordSize = targetKnowledge.wordCountInPerClass.size();
                double[] wordCountInPerClassInTargetDomain = targetKnowledge.wordCountInPerClass.get(featureStr);
                double[] countTotalWordsInPerClass = targetKnowledge.countTotalWordsInPerClass;
                // transfer wordCount(w,c) to Pr(w|c)
                double profOfFeatureGivenPositiveInTargetDomain = probTokenByIndexArray(0,
                        wordCountInPerClassInTargetDomain, countTotalWordsInPerClass, wordSize); // Pr(w|+)
                double profOfFeatureGivenNegativeInTargetDomain = probTokenByIndexArray(1,
                        wordCountInPerClassInTargetDomain, countTotalWordsInPerClass, wordSize); // Pr(w|-)
                // get Ratio: Pr(w|+)/Pr(w|-)
                double ratioInTargetDomain = profOfFeatureGivenPositiveInTargetDomain
                        / profOfFeatureGivenNegativeInTargetDomain;
                // get {Pr(w|+)*(\lambda|V|+\sum_v X_{+,v})} / {Pr(w|-)*(\lambda|V|+\sum_v X_{-,v})}
                double ratioExpected = profOfFeatureGivenPositiveInTargetDomain
                        / profOfFeatureGivenNegativeInTargetDomain
                        * (allKnowledge.countTotalWordsInPerClass[0] + knowledgeVocabularySize
                        * param.smoothingPriorForFeatureInNaiveBayes)
                        / (allKnowledge.countTotalWordsInPerClass[1] + knowledgeVocabularySize
                        * param.smoothingPriorForFeatureInNaiveBayes);
                if ((ratioInTargetDomain >= param.positiveRatioThreshold
                        && wordCountInPerClassInTargetDomain[0] >= param.positiveOrNegativeFrequencyThreshold)
                        || (ratioInTargetDomain <= param.negativeRatioThreshold
                        && wordCountInPerClassInTargetDomain[1] >= param.positiveOrNegativeFrequencyThreshold)) {
                    // get wordCount(w,c) from knowledge base
                    double[] wordCountInPerClassInKnowledge = allKnowledge.wordCountInPerClass
                            .get(featureStr);
                    // get word total count, i.e., wordCount(w,+) + wordCount(w,-)
                    double wordTotalCountInAllClasses = wordCountInPerClassInKnowledge[0]
                            + wordCountInPerClassInKnowledge[1];

//                    mWordCountInPerClass.get(featureStr)[0] = wordTotalCountInAllClasses * ratioExpected / (ratioExpected + 1);
//                    mWordCountInPerClass.get(featureStr)[1] = wordTotalCountInAllClasses * 1.0 / (ratioExpected + 1);

////                    mCountTotalWordsInPerClass[0] -= mWordCountInPerClass.get(featureStr)[0];
////                    mCountTotalWordsInPerClass[1] -= mWordCountInPerClass.get(featureStr)[1];
                    mWordCountInPerClass.get(featureStr)[0] = wordCountInPerClassInTargetDomain[0];
                    mWordCountInPerClass.get(featureStr)[1] = wordCountInPerClassInTargetDomain[1];
                    totalWordSize = targetKnowledge.wordCountInPerClass.size();
                    totalWordsInPerClass = targetKnowledge.countTotalWordsInPerClass;
                    mCountDocsInPerClass = targetKnowledge.countDocsInPerClass;
////                    mCountTotalWordsInPerClass[0] += mWordCountInPerClass.get(featureStr)[0];
////                    mCountTotalWordsInPerClass[1] += mWordCountInPerClass.get(featureStr)[1];
                }
            }

            // Penalty term 2, i.e., Eq(8) in paper
            // If knowledge reliable, utilize only those reliable parts of knowledge, i.e., will ignore penalty term 1.
            if (knowledge.wordCountInPerClass.containsKey(featureStr)) {
                // Add domain level knowledge as regularization.
                double[] domainCounts = knowledge.countDomainsInPerClass.get(featureStr);
                if (domainCounts != null
                        && (domainCounts[0] >= param.domainLevelKnowledgeSupportThreshold
                        || domainCounts[1] >= param.domainLevelKnowledgeSupportThreshold)) {
                    double positivePercentage = domainCounts[0]
                            / (domainCounts[0] + domainCounts[1]); // R_w in Eq(8)
                    double[] wordCountInPerClassInPastDomain = knowledge.wordCountInPerClass.get(featureStr);
                    mWordCountInPerClass.get(featureStr)[0] = wordCountInPerClassInPastDomain[0] * positivePercentage;
                    mWordCountInPerClass.get(featureStr)[1] = wordCountInPerClassInPastDomain[1] * (1 - positivePercentage);
                    totalWordSize = knowledge.wordCountInPerClass.size();
                    totalWordsInPerClass = knowledge.countTotalWordsInPerClass;
                    mCountDocsInPerClass = knowledge.countDocsInPerClass;
                }
            }

            double[] tokenCounts = mWordCountInPerClass.get(featureStr);
            if (tokenCounts == null) {
                continue;
            }
            double[] tempLogPs = new double[2]; // -> POS and NEG
            for (int catIndex = 0; catIndex < mCategories.length; ++catIndex) {
                // where mCategories.length is 2
                logps[catIndex] += com.aliasi.util.Math
                        .log2(probTokenByIndexArray(catIndex, tokenCounts, totalWordsInPerClass, totalWordSize));
                tempLogPs[catIndex] += com.aliasi.util.Math
                        .log2(probTokenByIndexArray(catIndex, tokenCounts, totalWordsInPerClass, totalWordSize));
            }
            // Normalize the probability array, including .Math.pow
            tempLogPs = logJointToConditional(tempLogPs);
        }

        // add log class probability, i.e., p(+) and p(-)
        mTotalDocsInAllClasses = mCountDocsInPerClass[0] + mCountDocsInPerClass[1];
        for (int catIndex = 0; catIndex < mCategories.length; ++catIndex) {
            logps[catIndex] += com.aliasi.util.Math.log2(probCatByIndex(catIndex));
        }
        // Normalize the probability array, including .Math.pow
        return logJointToConditional(logps);
    }

    /**
     * Classify the content and return best category with highest probability.
     */
    public String getBestCategoryByClassification(Features features) {
        // get the probability of each category. i.e., P(+|d) and P(-|d)
        double[] categoryProb = getCategoryProbByClassification(features);

        // get the best category with highest probability
        double maximumProb = -Double.MAX_VALUE;
        int maximumIndex = -1;
        for (int i = 0; i < categoryProb.length; ++i) {
            if (maximumProb < categoryProb[i]) {
                maximumProb = categoryProb[i];
                maximumIndex = i;
            }
        }
        return mCategories[maximumIndex]; // mCategories = { "+1", "-1" }
    }

    /**
     * Pr(w|c) = (0.5 + count(w,c)) / (|V| * 0.5 + count(all of w, c)).
     */
    public double probTokenByIndexArray(int catIndex, double[] tokenCounts,
                                        double[] countTotalWords, int totalWordSize) {
        double tokenCatCount = tokenCounts[catIndex];
        double totalCatCount = countTotalWords[catIndex];
        return (tokenCatCount + param.smoothingPriorForFeatureInNaiveBayes)
                / (totalCatCount + totalWordSize * param.smoothingPriorForFeatureInNaiveBayes);
        // Note: mWordCountInPerClass.size() is the size of target training vocabulary,
        // i.e., the number of current featured words (selected features),
        // not the size of knowledge vocabulary.
    }

    /**
     * Pr(w|c) = (0.5 + count(w,c)) / (|V| * 0.5 + count(all of w, c)).
     */
    public double probTokenByIndexArray(int catIndex, double[] tokenCounts, int totalWordSize) {
        double tokenCatCount = tokenCounts[catIndex];
        double totalCatCount = mCountTotalWordsInPerClass[catIndex];
        return (tokenCatCount + param.smoothingPriorForFeatureInNaiveBayes)
                / (totalCatCount + totalWordSize * param.smoothingPriorForFeatureInNaiveBayes);
        // Note: mWordCountInPerClass.size() is the size of target training vocabulary,
        // i.e., the number of current featured words (selected features),
        // not the size of knowledge vocabulary.
    }

    /**
     * Pr(w|c) = (0.5 + count(w,c)) / (|V| * 0.5 + count(all of w, c)).
     */
    public double probTokenByIndexArray(int catIndex, double[] tokenCounts) {
        double tokenCatCount = tokenCounts[catIndex];
        double totalCatCount = mCountTotalWordsInPerClass[catIndex];
        return (tokenCatCount + param.smoothingPriorForFeatureInNaiveBayes)
                / (totalCatCount + featureIndexerAllDomains.size()
                * param.smoothingPriorForFeatureInNaiveBayes);
        // Note: mWordCountInPerClass.size() is the size of target training vocabulary,
        // i.e., the number of current featured words (selected features),
        // not the size of knowledge vocabulary.
    }

    /**
     * P(c) = (0.5 + count(c)) / (|C|) * 0.5 + count(all of c)).
     */
    private double probCatByIndex(int catIndex) {
        double caseCountCat = mCountDocsInPerClass[catIndex];

        return (caseCountCat + param.mCategoryPrior)
                / (mTotalDocsInAllClasses + mCategories.length * param.mCategoryPrior);
    }

    /**
     * Get the index of the category in the array.
     */
    private int getIndex(String category) {
        for (int i = 0; i < mCategories.length; ++i) {
            if (category.equals(mCategories[i])) {
                return i;
            }
        }
        ExceptionUtility.throwAndCatchException("No category is found!");
        return -1;
    }

    /**
     * P(w|c): Returns the probability of the specified token in the specified
     * category.
     *
     * @throws IllegalArgumentException If the category is not known or the token is not known.
     */
    public double probToken(String token, String cat) {
        int catIndex = getIndex(cat);
        double[] tokenCounts = mWordCountInPerClass.get(token);
        if (tokenCounts == null) {
            String msg = "Requires known token." + " Found token=" + token;
            throw new IllegalArgumentException(msg);
        }
        return probTokenByIndexArray(catIndex, tokenCounts);
    }

    /**
     * P(c) : Returns the probability estimate for the specified category.
     *
     * @param category Category whose probability is returned.
     * @return Probability for category.
     */
    public double probCat(String category) {
        int catIndex = getIndex(category);
        return probCatByIndex(catIndex);
    }

    /**
     * Normalize the probability array. Copy from the class
     * com.aliasi.classify.ConditionalClassification.
     *
     * @param logJointProbs
     * @return
     */
    private double[] logJointToConditional(double[] logJointProbs) {
        for (int i = 0; i < logJointProbs.length; ++i) {
            if (logJointProbs[i] > 0.0 && logJointProbs[i] < 0.0000000001)
                logJointProbs[i] = 0.0;
            if (logJointProbs[i] > 0.0 || Double.isNaN(logJointProbs[i])) {
                StringBuilder sb = new StringBuilder();
                sb.append("Joint probs must be zero or negative."
                        + " Found log2JointProbs[" + i + "]="
                        + logJointProbs[i]);
                for (int k = 0; k < logJointProbs.length; ++k)
                    sb.append("\nlogJointProbs[" + k + "]=" + logJointProbs[k]);
                throw new IllegalArgumentException(sb.toString());
            }
        }
        double max = com.aliasi.util.Math.maximum(logJointProbs);
        double[] probRatios = new double[logJointProbs.length];
        for (int i = 0; i < logJointProbs.length; ++i) {
            probRatios[i] = java.lang.Math.pow(2.0, logJointProbs[i] - max); // diff
            // is
            // <=
            // 0.0
            if (probRatios[i] == Double.POSITIVE_INFINITY)
                // where POSITIVE_INFINITY = 1.0 / 0.0
                probRatios[i] = Float.MAX_VALUE;
            else if (probRatios[i] == Double.NEGATIVE_INFINITY
                    || Double.isNaN(probRatios[i]))
                // where NEGATIVE_INFINITY = -1.0 / 0.0
                probRatios[i] = 0.0;
        }
        return com.aliasi.stats.Statistics.normalize(probRatios);
    }

    public double[] getTotalCountsPerCategory() {
        return mCountTotalWordsInPerClass;
    }

    @Override
    public List<ItemWithValue> getFeaturesByRatio(Document testingDoc) {
        List<ItemWithValue> featuresWithRatios = new ArrayList<ItemWithValue>();
        for (Feature feature : testingDoc.featuresForNaiveBayes) {
            String featureStr = feature.featureStr;
            if (featureSelection != null
                    && !featureSelection.isFeatureSelected(featureStr)) {
                // The feature is not selected.
                continue;
            }
            double[] tokenCounts = mWordCountInPerClass.get(featureStr);
            if (tokenCounts == null) {
                continue;
            }
            double[] ps = new double[2];
            for (int catIndex = 0; catIndex < mCategories.length; ++catIndex) {
                ps[catIndex] = probTokenByIndexArray(catIndex, tokenCounts);
            }
            ItemWithValue iwv = new ItemWithValue(featureStr, ps[0] / ps[1]);
            featuresWithRatios.add(iwv);
        }
        return featuresWithRatios;
    }

    @Override
    public void printMisclassifiedDocuments(Documents testingDocs, String misclassifiedDocumentsForOneCVFolderForOneDomainFilePath) {

    }

    @Override
    public double[] getCountsOfClasses(String featureStr) {
        return mWordCountInPerClass.get(featureStr);
    }

    public void verifyUnfoundFeatures() {
        for (String featureStr : this.featureSelection.selectedFeatureStrs) {
            if (!this.mWordCountInPerClass.containsKey(featureStr)) {
                System.out.println(featureStr);
            }
        }
    }
}
