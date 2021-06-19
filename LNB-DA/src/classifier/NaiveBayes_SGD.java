package classifier;

import classificationevaluation.ClassificationEvaluation;
import feature.Feature;
import feature.FeatureIndexer;
import feature.FeatureSelection;
import feature.Features;
import main.Constant;
import nlp.Document;
import nlp.Documents;
import utility.ExceptionUtility;
import utility.FileOneByOneLineWriter;
import utility.FileReaderAndWriter;
import utility.ItemWithValue;

import java.util.*;

/**
 * Implement the proposed lifelong model using stochastic gradient descent.
 * <p>
 * Note: true training and testing body of the proposed algorithm: Lifelong sentiment classification
 */
public class NaiveBayes_SGD extends BaseClassifier {
    // The array of category, used in smoothing.
    private final String[] mCategories = {"+1", "-1"};

    private FeatureIndexer featureIndexerAllDomains = null;
    private FeatureIndexer featureIndexerTargetDomain = null;
    List<Instance> data = null;

    // Freq(+) and Freq(-). -> total number of documents in POS and NEG category
    public double[] classInstanceCount = null;    // i.e., Count(+) and Count(-).
    // Virtual counts for the optimized variables
    private double[][] x = null; // x[v][0]: the virtual count for word v in POS class.
    // x[v][1]: the virtual count for word v in NEG class.
    private double[] sum_x = null; // sum_x[c] = sum_i{x[v][c]} -> number of total words in class c.
    private int V = 0; // vocabulary

    private Map<String, Double> mpRegExpectedWordCountInPositive = null;
    private Map<String, Double> mpRegCoefficientAlphaInPositive = null;
    private Map<String, Double> mpRegExpectedWordCountInNegative = null;
    private Map<String, Double> mpRegCoefficientAlphaInNegative = null;
    private double learningRate = 0.0;

    public NaiveBayes_SGD(ClassifierParameters param2,
                          FeatureSelection featureSelection2,
                          ClassificationKnowledge knowledge2,
                          ClassificationKnowledge targetKnowledge2,
                          Map<String, Double> regExpectedWordCountInPositive2,
                          Map<String, Double> regCoefficientAlphaInPositive2,
                          Map<String, Double> regExpectedWordCountInNegative2,
                          Map<String, Double> regCoefficientAlphaInNegative2) {
        param = param2;
        featureSelection = featureSelection2;
        knowledge = knowledge2;
        targetKnowledge = targetKnowledge2;
        mpRegExpectedWordCountInPositive = regExpectedWordCountInPositive2;
        mpRegCoefficientAlphaInPositive = regCoefficientAlphaInPositive2;
        mpRegExpectedWordCountInNegative = regExpectedWordCountInNegative2;
        mpRegCoefficientAlphaInNegative = regCoefficientAlphaInNegative2;

        featureIndexerAllDomains = new FeatureIndexer();
        featureIndexerTargetDomain = new FeatureIndexer();
    }

    /*****************
     * Stochastic Gradient Descent
     *****************/
    private void SGDEntry() {
        // Shuffle the data.
        Collections.shuffle(data, new Random(Constant.RANDOMSEED));
        double before = getObjectiveFunctionValueForInstancesOriginal(data);
        int iter = 0;
        while (true) {
            ++iter;
            // TODO: learningRate. Why can do it like this?
            learningRate = param.learningRate
                    / (1 + param.learningRateChange * param.learningRate
                    * param.regCoefficientAlpha * iter);
//			System.out.println("Iter:" + iter + "\tlearningRate:" + learningRate
//					+ "\tobjValue:" + before);
            if (param.maxSGDIterations >= 0 && iter > param.maxSGDIterations) {
                // arrive at the maximal iteration
                break;
            }

            for (int d = 0; d < data.size(); ++d) {
                // handle each feature (i.e., word)
                Instance instance = data.get(d);
                if (instance.y > 0) {
                    SGDForPositiveInstance(instance);
                } else {
                    SGDForNegativeInstance(instance);
                }
            }
            double after = getObjectiveFunctionValueForInstancesOriginal(data);
            if (Math.abs(after - before) <= param.convergenceDifference) {
                // arrive at convergence condition
                break;
            }
            before = after;
        }
    }

    private void SGDForPositiveInstance(Instance instance) {
        // get the length of document, i.e., the number of featured words
        int lengthOfDocument = 0; // |d_i|.
        for (int frequency : instance.mpFeatureToFrequency.values()) {
            lengthOfDocument += frequency;
        }

        // Compute the values related with \beta in Eq(4).
        // \lambda|V| + +\sum_v X_{+,v}
        double positiveClassSum = (param.smoothingPriorForFeatureInNaiveBayes * V + sum_x[0]);
        // \lambda|V| + +\sum_v X_{-,v}
        double negativeClassSum = (param.smoothingPriorForFeatureInNaiveBayes * V + sum_x[1]);
        // \beta
        double ratioOfClasses = positiveClassSum / negativeClassSum;

        // Compute R. i.e., Pr(-) / Pr(+)
        double R = getProbOfClass(1) / getProbOfClass(0); // Pr(-) / Pr(+).
        for (Map.Entry<Integer, Integer> entry : instance.mpFeatureToFrequency
                .entrySet()) {
            int featureId = entry.getKey();
            int frequency = entry.getValue();
            for (int i = 0; i < frequency; ++i) {
                // (lambda + x_-k) / (lambda + x_+k).
                R *= ratioOfClasses
                        * (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][1])
                        / (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][0]);
            }
        } // Now, R = (P(-)/P(+))*\prod_w\in d_i((\lambda+x_{-,w})/(\lambda+x_{+,w}))^n_{w,di}*g(x)

        // Compute the gradient of each feature for the positive class.
        // i.e., \frac{\partial F_{+,i}}{\partial X_{+,w}} in Eq(5).
        Map<Integer, Double> gradientOfFeatureWithPositiveClass = new HashMap<Integer, Double>();
        for (Map.Entry<Integer, Integer> entry : instance.mpFeatureToFrequency
                .entrySet()) {
            int featureId = entry.getKey(); // the featured word ID
            int frequency = entry.getValue(); // n_{u,di}, i.e., the count of word u in doc di.
            if (featureIndexerAllDomains.getFeatureStrGivenFeatureId(featureId)
                    .equals("toy")) {
                // System.out.println("toy");
            }
            if (featureIndexerAllDomains.getFeatureStrGivenFeatureId(featureId)
                    .equals("amazon")) {
                // System.out.println("amazon");
            }

            // get the partial derivative of the first item in the first item in Eq(5)
            double part1 = frequency
                    / (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][0])
                    / (1.0 + R);
            // get the partial derivative of the second item in the first item in Eq(5)
            double part2 = R / (1.0 + R) * lengthOfDocument / positiveClassSum;
            // get the partial derivative of the second item in Eq(5)
            double part3 = frequency / (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][0]);
            double gradient = 0.0;
            if (Double.isInfinite(R)) {
                gradient = (lengthOfDocument / positiveClassSum) - part3;
            } else {
                gradient = part1 + part2 - part3;
            }
            if (Double.isNaN(gradient) || Double.isInfinite(gradient)) {
                System.out.println("Nan");
            }
            if (Double.isInfinite(-gradient)) {
                System.out.println("Infinity");
            }
            // get the partial derivative of the Regularization items.
            gradient += getRegularizationTermGradientPositive(featureId);
            gradientOfFeatureWithPositiveClass.put(featureId, gradient);
        }

        // Compute the gradient of each feature for the negative class.
        Map<Integer, Double> gradientOfFeatureWithNegativeClass = new HashMap<Integer, Double>();
        for (Map.Entry<Integer, Integer> entry : instance.mpFeatureToFrequency
                .entrySet()) {
            int featureId = entry.getKey();
            int frequency = entry.getValue();
            if (featureIndexerAllDomains.getFeatureStrGivenFeatureId(featureId)
                    .equals("toy")) {
                // System.out.println("toy");
            }
            if (featureIndexerAllDomains.getFeatureStrGivenFeatureId(featureId)
                    .equals("amazon")) {
                // System.out.println("amazon");
            }

            double numerator = frequency
                    / (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][1])
                    - lengthOfDocument / negativeClassSum;
            double denominator = 0.0;
            if (Double.isInfinite(R)) {
                denominator = 1.0;
            } else {
                denominator = 1.0 / R + 1;
            }
            double gradient = 0.0;
            if (Double.isInfinite(denominator)) {
                gradient = 0.0;
            } else {
                gradient = numerator / denominator;
            }
            if (Double.isNaN(gradient) || Double.isInfinite(gradient)) {
                System.out.println("Nan");
            }
            // Regularization.
            gradient += getRegularizationTermGradientNegative(featureId);
            if (Double.isInfinite(-gradient)) {
                System.out.println("Infinity");
            }
            gradientOfFeatureWithNegativeClass.put(featureId, gradient);
        }

        // Note that we need to compute all of the gradients first and then
        // update
        // the counts.
        // Update the count of each feature in this document for positive class.
        updateXs(instance, gradientOfFeatureWithPositiveClass,
                gradientOfFeatureWithNegativeClass);
    }

    private void SGDForNegativeInstance(Instance instance) {
        // Compute the values related with g(x) function.
        int lengthOfDocument = 0;
        for (int frequency : instance.mpFeatureToFrequency.values()) {
            lengthOfDocument += frequency;
        }

        double positiveClassSum = (param.smoothingPriorForFeatureInNaiveBayes
                * V + sum_x[0]);
        double negativeClassSum = (param.smoothingPriorForFeatureInNaiveBayes
                * V + sum_x[1]);
        double ratioOfClasses = positiveClassSum / negativeClassSum;

        // Compute R.
        double R = getProbOfClass(1) / getProbOfClass(0); // Pr(-) / Pr(+).
        for (Map.Entry<Integer, Integer> entry : instance.mpFeatureToFrequency
                .entrySet()) {
            int featureId = entry.getKey();
            int frequency = entry.getValue();
            for (int i = 0; i < frequency; ++i) {
                // (lambda + x_-k) / (lambda + x_+k).
                R *= ratioOfClasses
                        * (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][1])
                        / (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][0]);
            }
        }

        // Compute the gradient of each feature for the positive class.
        Map<Integer, Double> gradientOfFeatureWithPositiveClass = new HashMap<Integer, Double>();
        for (Map.Entry<Integer, Integer> entry : instance.mpFeatureToFrequency
                .entrySet()) {
            int featureId = entry.getKey();
            int frequency = entry.getValue();
            if (featureIndexerAllDomains.getFeatureStrGivenFeatureId(featureId)
                    .equals("toy")) {
                // System.out.println("toy");
            }
            if (featureIndexerAllDomains.getFeatureStrGivenFeatureId(featureId)
                    .equals("amazon")) {
                // System.out.println("amazon");
            }

            double part1 = frequency
                    / (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][0])
                    / (1.0 + R);
            double part2 = R / (1.0 + R) * lengthOfDocument / positiveClassSum;
            double gradient = 0.0;
            if (Double.isInfinite(R)) {
                gradient = lengthOfDocument / positiveClassSum
                        - lengthOfDocument / positiveClassSum;
            } else {
                gradient = part1 + part2 - lengthOfDocument / positiveClassSum;
            }
            // verifyGradient(instance, featureId, gradient, 0);
            // verifyGradient(instance, featureId, gradient, 0);
            if (Double.isNaN(gradient) || Double.isInfinite(gradient)) {
                System.out.println("Nan");
            }
            // Regularization.
            gradient += getRegularizationTermGradientPositive(featureId);
            if (Double.isInfinite(-gradient)) {
                System.out.println("Infinity");
            }
            gradientOfFeatureWithPositiveClass.put(featureId, gradient);
        }

        // Compute the gradient of each feature for the negative class.
        Map<Integer, Double> gradientOfFeatureWithNegativeClass = new HashMap<Integer, Double>();
        for (Map.Entry<Integer, Integer> entry : instance.mpFeatureToFrequency
                .entrySet()) {
            int featureId = entry.getKey();
            int frequency = entry.getValue();
            if (featureIndexerAllDomains.getFeatureStrGivenFeatureId(featureId)
                    .equals("toy")) {
                // System.out.println("toy");
            }
            if (featureIndexerAllDomains.getFeatureStrGivenFeatureId(featureId)
                    .equals("amazon")) {
                // System.out.println("amazon");
            }

            double numerator = frequency
                    / (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][1])
                    - lengthOfDocument / negativeClassSum;
            double denominator = 0.0;
            if (Double.isInfinite(R)) {
                denominator = 1.0;
            } else {
                denominator = 1.0 / R + 1;
            }
            double gradient = 0.0;
            if (Double.isInfinite(denominator)) {
                gradient = -numerator;
            } else {
                gradient = numerator / denominator - numerator;
            }
            if (Double.isNaN(gradient)) {
                System.out.println("Nan");
            }
            // Regularization.
            gradient += getRegularizationTermGradientNegative(featureId);
            if (Double.isInfinite(-gradient)) {
                System.out.println("Infinity");
            }
            gradientOfFeatureWithNegativeClass.put(featureId, gradient);
        }

        // Note that we need to compute all of the gradients first and then
        // update the counts.
        // Update the count of each feature in this document for positive class.
        updateXs(instance, gradientOfFeatureWithPositiveClass,
                gradientOfFeatureWithNegativeClass);
    }

    /**
     * Regularization for positive R_+.
     */
    private double getRegularizationTermGradientPositive(int featureId) {
        if (mpRegExpectedWordCountInPositive == null) {
            return 0.0;
        }
        // Add L2 regularization.
        String featureStr = featureIndexerAllDomains
                .getFeatureStrGivenFeatureId(featureId);
        if (!mpRegExpectedWordCountInPositive.containsKey(featureStr)) {
            return 0.0;
        }
        double expectedPositiveCount = mpRegExpectedWordCountInPositive
                .get(featureStr);
        double regCoef = mpRegCoefficientAlphaInPositive.get(featureStr);
        double regularizationGradient = regCoef
                * (x[featureId][0] - expectedPositiveCount);
        return regularizationGradient;
    }

    /**
     * Regularization for positive R_+.
     */
    private double getRegularizationTermGradientNegative(int featureId) {
        if (mpRegExpectedWordCountInNegative == null) {
            return 0.0;
        }
        // Add L2 regularization.
        String featureStr = featureIndexerAllDomains
                .getFeatureStrGivenFeatureId(featureId);
        if (!mpRegExpectedWordCountInNegative.containsKey(featureStr)) {
            return 0.0;
        }
        double expectedNegativeCount = mpRegExpectedWordCountInNegative
                .get(featureStr);
        double regCoef = mpRegCoefficientAlphaInNegative.get(featureStr);
        double regularizationGradient = regCoef
                * (x[featureId][1] - expectedNegativeCount);
        return regularizationGradient;
    }

    /***************** Components related with function g(x) *****************/
    /**
     * g(x) = \Big(\frac{{}\lambda |V|+\sum\nolimits_{v}{x_{+,v}}}{{}\lambda
     * |V|+\sum\nolimits_{v}{x_{-,v}}}\Big)^{|d_i|}
     */
    // private double getGFunctionValue(int lengthOfDocument) {
    // double numerator = param.smoothingPriorForFeatureInNaiveBayes * V +
    // sum_x[0];
    // double denominator = param.smoothingPriorForFeatureInNaiveBayes * V +
    // sum_x[1];
    // double ratio = numerator / denominator;
    // return Math.pow(ratio, 1.0 * lengthOfDocument);
    // }

    /*****************
     * Update Xs in SGD
     *****************/
    public void updateXs(Instance instance, Map<Integer, Double> deltaPositive,
                         Map<Integer, Double> deltaNegative) {

        if (param.gradientVerificationUsingFiniteDifferences) {
            verifyGradient(instance, deltaPositive, 0);
            verifyGradient(instance, deltaNegative, 1);
        }

        for (Map.Entry<Integer, Integer> entry : instance.mpFeatureToFrequency
                .entrySet()) {
            int featureId = entry.getKey();
            // if
            // (featureIndexerAllDomains.getFeatureStrGivenFeatureId(featureId)
            // .equals("toy")) {
            // System.out
            // .println("toy : "
            // + (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][0])
            // + " "
            // + (param.smoothingPriorForFeatureInNaiveBayes +
            // x[featureId][1]));
            // }
            // if
            // (featureIndexerAllDomains.getFeatureStrGivenFeatureId(featureId)
            // .equals("amazon")) {
            // System.out
            // .println("amazon "
            // + (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][0])
            // + " "
            // + (param.smoothingPriorForFeatureInNaiveBayes +
            // x[featureId][1]));
            // }

            // update x in positive
            // Here, we enforce the x to be non negative.
            if (x[featureId][0] - learningRate * deltaPositive.get(featureId) < 0) {
                double delta = x[featureId][0];
                x[featureId][0] -= delta;
                sum_x[0] -= delta;
            } else {
                x[featureId][0] -= learningRate * deltaPositive.get(featureId);
                sum_x[0] -= learningRate * deltaPositive.get(featureId);
            }

            // update x in negative
            if (x[featureId][1] - learningRate * deltaNegative.get(featureId) < 0) {
                double delta = x[featureId][1];
                x[featureId][1] -= delta;
                sum_x[1] -= delta;
                if (Double.isInfinite(sum_x[1])) {
                    System.out.println("Infinity");
                }
            } else {
                x[featureId][1] -= learningRate * deltaNegative.get(featureId);
                sum_x[1] -= learningRate * deltaNegative.get(featureId);
                if (Double.isInfinite(sum_x[1])) {
                    System.out.println("Infinity");
                }
            }
        }
    }

    /*****************
     * Verify SGD using finite difference
     *****************/
    private void verifyGradient(Instance instance,
                                Map<Integer, Double> mpFeatureIdToGradient, int classIndex) {
        double before = getObjectiveFunctionValueForSingleInstanceLog(instance);
        int dividedBy = 1;
        // Divide it into a value that is smaller than 1 for digit matching.
        while (before >= 1) {
            before /= 10;
            dividedBy *= 10;
        }
        if (before == Double.MIN_VALUE) {
            // Cannot verify due to the numerical issues.
            return;
        }
        for (int featureId : mpFeatureIdToGradient.keySet()) {
            double delta = param.gradientVerificationDelta;
            x[featureId][classIndex] += delta;
            sum_x[classIndex] += delta;
            double after = getObjectiveFunctionValueForSingleInstanceLog(instance);
            after /= dividedBy;
            double gradientCalculated = mpFeatureIdToGradient.get(featureId);
            if (!(Math.abs(before + delta * gradientCalculated - after) < Constant.SMALL_THAN_AS_ZERO)) {
                if (instance.y == -1 && classIndex == 1) {
                    System.out.println("Document label " + instance.y);
                    System.out.println("Class " + classIndex);
                    System.out.println("Before : "
                            + (before + delta * gradientCalculated));
                    System.out.println("After : " + after);
                    System.out.println("Calculated Gradient: "
                            + gradientCalculated);
                    System.out.println("Expected Gradient: " + (after - before)
                            / delta);
                    System.out.println();
                    getObjectiveFunctionValueForSingleInstanceLog(instance);
                }
            }
            // ExceptionUtility
            // .assertAsException(
            // Math.abs(before + delta * gradientCalculated
            // - after) < Constant.SMALL_THAN_AS_ZERO,
            // param.domain + " Gradient verification fails!");
            // Revert the actions.
            x[featureId][classIndex] -= delta;
            sum_x[classIndex] -= delta;
        }
    }

    /***************** Verify SGD using finite difference *****************/
    // private void verifyGradient(Instance instance, int featureId,
    // double gradientCalculated, int classIndex) {
    // double before = getObjectiveFunctionValueForSingleInstanceLog(instance);
    // int dividedBy = 1;
    // // Divide it into a value that is smaller than 1 for digit matching.
    // while (before >= 1) {
    // before /= 10;
    // dividedBy *= 10;
    // }
    // if (before == Double.MIN_VALUE) {
    // // Cannot verify due to the numerical issues.
    // return;
    // }
    // double delta = param.gradientVerificationDelta;
    // x[featureId][classIndex] += delta;
    // sum_x[classIndex] += delta;
    // double after = getObjectiveFunctionValueForSingleInstanceLog(instance);
    // after /= dividedBy;
    // if (!(Math.abs(before + delta * gradientCalculated - after) <
    // Constant.SMALL_THAN_AS_ZERO)) {
    // if (instance.y == -1 && classIndex == 1) {
    // System.out.println("Document label " + instance.y);
    // System.out.println("Class " + classIndex);
    // System.out.println("Before : "
    // + (before + delta * gradientCalculated));
    // System.out.println("After : " + after);
    // System.out
    // .println("Calculated Gradient: " + gradientCalculated);
    // System.out.println("Expected Gradient: " + (after - before)
    // / delta);
    // System.out.println();
    // getObjectiveFunctionValueForSingleInstanceLog(instance);
    // }
    // }
    // // ExceptionUtility
    // // .assertAsException(
    // // Math.abs(before + delta * gradientCalculated
    // // - after) < Constant.SMALL_THAN_AS_ZERO,
    // // param.domain + " Gradient verification fails!");
    // // Revert the actions.
    // x[featureId][classIndex] -= delta;
    // sum_x[classIndex] -= delta;
    // }

    /***************** Objective function *****************/
    /**
     * Objective function in the log form.
     */
    public double getObjectiveFunctionValueForInstancesLog(List<Instance> data) {
        double sum = 0.0;
        for (Instance instance : data) {
            sum += getObjectiveFunctionValueForSingleInstanceLog(instance);
        }
        return sum / data.size();
    }

    /**
     * Objective function for single instance in the log form.
     */
    private double getObjectiveFunctionValueForSingleInstanceLog(
            Instance instance) {
        double ob = 0.0;
        if (instance.y > 0) {
            ob = getObjectiveFunctionValueForSinglePositiveInstanceLog(instance);
        } else {
            ob = getObjectiveFunctionValueForSingleNegativeInstanceLog(instance);
        }

        double regularizationTerm = 0.0;
        if (mpRegExpectedWordCountInPositive != null) {
            // Regularization for R_+.
            for (Map.Entry<String, Double> entry : mpRegExpectedWordCountInPositive
                    .entrySet()) {
                String featureStr = entry.getKey();
                int featureId = featureIndexerAllDomains
                        .getFeatureIdGivenFeatureStr(featureStr);
                double expectedPositiveCount = entry.getValue();
                double regCoef = mpRegCoefficientAlphaInPositive.get(featureStr);
                regularizationTerm += 0.5 * regCoef
                        * (x[featureId][0] - expectedPositiveCount)
                        * (x[featureId][0] - expectedPositiveCount);
            }
        }
        if (mpRegExpectedWordCountInNegative != null) {
            // Regularization for R_+.
            for (Map.Entry<String, Double> entry : mpRegExpectedWordCountInNegative
                    .entrySet()) {
                String featureStr = entry.getKey();
                int featureId = featureIndexerAllDomains
                        .getFeatureIdGivenFeatureStr(featureStr);
                double expectedNegativeCount = entry.getValue();
                double regCoef = mpRegCoefficientAlphaInNegative.get(featureStr);
                regularizationTerm += 0.5 * regCoef
                        * (x[featureId][1] - expectedNegativeCount)
                        * (x[featureId][1] - expectedNegativeCount);
            }
        }
        return ob + regularizationTerm;
    }

    /**
     * Objective function for single positive instance in the log form.
     */
    private double getObjectiveFunctionValueForSinglePositiveInstanceLog(
            Instance instance) {
        double positiveClassSum = (param.smoothingPriorForFeatureInNaiveBayes
                * V + sum_x[0]);
        double negativeClassSum = (param.smoothingPriorForFeatureInNaiveBayes
                * V + sum_x[1]);
        double ratioOfClasses = positiveClassSum / negativeClassSum;

        // Compute R.
        double R = getProbOfClass(1) / getProbOfClass(0); // Pr(-) / Pr(+).
        for (Map.Entry<Integer, Integer> entry : instance.mpFeatureToFrequency
                .entrySet()) {
            int featureId = entry.getKey();
            int frequency = entry.getValue();
            for (int i = 0; i < frequency; ++i) {
                // (lambda + x_-k) / (lambda + x_+k).
                R *= ratioOfClasses
                        * (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][1])
                        / (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][0]);
            }
        }
        if (Double.isInfinite(R)) {
            return Double.MIN_VALUE; // Invalid value.
        }
        if (R == 0) {
            return Double.MIN_VALUE; // Invalid value.
        }
        double value = 1.0 / 2 + R / 2.0; // (1 + R) / 2.
        if (Double.isInfinite(value)) {
            return Double.MIN_VALUE;
        }

        return Math.log(value);
    }

    /**
     * Objective function for single negative instance in the log form.
     */
    private double getObjectiveFunctionValueForSingleNegativeInstanceLog(
            Instance instance) {
        double positiveClassSum = (param.smoothingPriorForFeatureInNaiveBayes
                * V + sum_x[0]);
        double negativeClassSum = (param.smoothingPriorForFeatureInNaiveBayes
                * V + sum_x[1]);
        double ratioOfClasses = positiveClassSum / negativeClassSum;

        // Compute R.
        double R = getProbOfClass(1) / getProbOfClass(0); // Pr(-) / Pr(+).
        for (Map.Entry<Integer, Integer> entry : instance.mpFeatureToFrequency
                .entrySet()) {
            int featureId = entry.getKey();
            int frequency = entry.getValue();
            for (int i = 0; i < frequency; ++i) {
                // (lambda + x_-k) / (lambda + x_+k).
                R *= ratioOfClasses
                        * (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][1])
                        / (param.smoothingPriorForFeatureInNaiveBayes + x[featureId][0]);
            }
        }
        if (Double.isInfinite(R)) {
            return Math.log(0.5);
        }
        if (R == 0) {
            return Double.MIN_VALUE; // Invalid value.
        }
        double value = 1.0 / 2.0 / R + 1.0 / 2.0; // (1 + R) / (2 * R).
        if (Double.isInfinite(value)) {
            return Double.MIN_VALUE;
        }
        return Math.log(value);
    }

    /**
     * sum_i {y * (Pr(+) - Pr(-))} for multiple instance.
     */
    public double getObjectiveFunctionValueForInstancesOriginal(
            List<Instance> data) {
        // basic objective func value
        double sum = 0.0;
        for (Instance instance : data) {
            sum += getObjectiveFunctionValueForSingleInstanceOriginal(instance);
        }
        double ob = sum / data.size();

        // regularization terms
        double regularizationTerm = 0.0;
        if (mpRegExpectedWordCountInPositive != null) {
            // Regularization for R_+.
            for (Map.Entry<String, Double> entry : mpRegExpectedWordCountInPositive
                    .entrySet()) {
                String featureStr = entry.getKey();
                int featureId = featureIndexerAllDomains
                        .getFeatureIdGivenFeatureStr(featureStr);
                double expectedPositiveCount = entry.getValue();
                double regCoef = mpRegCoefficientAlphaInPositive.get(featureStr);
                regularizationTerm += 0.5 * regCoef
                        * (x[featureId][0] - expectedPositiveCount)
                        * (x[featureId][0] - expectedPositiveCount);
            }
        }
        if (mpRegExpectedWordCountInNegative != null) {
            // Regularization for R_+.
            for (Map.Entry<String, Double> entry : mpRegExpectedWordCountInNegative
                    .entrySet()) {
                String featureStr = entry.getKey();
                int featureId = featureIndexerAllDomains
                        .getFeatureIdGivenFeatureStr(featureStr);
                double expectedNegativeCount = entry.getValue();
                double regCoef = mpRegCoefficientAlphaInNegative.get(featureStr);
                regularizationTerm += 0.5 * regCoef
                        * (x[featureId][1] - expectedNegativeCount)
                        * (x[featureId][1] - expectedNegativeCount);
            }
        }

        return ob + regularizationTerm;
    }

    /**
     * y * (Pr(+) - Pr(-)) for single instance.
     */
    public double getObjectiveFunctionValueForSingleInstanceOriginal(
            Instance instance) {
        double[] logps = new double[2];
        for (Map.Entry<Integer, Integer> entry : instance.mpFeatureToFrequency
                .entrySet()) {
            int featureId = entry.getKey();
            int frequency = entry.getValue();

            for (int i = 0; i < frequency; ++i) {
                double[] tokenCounts = x[featureId];
                if (tokenCounts == null) {
                    continue;
                }
                for (int catIndex = 0; catIndex < mCategories.length; ++catIndex) {
                    logps[catIndex] += com.aliasi.util.Math
                            .log2(probTokenByIndexArray(catIndex, tokenCounts));
                }
            }
        }
        for (int catIndex = 0; catIndex < 2; ++catIndex) {
            logps[catIndex] += com.aliasi.util.Math
                    .log2(getProbOfClass(catIndex));
        }
        double[] probs = logJointToConditional(logps);
        return instance.y * (probs[0] - probs[1]);
    }

    /***************** Training and testing *****************/
    /**
     * This only works for binary classification for now.
     */
    @Override
    public void train(Documents trainingDocs) {
        // Convert the training documents to data, which consist of Map(featureID, Count), and label
        data = convertDocumentsToData(trainingDocs);

        // take out all featured words from the source domains (i.e., knowledge base)
        for (String featureStr : knowledge.wordCountInPerClass.keySet()) {
            featureIndexerAllDomains
                    .addFeatureStrIfNotExistStartingFrom0(featureStr);
        }

        // N(+) and N(-): Initialize array from data.
        classInstanceCount = new double[2];
        classInstanceCount[0] = trainingDocs.getNoOfPositiveLabels();
        classInstanceCount[1] = trainingDocs.getNoOfNegativeLabels();
        // X_{+,w}^0 and X_{-,w}^0: Initialize array from knowledge.
        V = featureIndexerAllDomains.size(); // knowledge vocabulary size
        x = new double[V][2]; // x = {featureId, wordCount in per class} // Actually, x = N^t + N^KB.
        sum_x = new double[2];
        for (int v = 0; v < V; ++v) {
            String featureStr = featureIndexerAllDomains.getFeatureStrGivenFeatureId(v);
            // TODO: unseen word trick
//            if (targetKnowledge.wordCountInPerClass.containsKey(featureStr)) {
//                x[v] = knowledge.wordCountInPerClass.get(featureStr);
//            } else if (knowledge.wordCountInPerClass.containsKey(featureStr)) {
//                x[v][0] = knowledge.wordCountInPerClass.get(featureStr)[0]*4/95
//                        + knowledge.wordCountInPerClass.get(featureStr)[0];
//                x[v][1] = knowledge.wordCountInPerClass.get(featureStr)[1]*4/95
//                        + knowledge.wordCountInPerClass.get(featureStr)[1];
//            } else {
//                x[v] = new double[]{0.0, 0.0};
//            }
            if (knowledge.wordCountInPerClass.containsKey(featureStr)) {
                x[v] = knowledge.wordCountInPerClass.get(featureStr);
            } else {
                // The word only appears in the target domain.
                x[v] = new double[]{0.0, 0.0};
            }
            sum_x[0] += x[v][0]; // the second item (cj = 0) in denominator of Eq(1)
            sum_x[1] += x[v][1]; // the second item (cj = 1) in denominator of Eq(1)
        }

//        // Check the size of knowledge vocabulary size
//        System.out.println("Knowledge vocabulary size: " + V);
        // Check if any value in x is nan or infinity.
        for (double[] aX : x) {
            ExceptionUtility
                    .assertAsException(!Double.isNaN(aX[0]), "Is Nan");
            ExceptionUtility
                    .assertAsException(!Double.isNaN(aX[1]), "Is Nan");
            ExceptionUtility.assertAsException(!Double.isInfinite(aX[0]),
                    "Is Infinite");
            ExceptionUtility.assertAsException(!Double.isInfinite(aX[1]),
                    "Is Infinite");
        }

        // call for Stochastic Gradient Descent
        if (param.convergenceDifference != Double.MAX_VALUE) {
            // Stochastic gradient descent.
            SGDEntry();
        }

        // Update classification knowledge. TODO: uncompleted coding
        // knowledge = new ClassificationKnowledge();
        // knowledge.countDocsInPerClass = mCaseCounts;
        // knowledge.wordCountInPerClass =
        // mFeatureStrToCountsMap;
        // knowledge.countTotalWordsInPerClass =
        // mTotalCountsPerCategory;
    }

    private List<Instance> convertDocumentsToData(Documents trainingDocs) {
        List<Instance> data = new ArrayList<Instance>();
        for (Document trainingDoc : trainingDocs) {
            Instance instance = new Instance();
            for (Feature feature : trainingDoc.featuresForNaiveBayes) {
                String featureStr = feature.featureStr;
                // if (featureSelection != null
                // && !featureSelection.isFeatureSelected(featureStr)) {
                // // The feature is not selected.
                // continue;
                // }
                int featureId = featureIndexerAllDomains
                        .getFeatureIdOtherwiseAddFeatureStrStartingFrom0(featureStr);
                featureIndexerTargetDomain
                        .addFeatureStrIfNotExistStartingFrom0(featureStr);
                if (!instance.mpFeatureToFrequency.containsKey(featureId)) {
                    instance.mpFeatureToFrequency.put(featureId, 0);
                }
                instance.mpFeatureToFrequency.put(featureId,
                        instance.mpFeatureToFrequency.get(featureId) + 1);
            }
            instance.y = Integer.parseInt(trainingDoc.label);
            data.add(instance);
        }
        return data;
    }

    @Override
    public ClassificationEvaluation test(Documents testingDocs) {
        String filepath = param.nbRootDirectory + param.domain + ".txt";
        FileOneByOneLineWriter writer = new FileOneByOneLineWriter(filepath);
        for (Document testingDoc : testingDocs) {
            testingDoc.predict = this.getBestClassByClassification(
                    testingDoc.featuresForNaiveBayes, writer);
        }
        writer.close();
        ClassificationEvaluation evaluation = new ClassificationEvaluation(
                testingDocs.getLabels(), testingDocs.getPredicts(),
                param.domain);
        // 王松添加的
//        for(String str : testingDocs.getPredicts()){
//            System.out.println(str);
//        }
//        System.out.println(testingDocs.size());

        return evaluation;
    }


    public ClassificationEvaluation test_with_only_training_feature(Documents testingDocs,
                                                                    Documents trainingDocs) {
        String filepath = param.lifelongWithOnlyTrainingData + param.domain + ".txt";
        FileOneByOneLineWriter writer = new FileOneByOneLineWriter(filepath);
        for (Document testingDoc : testingDocs) {
            testingDoc.predict = this.getBestClassByClassification(
                    testingDoc.featuresForNaiveBayes, writer, trainingDocs);
        }
        writer.close();
        ClassificationEvaluation evaluation = new ClassificationEvaluation(
                testingDocs.getLabels(), testingDocs.getPredicts(),
                param.domain);
        return evaluation;
    }

    /**
     * Classify the content and return best category with highest probability.
     */
    public String getBestClassByClassification(Features features,
                                               FileOneByOneLineWriter writer) {
        double[] categoryProb = getClassProbByClassification(features, writer);
        double maximumProb = -Double.MAX_VALUE;
        int maximumIndex = -1;
        for (int i = 0; i < categoryProb.length; ++i) {
            if (maximumProb < categoryProb[i]) {
                maximumProb = categoryProb[i];
                maximumIndex = i;
            }
        }
        return mCategories[maximumIndex];
    }

    public String getBestClassByClassification(Features features,
                                               FileOneByOneLineWriter writer,
                                               Documents trainingDocs) {
        double[] categoryProb = getClassProbByClassification(features, writer, trainingDocs);
        double maximumProb = -Double.MAX_VALUE;
        int maximumIndex = -1;
        for (int i = 0; i < categoryProb.length; ++i) {
            if (maximumProb < categoryProb[i]) {
                maximumProb = categoryProb[i];
                maximumIndex = i;
            }
        }
        return mCategories[maximumIndex];
    }

    /**
     * Classify the content and return the probability of each category. i.e., P(+|d) and P(-|d)
     */
    public double[] getClassProbByClassification(Features features,
                                                 FileOneByOneLineWriter writer) {

        // logps will add all P(w|c) and P(c) for each class
        // Why do this, why transfer to log?
        // The reason is that,
        // in this way, multiplication (\prod{p(w_i|c)}) will transfer to accumulation (\sum{log2(p(w_i|c))}).
        double[] logps = new double[2];
        for (Feature feature : features) {
            String featureStr = feature.featureStr;


            if (featureStr.equals("antenna")) {
                // System.out.println("get it");
            }

            // If Knowledge Base does not contain this feature (word), continue
            if (!featureIndexerAllDomains.containsFeatureStr(featureStr)) {
                // The feature is not selected.
                continue;
            }

            // get wordCount(w,c)in POS and NEG documents
            int featureId = featureIndexerAllDomains
                    .getFeatureIdGivenFeatureStr(featureStr);
            double[] tokenCounts = x[featureId];
            if (tokenCounts == null) {
                continue;
            }
            for (int catIndex = 0; catIndex < mCategories.length; ++catIndex) {
                // where mCategories.length is 2
                logps[catIndex] += com.aliasi.util.Math
                        .log2(probTokenByIndexArray(catIndex, tokenCounts));
                // TODO: close now
//                writer.writeLine(featureStr + " : "
//                        + probTokenByIndexArray(catIndex, tokenCounts));
            }
        }
        // add log class probability, i.e., p(+) and p(-)
        for (int catIndex = 0; catIndex < 2; ++catIndex) {
            logps[catIndex] += com.aliasi.util.Math
                    .log2(getProbOfClass(catIndex));
        }
        // Normalize the probability array, including .Math.pow
        return logJointToConditional(logps);
    }


    public double[] getClassProbByClassification(Features features,
                                                 FileOneByOneLineWriter writer,
                                                 Documents trainingDocs) {

        Map<String, Integer> trainWordMP = trainingDocs.getMpWordFrequency();

        double[] logps = new double[2];
        for (Feature feature : features) {
            String featureStr = feature.featureStr;

            if (featureStr.equals("antenna")) {
                // System.out.println("get it");
            }

            // check whether training data contains this feature (or word)
            if (!featureIndexerTargetDomain.containsFeatureStr(featureStr)) {
                continue;
            }

            // check whether knowledge base contains this feature
            if (!featureIndexerAllDomains.containsFeatureStr(featureStr)) {
                continue;
            }

            // take out featureId from knowledge base for this feature
            int featureId = featureIndexerAllDomains
                    .getFeatureIdGivenFeatureStr(featureStr);
            double[] tokenCounts = x[featureId];
            if (tokenCounts == null) {
                continue;
            }
            for (int catIndex = 0; catIndex < mCategories.length; ++catIndex) {
                logps[catIndex] += com.aliasi.util.Math
                        .log2(probTokenByIndexArray(catIndex, tokenCounts));
                // Todo: close now
//                writer.writeLine(featureStr + " : "
//                        + probTokenByIndexArray(catIndex, tokenCounts));
            }
        }
        for (int catIndex = 0; catIndex < 2; ++catIndex) {
            logps[catIndex] += com.aliasi.util.Math
                    .log2(getProbOfClass(catIndex));
        }
        return logJointToConditional(logps);
    }

    /**
     * Pr(w|c) = (0.5 + count(w,c)) / (|V| * 0.5 + count(all of w, c)).
     */
    private double probTokenByIndexArray(int catIndex, double[] tokenCounts) {
        double tokenCatCount = tokenCounts[catIndex];
        if (tokenCatCount < 0) {
            tokenCatCount = 0;
        }
        double totalCatCount = sum_x[catIndex];
//        double totalWordsCount = 0;
//        for (String featureStr : targetKnowledge.wordCountInPerClass.keySet()) {
//            totalWordsCount += knowledge.wordCountInPerClass.get(featureStr)[catIndex];
//        }

        return (tokenCatCount + param.smoothingPriorForFeatureInNaiveBayes)
                / (totalCatCount + featureIndexerAllDomains.size()
                * param.smoothingPriorForFeatureInNaiveBayes);
        // Note: featureIndexerAllDomains.size() is the size of knowledge vocabulary.
        // targetKnowledge.wordCountInPerClass.size() is the size of target vocabulary.
    }

    /**
     * P(c) = (0.5 + count(c)) / (|C) * 0.5 + count(all of c)).
     */
    private double getProbOfClass(int catIndex) {
        return (classInstanceCount[catIndex] + param.mCategoryPrior)
                / (classInstanceCount[0] + classInstanceCount[1] + mCategories.length
                * param.mCategoryPrior);
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
                probRatios[i] = Float.MAX_VALUE;
            else if (probRatios[i] == Double.NEGATIVE_INFINITY
                    || Double.isNaN(probRatios[i]))
                probRatios[i] = 0.0;
        }
        return com.aliasi.stats.Statistics.normalize(probRatios);
    }

    @Override
    public List<ItemWithValue> getFeaturesByRatio(Document testingDoc) {
        List<ItemWithValue> featuresWithRatios = new ArrayList<ItemWithValue>();
        for (Feature feature : testingDoc.featuresForNaiveBayes) {
            String featureStr = feature.featureStr;
            // if emerging not in knowledge base
            if (featureIndexerAllDomains != null
                    && !featureIndexerAllDomains.containsFeatureStr(featureStr)) {
                continue;
            }
            // if emerging not in training
//            if (featureSelection != null
//                    && !featureSelection.isFeatureSelected(featureStr)) {
//                continue;
//            }
            int featureId = featureIndexerAllDomains
                    .getFeatureIdGivenFeatureStr(featureStr);
            double[] tokenCounts = x[featureId];
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

    private double[] getProbOfClasses(Document testingDoc){
        double[] logps = new double[2];
        for (Feature feature : testingDoc.featuresForNaiveBayes) {
            String featureStr = feature.featureStr;

            // If Knowledge Base does not contain this feature (word), continue
            if (!featureIndexerAllDomains.containsFeatureStr(featureStr)) {
                // The feature is not selected.
                continue;
            }

            // get wordCount(w,c)in POS and NEG documents
            int featureId = featureIndexerAllDomains
                    .getFeatureIdGivenFeatureStr(featureStr);
            double[] tokenCounts = x[featureId];
            if (tokenCounts == null) {
                continue;
            }
            for (int catIndex = 0; catIndex < mCategories.length; ++catIndex) {
                // where mCategories.length is 2
                logps[catIndex] += com.aliasi.util.Math
                        .log2(probTokenByIndexArray(catIndex, tokenCounts));
            }
        }
        // add log class probability, i.e., p(+) and p(-)
        for (int catIndex = 0; catIndex < 2; ++catIndex) {
            logps[catIndex] += com.aliasi.util.Math
                    .log2(getProbOfClass(catIndex));
        }
        // Normalize the probability array, including .Math.pow
        return logJointToConditional(logps);
    }

    @Override
    public double[] getCountsOfClasses(String featureStr) {
        int featureId = featureIndexerAllDomains
                .getFeatureIdGivenFeatureStr(featureStr);
        return x[featureId];
    }

    @Override
    public void printMisclassifiedDocuments(Documents testingDocs,
                                             String misclassifiedDocumentsForOneCVFolderForOneDomainFilePath) {
        StringBuilder sbOutput = new StringBuilder();
        sbOutput.append("Target knowledge size: " + targetKnowledge.wordCountInPerClass.size());
        sbOutput.append(System.lineSeparator());
        sbOutput.append("Source and target knowledge size: " + knowledge.wordCountInPerClass.size());
        sbOutput.append(System.lineSeparator());
        sbOutput.append("===================================================");
        sbOutput.append(System.lineSeparator());
        sbOutput.append("No.\t Document\t Label\t Predict\t CorrectOrWrong");
        sbOutput.append(System.lineSeparator());
        sbOutput.append("ClassProb <+:->");
        sbOutput.append(System.lineSeparator());
        sbOutput.append("Feature Ratio: Pr(w|+)/Pr(w|-) Pr(w|+):Pr(w|-)" + "\t"
                + "wordCount(+,w):wordCount(-,w)" + "\t"
                + "VirtualCount(+,w):VirtualCount(-,w)" + "\t"
                + "#Domains(Pr(w|+)>Pr(w|-)):#Domains(Pr(w|+)<Pr(w|-))" + "\t");
        sbOutput.append(System.lineSeparator());
        sbOutput.append("===================================================");
        sbOutput.append(System.lineSeparator());
        int index = 0;
        for (Document testingDoc : testingDocs) {
            if (!testingDoc.label.equals(testingDoc.predict)) {
                // title information
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

                // probability for both POS and NEG
                double[] probForClasses = getProbOfClasses(testingDoc);
                sbOutput.append("ClassProb<+:-> " + probForClasses[0] + ":" + probForClasses[1]);
                sbOutput.append(System.lineSeparator());

                // word probability in both POS and NEG
                List<ItemWithValue> featuresWithRatios = getFeaturesByRatio(testingDoc);
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
                    int featureId = featureIndexerAllDomains
                            .getFeatureIdGivenFeatureStr(featureStr);
                    double[] tokenCounts = x[featureId];
                    double[] probOfWord = new double[2];
                    for (int catIndex = 0; catIndex < mCategories.length; ++catIndex) {
                        probOfWord[catIndex] = probTokenByIndexArray(catIndex, tokenCounts);
                    }
                    if (featureSelection != null
                            && !featureSelection.isFeatureSelected(featureStr)) {
                        sbOutput.append(featureStr + "\tRatio: " + ratio + "\t"
                                + probOfWord[0] + ":" + probOfWord[1] + "\t"
                                + (int)knowledge.wordCountInPerClass.get(featureStr)[0]
                                + ":"
                                + (int)knowledge.wordCountInPerClass.get(featureStr)[1]
                                + "\t"
                                + tokenCounts[0]
                                + ":"
                                + tokenCounts[1]
                                + "\t"
                                + (int)knowledge.countDomainsInPerClass.get(featureStr)[0]
                                + ":"
                                + (int)knowledge.countDomainsInPerClass.get(featureStr)[1]
                                + "\t::not in training");
                        sbOutput.append(System.lineSeparator());
                    } else {
                        sbOutput.append(featureStr + "\tRatio: " + ratio + "\t"
                                + probOfWord[0] + ":" + probOfWord[1] + "\t"
                                + (int)knowledge.wordCountInPerClass.get(featureStr)[0]
                                + ":"
                                + (int)knowledge.wordCountInPerClass.get(featureStr)[1]
                                + "\t"
                                + tokenCounts[0]
                                + ":"
                                + tokenCounts[1]
                                + "\t"
                                + (int)knowledge.countDomainsInPerClass.get(featureStr)[0]
                                + ":"
                                + (int)knowledge.countDomainsInPerClass.get(featureStr)[1]);
                        sbOutput.append(System.lineSeparator());
                    }
                }
                sbOutput.append(System.lineSeparator());
            }
        }
        FileReaderAndWriter.writeFile(
                misclassifiedDocumentsForOneCVFolderForOneDomainFilePath,
                sbOutput.toString());
    }
}
