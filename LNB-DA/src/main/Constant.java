package main;

public class Constant {
	public static final double SMOOTH_PROBABILITY = 0.000001;
	public static final double SMALL_THAN_AS_ZERO = 1e-4;
	// Note that this threshold is 0 for classification and 5 for topic model.
	public static int INFREQUENT_WORD_REMOVAL_THRESHOLD = 0;

	public final static int RANDOMSEED = 837191;
	public final static String SVM_LIGHT_LEARN_PATH = ".\\lib\\svm_light_program\\Windows\\svm_learn.exe";
	public final static String SVM_LIGHT_CLASSIFY_PATH = ".\\lib\\svm_light_program\\Windows\\svm_classify.exe";

	public final static boolean USE_DOMAIN_DEPENDENT_STOPWORDS = false;
}