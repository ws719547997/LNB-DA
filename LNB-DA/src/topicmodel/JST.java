package topicmodel;

import nlp.Corpus;
import nlp.Topics;
import utility.ArrayAllocationAndInitialization;
import utility.ExceptionUtility;
import utility.InverseTransformSampler;

/**
 * This implements the JST (joint sentiment/topic) model (Lin and He, CIKM
 * 2009).
 * 
 * Sentiment 0: positive. Sentiment 1: neutral. Sentiment 2: negative.
 */
public class JST extends TopicModel {
	/******************* Hyperparameters *********************/
	// The hyperparameter for the document-sentiment distribution.
	private double gamma = 0;
	private double sGamma = 0;

	// The hyperparameter for the document-sentiment-topic distribution.
	// alpha is in the variable param in TopicModel.
	private double tAlpha = 0;

	// The hyperparameter for the sentiment-topic-word distribution.
	// beta is in the variable param in TopicModel.
	private double vBeta = 0;

	/******************* Posterior distributions *********************/
	private double[][] pi = null; // Document-sentiment distribution, size D *
									// S.
	private double[][] pisum = null; // Cumulative document-sentiment
										// distirbution, size D * S.
	private double[][][] theta = null; // Document-sentiment-topic distribution,
										// size D * S * T.
	private double[][][] thetasum = null; // Cumulative document-sentiment-topic
											// distribution, size D * S * T.
	private double[][][] phi = null; // Sentiment-topic-word distribution, size
										// S * T * V.
	private double[][][] phisum = null; // Cumulative sentiment-topic-word
										// distribution, size S * T * V.
	// Number of times to add the sum arrays, such as thetasum and phisum.
	public int numstats = 0;

	/******************* Temp variables while sampling *********************/
	// z is defined in the superclass TopicModel.
	// y is defined in the superclass TopicModel.
	// nds[d][s]: the counts words with sentiment s in document d.
	private int[][] nds = null;
	// ndsum[d]: the counts words with any sentiment in document d.
	private int[] ndsum = null;

	// ndst[d][s][t]: the counts words with sentiment s and topic t in document
	// d.
	private int[][][] ndst = null;
	// ndssum[d][s] is nds[d][s].
	// private int[] ndssum = null;

	// nstw[s][t][w]: the counts of word w appearing under topic t and sentiment
	// s.
	private int[][][] nstw = null;
	// nstsum[s][t]: the counts of any word appearing under topic t and
	// sentiment s.
	private int[][] nstsum = null;

	/**
	 * Create a new topic model with all variables initialized. The z[][] is
	 * randomly assigned.
	 */
	public JST(Corpus corpus2, TopicModelParameters param2) {
		super(corpus2, param2);
		tAlpha = param.T * param.alpha;
		vBeta = param.V * param.beta;
		gamma = param.gamma;
		sGamma = param.S * param.gamma;
		// Allocate memory for temporary variables and initialize their
		// values.
		allocateMemoryForTempVariables();
		// Initialize the first status of Markov chain randomly.
		initializeFirstMarkovChainRandomly();
	}

	public JST(Corpus corpus2, TopicModelParameters param2, int[][] z2,
			int[][] y2, double[][] dsdist,
			double[][][] sentimentTopicWordDistribution) {
		super(corpus2, param2);
		tAlpha = param.T * param.alpha;
		vBeta = param.V * param.beta;
		gamma = param.gamma;
		sGamma = param.S * param.gamma;
		// Allocate memory for temporary variables and initialize their
		// values.
		allocateMemoryForTempVariables();
		z = z2;
		y = y2;

		pi = dsdist;
		phi = sentimentTopicWordDistribution;
	}

	// ------------------------------------------------------------------------
	// Memory Allocation and Initialization
	// ------------------------------------------------------------------------
	/**
	 * Allocate memory for temporary variables and initialize their values. Note
	 * that z[][] is not created in this function, but in the function
	 * initializeFirstMarkovChainRandomly().
	 */
	private void allocateMemoryForTempVariables() {
		/******************* Posterior distributions *********************/
		pi = ArrayAllocationAndInitialization.allocateAndInitialize(pi,
				param.D, param.S);
		theta = ArrayAllocationAndInitialization.allocateAndInitialize(theta,
				param.D, param.S, param.T);
		phi = ArrayAllocationAndInitialization.allocateAndInitialize(phi,
				param.S, param.T, param.V);
		if (param.sampleLag > 0) {
			pisum = ArrayAllocationAndInitialization.allocateAndInitialize(pi,
					param.D, param.S);
			thetasum = ArrayAllocationAndInitialization.allocateAndInitialize(
					theta, param.D, param.S, param.T);
			phisum = ArrayAllocationAndInitialization.allocateAndInitialize(
					phi, param.S, param.T, param.V);
		}

		/******************* Temp variables while sampling *********************/
		nds = ArrayAllocationAndInitialization.allocateAndInitialize(nds,
				param.D, param.S);
		ndsum = ArrayAllocationAndInitialization.allocateAndInitialize(ndsum,
				param.D);
		ndst = ArrayAllocationAndInitialization.allocateAndInitialize(ndst,
				param.D, param.S, param.T);
		nstw = ArrayAllocationAndInitialization.allocateAndInitialize(nstw,
				param.S, param.T, param.V);
		nstsum = ArrayAllocationAndInitialization.allocateAndInitialize(nstsum,
				param.S, param.T);
	}

	/**
	 * Initialize the first status of Markov chain randomly. Note that z[][] is
	 * created in this function.
	 */
	private void initializeFirstMarkovChainRandomly() {
		z = new int[param.D][];
		y = new int[param.D][];

		for (int d = 0; d < param.D; ++d) {
			int N = docs[d].length;
			z[d] = new int[N];
			y[d] = new int[N];

			for (int n = 0; n < N; ++n) {
				int word = docs[d][n];
				int topic = (int) Math.floor(randomGenerator.nextDouble()
						* param.T);
				int sentiment = (int) Math.floor(randomGenerator.nextDouble()
						* param.S);
				z[d][n] = topic;
				y[d][n] = sentiment;

				updateCount(d, topic, sentiment, word, +1);
			}
		}
	}

	/**
	 * There are several main steps:
	 * 
	 * 1. Run a certain number of Gibbs Sampling sweeps.
	 * 
	 * 2. Compute the posterior distributions.
	 */
	@Override
	public void run() {
		// 1. Run a certain number of Gibbs Sampling sweeps.
		runGibbsSampling();
		// 2. Compute the posterior distributions.
		computePosteriorDistribution();
	}

	// ------------------------------------------------------------------------
	// Gibbs Sampler
	// ------------------------------------------------------------------------

	/**
	 * Run a certain number of Gibbs Sampling sweeps.
	 */
	private void runGibbsSampling() {
		for (int i = 0; i < param.nIterations; ++i) {
			// System.out.println("Gibbs Iter " + i);
			for (int d = 0; d < param.D; ++d) {
				int N = docs[d].length;
				for (int n = 0; n < N; ++n) {
					// Sample from p(z_i|z_-i, w)
					sampleTopicAssignment(d, n);
				}
			}

			if (i >= param.nBurnin && param.sampleLag > 0
					&& i % param.sampleLag == 0) {
				updatePosteriorDistribution();
			}
		}
	}

	/**
	 * Sample a topic assigned to the word in position n of document d.
	 */
	private void sampleTopicAssignment(int d, int n) {
		int old_topic = z[d][n];
		int old_sentiment = y[d][n];
		int word = docs[d][n];
		updateCount(d, old_topic, old_sentiment, word, -1);

		double[] p = new double[param.S * param.T];
		for (int s = 0; s < param.S; ++s) {
			for (int t = 0; t < param.T; ++t) {
				p[s * param.T + t] = (nds[d][s] + gamma) / (ndsum[d] + sGamma)
						* (ndst[d][s][t] + param.alpha) / (nds[d][s] + tAlpha)
						* (nstw[s][t][word] + param.beta)
						/ (nstsum[s][t] + vBeta);
			}
		}

		int pairIndex = InverseTransformSampler.sample(p,
				randomGenerator.nextDouble());
		int new_topic = pairIndex % param.T;
		int new_sentiment = pairIndex / param.T;
		z[d][n] = new_topic;
		y[d][n] = new_sentiment;
		updateCount(d, new_topic, new_sentiment, word, +1);
	}

	/**
	 * Update the counts in the Gibbs sampler.
	 */
	private void updateCount(int d, int topic, int s, int word, int flag) {
		nds[d][s] += flag;
		ndsum[d] += flag;
		ndst[d][s][topic] += flag;
		nstw[s][topic][word] += flag;
		nstsum[s][topic] += flag;
	}

	// ------------------------------------------------------------------------
	// Posterior Distribution Computation
	// ------------------------------------------------------------------------

	/**
	 * After burn in phase, update the posterior distributions every sample lag.
	 */
	private void updatePosteriorDistribution() {
		for (int d = 0; d < param.D; ++d) {
			for (int s = 0; s < param.S; ++s) {
				pisum[d][s] += (nds[d][s] + gamma) / (ndsum[d] + sGamma);
			}
		}

		try {
			for (int d = 0; d < param.D; ++d) {
				for (int s = 0; s < param.S; ++s) {
					for (int t = 0; t < param.T; ++t) {
						thetasum[d][s][t] += (ndst[d][s][t] + param.alpha)
								/ (nds[d][s] + tAlpha);
					}
				}
			}
		} catch (Exception ex) {
			ex.printStackTrace();
		}

		for (int s = 0; s < param.S; ++s) {
			for (int t = 0; t < param.T; ++t) {
				for (int w = 0; w < param.V; ++w) {
					phisum[s][t][w] += (nstw[s][t][w] + param.beta)
							/ (nstsum[s][t] + vBeta);
				}
			}
		}
		++numstats;
	}

	/**
	 * Compute the posterior distributions.
	 */
	private void computePosteriorDistribution() {
		computeDocumentSentimentDistribution();
		computeDocumentSentimentTopicDistribution();
		computeSentimentTopicWordDistribution();
	}

	/**
	 * Document-sentiment distribution: pi[][].
	 */
	private void computeDocumentSentimentDistribution() {
		if (param.sampleLag > 0) {
			for (int d = 0; d < param.D; ++d) {
				for (int s = 0; s < param.S; ++s) {
					pi[d][s] = pisum[d][s] / numstats;
				}
			}
		} else {
			for (int d = 0; d < param.D; ++d) {
				for (int s = 0; s < param.S; ++s) {
					pi[d][s] = (nds[d][s] + gamma) / (ndsum[d] + sGamma);
				}
			}
		}
	}

	/**
	 * Document-sentiment-topic distribution: theta[][].
	 */
	private void computeDocumentSentimentTopicDistribution() {
		if (param.sampleLag > 0) {
			for (int d = 0; d < param.D; ++d) {
				for (int s = 0; s < param.S; ++s) {
					for (int t = 0; t < param.T; ++t) {
						theta[d][s][t] = thetasum[d][s][t] / numstats;
					}
				}
			}
		} else {
			for (int d = 0; d < param.D; ++d) {
				for (int s = 0; s < param.S; ++s) {
					for (int t = 0; t < param.T; ++t) {
						theta[d][s][t] = (ndst[d][s][t] + param.alpha)
								/ (nds[d][s] + tAlpha);
					}
				}
			}

		}
	}

	/**
	 * Topic-word distribution: theta[][].
	 */
	private void computeSentimentTopicWordDistribution() {
		if (param.sampleLag > 0) {
			for (int s = 0; s < param.S; ++s) {
				for (int t = 0; t < param.T; ++t) {
					for (int w = 0; w < param.V; ++w) {
						phi[s][t][w] = phisum[s][t][w] / numstats;
					}
				}
			}
		} else {
			for (int s = 0; s < param.S; ++s) {
				for (int t = 0; t < param.T; ++t) {
					for (int w = 0; w < param.V; ++w) {
						phi[s][t][w] = (nstw[s][t][w] + param.beta)
								/ (nstsum[s][t] + vBeta);
					}
				}
			}
		}
	}

	@Override
	public double[][] getTopicWordDistribution() {
		// Flatten the phi array.
		double[][] topicWordDist = new double[param.S * param.T][];
		int index = 0;
		for (int s = 0; s < param.S; ++s) {
			for (int t = 0; t < param.T; ++t) {
				topicWordDist[index++] = phi[s][t];
			}
		}
		return topicWordDist;
	}

	@Override
	public double[][] getDocumentTopicDistrbution() {
		return null;
	}

	@Override
	public double[][] getDocumentSentimentDistribution() {
		return pi;
	}

	@Override
	public double[][][] getDocumentSentimentTopicDistribution() {
		return theta;
	}

	@Override
	public double[][][] getSentimentTopicWordDistribution() {
		return phi;
	}

	/**************************** For JST ******************************/
	@Override
	public Topics getTopics(int twords, int s) {
		return new Topics(this.getTopWordStrsWithProbabilitiesUnderTopics(
				twords, phi[s]));
	}

	// 0: Positive, 1: Neutral, 2: Negative.
	@Override
	public Topics getPositiveTopics(int twords) {
		return this.getTopics(twords, 0);
	}

	@Override
	public Topics getNegativeTopics(int twords) {
		return this.getTopics(twords, param.S - 1);
	}

	@Override
	public Topics getNeutralTopics(int twords) {
		ExceptionUtility.assertAsException(param.S == 3);
		return this.getTopics(twords, 1);
	}
}
