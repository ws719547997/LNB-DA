package statistics;

import org.apache.commons.math3.distribution.ChiSquaredDistribution;
import org.apache.commons.math3.distribution.NormalDistribution;

public class SignificanceTest {
	// private final static double SIGNIFICANT_LEVEL = 0.05;

	/******************************* Test of Proportions ***************************/
	public boolean isProportionSignificantDifferentTwoTailed(double p1,
			double n1, double p2, double n2, double significanceLevel) {
		NormalDistribution distribution = new NormalDistribution();

		// Compute pooled sample proportion;
		double p = (p1 * n1 + p2 * n2) / (n1 + n2);
		// Compute standard error.
		double SE = Math.sqrt(p * (1.0 - p) * ((1.0 / n1) + (1.0 / n2)));
		// Compute z-score.
		double z = (p1 - p2) / SE;
		if (z >= 0) {
			z = -z; // For computer P(x<=z).
		}

		double pValue = distribution.cumulativeProbability(z);
		pValue = pValue * 2.0; // Two tailed significance test.
		return pValue <= significanceLevel;
	}

	public boolean isProportionSignificantDifferentTwoTailed(double p1, int n1,
			double p2, int n2, double significanceLevel) {
		return isProportionSignificantDifferentTwoTailed(p1, 1.0 * n1, p2,
				1.0 * n2, significanceLevel);
	}

	/******************************* Chi Square ***************************/
	public boolean isChiSqurareSignificant(double N, double A, double B,
			double C, double D, double significanceLevel) {
		ChiSquaredDistribution distribution = new ChiSquaredDistribution(1.0);

		double chiSquarePositive = getChiSquareStatistic(N, A, B, C, D);
		double chiSquareNegative = getChiSquareStatistic(N, B, A, D, C);
		double averageChiSquare = (chiSquarePositive + chiSquareNegative) / 2.0;

		double pValue = 1.0 - distribution
				.cumulativeProbability(averageChiSquare);
		return pValue <= significanceLevel;
	}

	public double getChiSquareStatistic(double N, double A, double B, double C,
			double D) {
		return N * (A * D - C * B) * (A * D - C * B)
				/ ((A + C) * (B + D) * (A + B) * (C + D));
	}
}
