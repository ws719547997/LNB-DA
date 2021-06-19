package feature;

public class Feature implements Comparable<Feature>, Cloneable {
	// We do not store feature id here. In the classifier, we will build a map
	// from featurestr to feature id.
	// public int featureId = 0;
	public String featureStr = null;
	public double featureValue = 0.0;
	public int frequency = 0;
	public double informationGain = 0.0;
	// P(+|f).
	public double positiveGivenFeature = 0.0;
	// P(-|f).
	public double negativeGivenFeature = 0.0;;

	public Feature(String featureStr2) {
		featureStr = featureStr2;
	}

	@Override
	protected Object clone() throws CloneNotSupportedException {
		return (Feature) super.clone();
	}

	public Feature getDeepClone() {
		try {
			return (Feature) this.clone();
		} catch (Exception ex) {
			ex.printStackTrace();
		}
		return null;
	}

	@Override
	public boolean equals(Object o) {
		if (o instanceof Feature) {
			Feature feature = (Feature) o;
			return this.featureStr.equals(feature.featureStr);
		}
		return false;
	}

	@Override
	public int hashCode() {
		return this.featureStr.hashCode();
	}

	@Override
	public int compareTo(Feature feature) {
		return this.featureStr.compareTo(feature.featureStr);
	}

	@Override
	public String toString() {
		return featureStr;
	}

}
