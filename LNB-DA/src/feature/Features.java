package feature;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class Features implements Iterable<Feature> {
	public List<Feature> featureList = null;

	public Features getDeepClone() {
		Features clone = new Features();
		for (Feature feature : this.featureList) {
			clone.addFeature(feature.getDeepClone());
		}
		return clone;
	}

	// construction method,
	public Features() {
		featureList = new ArrayList<Feature>(); // dynamic array
	}

	@Override
	public Iterator<Feature> iterator() {
		return featureList.iterator();
	}

	public void addFeature(Feature feature) {
		this.featureList.add(feature);
	}

	public void addFeatures(Features features) {
		this.featureList.addAll(features.featureList);
	}

	public Feature getFeature(int j) {
		return featureList.get(j);
	}

	public int size() {
		return featureList.size();
	}

	@Override
	public String toString() {
		StringBuilder sbFeatures = new StringBuilder();
		for (Feature feature : this.featureList) {
			sbFeatures.append(feature);
			sbFeatures.append(" ");
		}
		return sbFeatures.toString().trim();
	}

}
