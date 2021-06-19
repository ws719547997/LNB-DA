package feature;

import java.util.Map;
import java.util.HashMap;

import utility.ExceptionUtility;

public class FeatureIndexer {
	public Map<String, Integer> mpFeatureStrToFeatureId = null;
	public Map<Integer, String> mpFeatureIdToFeatureStr = null;

	public FeatureIndexer() {
		mpFeatureIdToFeatureStr = new HashMap<Integer, String>();
		mpFeatureStrToFeatureId = new HashMap<String, Integer>();
	}

	/**
	 * Feature Id starts from 1.
	 */
	public void addFeatureStrIfNotExist(String featureStr) {
		if (mpFeatureStrToFeatureId.containsKey(featureStr)) {
			return;
		}
		// Feature Id starts with 1 due to SVMLight.
		int featureId = mpFeatureStrToFeatureId.size() + 1;
		mpFeatureStrToFeatureId.put(featureStr, featureId);
		mpFeatureIdToFeatureStr.put(featureId, featureStr);
	}

	/**
	 * Feature Id starts from 0.
	 */
	public void addFeatureStrIfNotExistStartingFrom0(String featureStr) {
		if (mpFeatureStrToFeatureId.containsKey(featureStr)) {
			return;
		}
		int featureId = mpFeatureStrToFeatureId.size();
		mpFeatureStrToFeatureId.put(featureStr, featureId);
		mpFeatureIdToFeatureStr.put(featureId, featureStr);
	}

	/**
	 * Feature Id starts from 1.
	 */
	public int getFeatureIdOtherwiseAddFeatureStr(String featureStr) {
		if (!mpFeatureStrToFeatureId.containsKey(featureStr)) {
			addFeatureStrIfNotExist(featureStr);
		}
		return mpFeatureStrToFeatureId.get(featureStr);
	}

	/**
	 * Feature Id starts from 0.
	 */
	public int getFeatureIdOtherwiseAddFeatureStrStartingFrom0(String featureStr) {
		if (!mpFeatureStrToFeatureId.containsKey(featureStr)) {
			addFeatureStrIfNotExistStartingFrom0(featureStr);
		}
		return mpFeatureStrToFeatureId.get(featureStr);
	}

	public int getFeatureIdGivenFeatureStr(String featureStr) {
		ExceptionUtility.assertAsException(this.containsFeatureStr(featureStr),
				"The feature str is not in the map!");
		return mpFeatureStrToFeatureId.get(featureStr);
	}

	public String getFeatureStrGivenFeatureId(int featureId) {
		ExceptionUtility.assertAsException(this.containsFeatureId(featureId),
				"The feature id is not in the map!");
		return mpFeatureIdToFeatureStr.get(featureId);
	}

	public boolean containsFeatureId(int featureId) {
		return mpFeatureIdToFeatureStr.containsKey(featureId);
	}

	public boolean containsFeatureStr(String featureStr) {
		return mpFeatureStrToFeatureId.containsKey(featureStr);
	}

	public int size() {
		return mpFeatureStrToFeatureId.size();
	}

}
