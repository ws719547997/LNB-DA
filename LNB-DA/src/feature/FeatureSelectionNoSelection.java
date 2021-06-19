package feature;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import utility.ItemWithValue;
import nlp.Documents;

/*
 * All features are selected.
 */
public class FeatureSelectionNoSelection extends FeatureSelection {
	public FeatureSelectionNoSelection(Documents trainingDocs) {
		// Calculate the information gain of each featureStr (i.e., each word).
		informationGain = new InformationGain(trainingDocs);

		List<ItemWithValue> featureListRankedByIG = new ArrayList<ItemWithValue>();
		for (Map.Entry<String, Double> entry : informationGain.mpFeatureStrToInformationGain
				.entrySet()) {
			String featureStr = entry.getKey();
			double ig = entry.getValue();
			ItemWithValue iwv = new ItemWithValue(featureStr, ig);
			featureListRankedByIG.add(iwv);
		}

		// Rank featureStrs by information gain.
		Collections.sort(featureListRankedByIG);
		for (int i = 0; i < featureListRankedByIG.size(); ++i) {
			ItemWithValue iwv = featureListRankedByIG.get(i);
			String featureStr = iwv.getItem().toString();
			selectedFeatureStrs.add(featureStr);
			featureIndexer.addFeatureStrIfNotExist(featureStr);
			mpFeatureStrToSelected.put(featureStr, true);
		}
	}
}
