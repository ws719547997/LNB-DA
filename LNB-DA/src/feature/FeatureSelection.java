package feature;

import classifier.ClassifierParameters;
import nlp.Documents;
import utility.ExceptionUtility;
import utility.FileReaderAndWriter;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public abstract class FeatureSelection {
    // The following FeatureIndexer only contain the selected features.
    public FeatureIndexer featureIndexer = null;
    // The following map contains all features (selected and unselected).
    public Map<String, Boolean> mpFeatureStrToSelected = null;

    public List<String> selectedFeatureStrs = null;

    public InformationGain informationGain = null;

    // Construction method: the name is same to class name.
    public FeatureSelection() {
        featureIndexer = new FeatureIndexer();
        mpFeatureStrToSelected = new HashMap<String, Boolean>();
        selectedFeatureStrs = new ArrayList<String>();
    }

    /**
     * Note that we can only select features based on training data as the
     * labels of testing data is unknown.
     */
    public static FeatureSelection selectFeatureSelection(
            Documents trainingDocs, ClassifierParameters param) {
        FeatureSelection featureSelection = null;
        switch (param.featureSelectionSetting) {
            case "InformationGain":
                return new FeatureSelectionByInformationGain(trainingDocs,
                        param.noOfSelectedFeatures);
            case "TestOfProportion":
                return new FeatureSelectionByTestOfProportion(trainingDocs,
                        param.featureSelectionSignificanceLevel);
            case "ChiSquare":
                return new FeatureSelectionByChiSquareStatistics(trainingDocs,
                        param.featureSelectionSignificanceLevel);
            case "NoSelection":
                return new FeatureSelectionNoSelection(trainingDocs);
            default:
                ExceptionUtility
                        .throwAndCatchException("The feature selection setting is not recognizable!");
                break;
        }
        return featureSelection;
    }


    // /**
    // * In some case, we may create an instance of FeatureSelection when we
    // * already know which feature is selected.
    // *
    // * @param mpFeatureStrToSelected
    // */
    // public FeatureSelection(Map<Feature, Boolean> mpFeatureIdToSelected2) {
    // mpFeatureIdToNewFeatureIndexAfterFeatureSelection = new HashMap<Integer,
    // Integer>();
    // mpFeatureStrToNewFeatureIndexAfterFeatureSelection = new HashMap<String,
    // Integer>();
    // mpNewFeatureIdToFeaturStr = new HashMap<Integer, String>();
    // mpFeatureStrToSelected = new HashMap<String, Boolean>();
    // topFeatures = new ArrayList<Feature>();
    //
    // int newFeatureId = 1;
    // for (Map.Entry<Feature, Boolean> entry : mpFeatureIdToSelected2
    // .entrySet()) {
    // Feature feature = entry.getKey();
    // int featureId = feature.featureId;
    // String featureStr = feature.featureStr;
    // boolean isSelected = entry.getValue();
    // if (isSelected) {
    // mpFeatureIdToNewFeatureIndexAfterFeatureSelection.put(
    // featureId, newFeatureId);
    // mpFeatureStrToNewFeatureIndexAfterFeatureSelection.put(
    // featureStr, newFeatureId);
    // mpNewFeatureIdToFeaturStr.put(newFeatureId, featureStr);
    // ++newFeatureId;
    // topFeatures.add(feature);
    // }
    // mpFeatureStrToSelected.put(featureStr, isSelected);
    // }
    // }

    public List<String> getSelectedFeatureStrs() {
        return selectedFeatureStrs;
    }

    public void printSelectedFeaturesToFile(String filepath) {
        StringBuilder sbFeatures = new StringBuilder();
        sbFeatures.append("Feature\tInformationGain\tP(+|f)\tP(-|f)");
        sbFeatures.append(System.lineSeparator());
        for (String featureStr : selectedFeatureStrs) {
            if (Double.isNaN(informationGain
                    .getProbOfPositiveGivenFeatureStr(featureStr))) {
                System.out.println("Nan");
            }
            sbFeatures.append(featureStr
                    + "\t"
                    + informationGain.getIGGivenFeatureStr(featureStr)
                    + "\t"
                    + informationGain
                    .getProbOfPositiveGivenFeatureStr(featureStr)
                    + "\t"
                    + informationGain
                    .getProbOfNegativeGivenFeatureStr(featureStr));
            sbFeatures.append(System.lineSeparator());
        }
        FileReaderAndWriter.writeFile(filepath, sbFeatures.toString());
    }

    public int sizeOfSelectedFeatures() {
        return selectedFeatureStrs.size();
    }

    public boolean isFeatureSelected(String featureStr) {
        if (!mpFeatureStrToSelected.containsKey(featureStr)) {
            // The feature does not appear in the training data before.
            return false;
        }
        return mpFeatureStrToSelected.get(featureStr);
    }

    public int getFeatureIdGivenFeatureStr(String featureStr) {
        ExceptionUtility.assertAsException(
                featureIndexer.containsFeatureStr(featureStr),
                "The feature is not in the map!");
        return featureIndexer.getFeatureIdGivenFeatureStr(featureStr);
    }
}
