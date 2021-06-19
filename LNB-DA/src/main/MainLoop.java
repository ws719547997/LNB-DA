package main;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;

import task.*; // import package task, i.e., import all the classes in task
import classifier.ClassifierParameters;
import task.SVM_Sequence;
import utility.ExceptionUtility;

public class MainLoop {
    public static void main(String[] args) {
        CmdOption cmdOption = new CmdOption();
        // wangsong add:we can setting parameter here through cmdoption.
        cmdOption.ngram = 2;
        cmdOption.attantionMode = "att";
        //"att" "att_max" "att_percent"
        cmdOption.vtMode = "none"; // add
        cmdOption.vkbMode = "none"; //ds

        boolean loop = true;
        int GAMMA = 2;
        int POS = 2;
        int DOM = 12;

        if(loop){
            for (int gamma=6;gamma<=6;gamma++){
                for(int pos=1;pos<=12;pos++){
                    for(int domain=12;domain<=1;domain++){
                        cmdOption.gammaThreshold = gamma;
                        cmdOption.positiveRatioThreshold = pos;
                        cmdOption.domainNumLavege = domain;
                        NaiveBayesSequenceLearningGoBack
                                naiveBayesSequenceGoBack = new NaiveBayesSequenceLearningGoBack(cmdOption);
                        naiveBayesSequenceGoBack.run();
                    }
                }
            }
        }else {
            for (int gamma=GAMMA;gamma<=GAMMA;gamma++){
                for(int pos=POS;pos<=POS;pos++){
                    for(int domain=DOM;domain<=DOM;domain++) {
                        cmdOption.gammaThreshold = gamma;
                        cmdOption.positiveRatioThreshold = pos;
                        cmdOption.domainNumLavege = domain;
                        NaiveBayesSequenceLearningGoBack
                                naiveBayesSequenceGoBack = new NaiveBayesSequenceLearningGoBack(cmdOption);
                        naiveBayesSequenceGoBack.run();
                    }
                }
            }
        }
    }
}
