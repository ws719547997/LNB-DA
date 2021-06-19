package utility;

import main.Constant;
import nlp.Documents;

import java.util.Random;

public class CrossValidationOperator {
    private Documents shuffledDocuments = null;

    public CrossValidationOperator(Documents documents) {
        shuffledDocuments = shuffleDocuments(documents); //rearrange documents, i.e., disordering
    }

    /**
     * get training and test documents
     * @param folderIndex
     * @param noOfCVFolder
     * @return
     */
    public Pair<Documents, Documents> getTrainingAndTestingDocuments(
            int folderIndex, int noOfCVFolder) {
        Documents trainingDocs = new Documents();
        Documents testingDocs = new Documents();
        int N = shuffledDocuments.size();
        int sizeOfEachFolder = N / noOfCVFolder;
        int lowerBound = folderIndex * sizeOfEachFolder;
        int upperBound = (folderIndex + 1) * sizeOfEachFolder - 1;
        for (int i = 0; i < N; ++i) {
            if (lowerBound <= i && i <= upperBound) {
                testingDocs.addDocument(shuffledDocuments.getDocument(i));
            } else {
                trainingDocs.addDocument(shuffledDocuments.getDocument(i));
            }
        }
        return new Pair<Documents, Documents>(trainingDocs, testingDocs);
    }

    /**
     * rearrange documents, i.e., disordering
     * @param documents
     * @return
     */
    private Documents shuffleDocuments(Documents documents) {
        Documents shuffledDocuments = new Documents();
        int[] shuffledNumbers = getRandomOrderNumbersFrom0ToNminus1(documents
                .size());
        for (int i = 0; i < documents.size(); ++i) {
            shuffledDocuments.addDocument(documents
                    .getDocument(shuffledNumbers[i]));
        }
        return shuffledDocuments;
    }


    /**
     * Generate the array containing the number from 0 to n - 1.
     * @param n
     * @return random(0, n-1)
     */
    public static int[] getRandomOrderNumbersFrom0ToNminus1(int n) {
        int[] numbers = new int[n];
        for (int i = 0; i < n; ++i) {
            numbers[i] = i;
        }

        // if need generate the same random numbers for each time, use the follows
        Random rand = new Random(Constant.RANDOMSEED);
        // Swap more times to get perfect random orders.
        for (int i = 0; i < n << 5; ++i) {
            int pos = rand.nextInt() % n;
            // Attention: pos may be negative.
            if (pos < 0) {
                pos += n;
            }
            int temp = numbers[0];
            numbers[0] = numbers[pos];
            numbers[pos] = temp;
        }
        for (int i = 0; i < n << 5; ++i) {
            int pos = rand.nextInt() % n;
            // Attention: pos may be negative.
            if (pos < 0) {
                pos += n;
            }
            int temp = numbers[n - 1];
            numbers[n - 1] = numbers[pos];
            numbers[pos] = temp;
        }

        // if need generate different random numbers for each time, use the follows
//        Random rand = new Random();
//        // Swap more times to get perfect random orders.
//        for (int i = 0; i < n; ++i) {
//            int pos = rand.nextInt(n);
//            if (pos < 0) {
//                continue;
//            }
//            int temp = numbers[i];
//            numbers[i] = numbers[pos];
//            numbers[pos] = temp;
//        }
        return numbers;
    }
}
