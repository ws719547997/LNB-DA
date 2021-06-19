package task;

import java.io.File;
import java.io.FilenameFilter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import utility.FileReaderAndWriter;
import nlp.Document;
import nlp.Documents;
import main.CmdOption;

public class InputReaderTask {
    private CmdOption cmdOption = null;

    public InputReaderTask(CmdOption cmdOption2) {
        cmdOption = cmdOption2;
    }

    /**
     * Read the documents from 100 Positive and 100 Negative domains.
     */
    public List<Documents> readDocumentsListFrom100P100NDomains() {
        List<Documents> documentsList = new ArrayList<Documents>();
        // get domain name list
        List<String> domainList = FileReaderAndWriter.
                readFileAllLines(cmdOption.inputListOfDomainsToEvaluate);

        // get all documents, including pre-processing steps
        documentsList.addAll(Documents
                .readListOfDocumentsFromDifferentDomains(cmdOption.input100P100NReviewAllDirectory, domainList));
        return documentsList;
    }


    // TODO: no need to focus on the following codes, (only focus on the above first part)
    /**
     * Read the documents from 1K natural class distribution 20 domains.
     */
    public List<Documents> readDocumentsListFrom1KReviewsNaturalClassDistributionDomains() {
        List<Documents> documentsList = new ArrayList<Documents>();
        documentsList
                .addAll(Documents
                        .readListOfDocumentsFromDifferentDomains(cmdOption.input1KReviewNaturalClassDistribution));
        return documentsList;
    }

    /**
     * Read the documents from balanced domains
     * with most negative reviews.
     */
    public List<Documents> readDocumentsFromBalancedWithMostNegativeReviews() {
        List<Documents> documentsList = new ArrayList<Documents>();
        documentsList
                .addAll(Documents
                        .readListOfDocumentsFromDifferentDomains(cmdOption.inputBalancedWithMostNegativeReviews));
        return documentsList;
    }

    /**
     * Read the documents from 1K+/1K- domains.
     */
    public List<Documents> readDocumentsListFrom1KP1KNDomains() {
        return Documents
                .readListOfDocumentsFromDifferentDomains(cmdOption.input1KP1KNReviewDirectory);
    }

    /**
     * Read the documents from different products of the same domain.
     */
    public List<Documents> readDocumentsFromDifferentProductsOfSameDomain() {
        List<Documents> documentsList = new ArrayList<Documents>();
        // Go through each domain.
        String[] domainNames = new File(cmdOption.inputSameDomainDifferntProductsDirectoy)
                .list(new FilenameFilter() {
                    @Override
                    public boolean accept(File current, String name) {
                        return new File(current, name).isDirectory();
                    }
                });
        for (String domain : domainNames) {
            String domainDirectoryPath = cmdOption.inputSameDomainDifferntProductsDirectoy
                    + File.separator + domain;
            documentsList
                    .addAll(Documents
                            .readListOfDocumentsFromDifferentDomains(domainDirectoryPath));
        }
        return documentsList;
    }

    /**
     * Read the documents from Pang and Lee movie reviews.
     * 2 domains: document-level and sentence-level.
     */
    public List<Documents> readDocumentsFromPangAndLeeMovieReview() {
        return Documents
                .readListOfDocumentsFromDifferentDomains(cmdOption.inputPangAndLeeReviewsDirectory);
    }

    /**
     * Read the documents from Reuters 10 domains. Each domain works as the
     * positive class while the rest are the negative.
     */
    public List<Documents> readReuters10domains() {
        List<Documents> documentsList = new ArrayList<Documents>();
        documentsList
                .addAll(Documents
                        .readListOfDocumentsFromChineseStockComments(cmdOption.inputstock));
        return documentsList;
    }

    /**
     * Read the documents from 20 News group. Each news group works as the
     * positive class while the rest are the negative.
     */
    public List<Documents> read20Newsgroup() {
        Map<String, Documents> mpDomainToDocuments = new HashMap<String, Documents>();
        List<String> lines = FileReaderAndWriter
                .readFileAllLines(cmdOption.input20NewsgroupFilepath);
        for (int i = 0; i < lines.size(); ++i) {
            String line = lines.get(i);
            String[] strSplits = line.split("\t");
            if (strSplits.length != 2) {
                continue;
            }
            String domain = strSplits[0];
            String content = strSplits[1];
            Document document = new Document(domain, i, content);
            if (!mpDomainToDocuments.containsKey(domain)) {
                mpDomainToDocuments.put(domain, new Documents(domain));
            }
            mpDomainToDocuments.get(domain).addDocument(document);
        }
        List<Documents> documentsOf20NewsgroupOriginal = new ArrayList<Documents>();
        for (Documents documents : mpDomainToDocuments.values()) {
            documentsOf20NewsgroupOriginal.add(documents);
        }
        if (cmdOption.dataVsSetting.equals("One-vs-Rest")) {
            return createDocumentsListOneVsRest(documentsOf20NewsgroupOriginal);
        } else {
            return createDocumentsListOneVsOne(documentsOf20NewsgroupOriginal);
        }
    }

    /**
     * Read the documents from Non-Electronics 50 domains (1000 reviews each).
     * Each domain works as the positive class while the rest are the negative.
     */
    public List<Documents> read50NonElectronicsReviewsOneVsRest() {
        // Non-Electronics.
        List<Documents> documentsOfNonElectronicsReviews = Documents
                .readListOfDocumentsFromDifferentDomains(cmdOption.input50NonElectronics1KReview);
        return createDocumentsListOneVsRest(documentsOfNonElectronicsReviews);
    }

    /**
     * For each domain d, d works as positive while the rest is negative. // OneVsRest
     */
    private List<Documents> createDocumentsListOneVsRest(
            List<Documents> documentsOfAllDomainsOriginal) {
        List<Documents> documentsOfOneVsRest = new ArrayList<Documents>();
        for (int i = 0; i < documentsOfAllDomainsOriginal.size(); ++i) {
            // Positive documents.
            Documents documentsPositive = documentsOfAllDomainsOriginal.get(i)
                    .getDeepClone();
            documentsPositive.assignLabel("+1");

            // Negative documents.
            Documents documentsNegative = new Documents();
            for (int j = 0; j < documentsOfAllDomainsOriginal.size(); ++j) {
                if (j == i) {
                    continue;
                }
                Documents documentsOfThisDomain = documentsOfAllDomainsOriginal
                        .get(j).getDeepClone();
                if (cmdOption.randomlySampleNegativeToBalancedClass) {
                    documentsOfThisDomain = documentsOfThisDomain
                            .selectSubsetOfDocumentsRandomly(documentsPositive
                                    .size()
                                    / (documentsOfAllDomainsOriginal.size() - 1));
                }
                documentsNegative.addDocuments(documentsOfThisDomain);
            }

            documentsNegative.assignLabel("-1");
            documentsNegative.domain = documentsPositive.domain;

            Documents documentsMerged = Documents.getMergedDocuments(
                    documentsPositive, documentsNegative);
            System.out.println(documentsMerged.domain + " "
                    + documentsMerged.getNoOfPositiveLabels() + " "
                    + documentsMerged.getNoOfNegativeLabels());

            documentsOfOneVsRest.add(documentsMerged);
        }
        return documentsOfOneVsRest;
    }

    /**
     * For each domain d, d works as positive while the rest is negative. // OneVsOne
     */
    private List<Documents> createDocumentsListOneVsOne(
            List<Documents> documentsOfAllDomainsOriginal) {
        List<Documents> documentsOfOneVsOne = new ArrayList<Documents>();
        for (int i = 0; i < documentsOfAllDomainsOriginal.size(); ++i) {
            for (int j = i + 1; j < documentsOfAllDomainsOriginal.size(); ++j) {
                // Positive documents.
                Documents documentsPositive = documentsOfAllDomainsOriginal
                        .get(i).getDeepClone();
                documentsPositive.assignLabel("+1");
                // Negative documents.
                Documents documentsNegative = documentsOfAllDomainsOriginal
                        .get(j).getDeepClone();
                documentsNegative.assignLabel("-1");
                String domain = documentsPositive.domain + " VS "
                        + documentsNegative.domain;
                if (cmdOption.positiveNegativeRatio > 0) {
                    documentsPositive = documentsPositive
                            .selectSubsetOfDocumentsRandomly((int) (documentsNegative
                                    .size() * cmdOption.positiveNegativeRatio) + 1);
                }

                documentsPositive.domain = domain;
                documentsNegative.domain = domain;
                Documents documentsMerged = Documents.getMergedDocuments(
                        documentsPositive, documentsNegative);
                documentsOfOneVsOne.add(documentsMerged);
            }
        }
        return documentsOfOneVsOne;
    }

    public List<Documents> readDocumentsFromstock() {
        List<Documents> documentsList = new ArrayList<Documents>();
        documentsList
                .addAll(Documents
                        .readListOfDocumentsFromChineseStockComments(cmdOption.inputstock));
        return documentsList;
    }
}
