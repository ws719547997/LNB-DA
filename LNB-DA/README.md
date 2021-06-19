## LNB: Lifelong Naive Bayes
### This repo hosts data and code for the paper "Forward and Backward Knowledge Transfer for Sentiment Classification" in ACML 2019. [[Paper](http://proceedings.mlr.press/v101/wang19f.html)]
Below we take previous task evaluation as an example to introduce how to perform our system.

## Requirements
- Integrated Development Environment: IntelliJ IDEA
- Language: JAVA (We performed experiments using JAVA 1.8.0.)

## Installation
1. Clone or download the project lnb;
2. Unzip the downloaded project;
3. Extract folders '.idea' and 'lib' from the compressed file 'third-party-libraries.7z'. (The file 'third-party-libraries.7z' were split into three parts as GitHub has a maximum single-file size limitation.)

Then the project is organizated as follows

    ├── .idea                 <- IntelliJ’s project specific settings files
    ├── classes               <- Project compilation results
    ├── Data
    │   ├── DomainToEvaluate  <- Each domain/task sequence (e.g., S1, ..., S10) in evaluation
    │   ├── Input             <- Data fed into system
    │   ├── Intermediate      <- Training data (if target domain is new domain), test data, and learned knowledge
    │   └── Output            <- Sentiment classification resultes
    │
    ├── lib                   <- Third-party libraries
    ├── resources             <- Stopwords
    ├── src                   <- Source code used in this project (core files)
    ├── LifelongSentimentClassification.iml  <- IntelliJ’s configuration information for modules
    ├── README.md             <- Guide for user(s) to perform this project.

## Usage
1. Build project to create IntelliJ's project folder './bin/', which stores the class files;
2. Run MainEntry (see './src/main/') to produce sentiment classification results on task sequence S1 for previous task evaluation (*The sentiment classification results will be stored in './data/output/SentimentClassificaton/'*.);
3. For other task sequences, please modify the end of line 104 in file CmdOption.java (see './src/main/'), e.g., using "shuffle2.txt", where shuffle2 denotes task sequence S2.

If there are any questions, please let me know. Best regards.
                --- Hao Wang (Email: cshaowang@gmail.com).
