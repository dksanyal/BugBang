#BugBang

#Software name and version
swname <- "BugBang_rel_v4.0.R"
#Script to classify software issue reports using ML tools

#Authors: Debarshi Kumar Sanyal, Nitish Pandey
#Date: 09.02.2017

#CHANGE LOG (version 1.6)
#v1.1:              Cleaned up dtm preparation using funcs <- list(content_transformer(tolower), removePunctuation,  ...)
#v1.2:              Dimensions of DTM logged
#v1.3:              Adding custom stopwords
#v1.4:              Support for customized DTM pruning
#v1.5:              Temporary/experimental fix. (to keep stopwords)
#v1.6:              Fix stopword list. Fixed MAX_TERMS_IN_DTM to 60. k-fold cv in tune.svm(). 
#                   Support for logging running time of the script.
#v1.7:              Updated stopword list. Removed c("with", "for") from stopword list.
#v1.8:              Defined and added measures for NUG class. 
#                   Rolled back changes of v1.7 due to accuracy fall.
#v1.9:              Commenting out nnet and testing for larger number of features.
#v1.10:             Changed log formats. Log tags: N < P < S (increasing order of priority)
#v1.11:             Pass commandline parameters
#v1.12:             Moved commandline parser to top
#v1.13:             Introduced k-fold cross validation
#v1.16:             Output path, multiple cv
#v1.19:  16-02-2017 Multiple cv revoked, constant parameters for SVM
#v1.20:  16-02-2017 Multiple cv reinstated, seed added for RF
#v1.21:  16-02-2017 Support for normalization
#v2.0:   17-02-2017 Support for multiple normalization transforms
#v2.1:   17-02-2017 Support for purging runs with NaN
#v2.2:   20-02-2017 Changed computation of predictive measures according to
#                   'Forman, George, and Martin Scholz. "Apples-to-apples in cross-validation 
#                    studies: pitfalls in classifier performance measurement." ACM SIGKDD 
#                    Explorations Newsletter 12.1 (2010): 49-57.'
#                   NaNs do not affect result
#v2.4:   20-02-2017 Support for stratified sampling
#v2.5:   21-02-2017 Minor changes to stratification function. Added paths as commandline options.
#                   Fixed division-by-zero issues in tfNormalize.
#v2.6:   21-02-2017 Formatting updates.
#v2.7:   22-02-2017 Due to text preprocessing, some entries may be left with null summaries.
#                   Add functionality to remove these entries from the input.
#                   Added error checks for missing input files and empty training & test sets
#v3.1:   27-02-2017 Added severity as a feature for classification
#v3.2:   27-02-2017 More log support
#v3.3:   27-02-2017 Sparsity as fraction supported
#v3.4:   28-02-2017 Using full datasets for the 3 JIRA-based projects as provided by Prof. Hata
#v3.5:   28-02-2017 MAX_TERMS_IN_DTM can be a fraction of the total number of terms
#v3.6:   07-03-2017 Changed field names in input csv files
#v4.0:   16-03-2017 Added options to enter input and output file names through commandline
#Time the running
log.starttime <- proc.time();

#Create list of command line option list
library(optparse)
option_list <- list(
  make_option(c("-d", "--max_terms_in_dtm"), type="numeric", default=100,
    help="Retain specified number of terms in DTM [default=%default]"),
  make_option(c("-t", "--max_iter"), type="integer", default=1,
    help="Deprecated: Number of times to run cv [default=%default]"),
  make_option(c("-n", "--normalize"), type="integer", default=1,
    help="Transform function for normalization (1:None, 2:cosine length, 3:log damping, 4:log damping + cosine length, 5:maxtf, 6:maxtf + cosine length, 7:sumtf). [default=%default]"),
  make_option(c("-k", "--cv_fold"), type="integer", default=10,
    help="Number of folds for cross-validation for classifier [default=%default]"),
  make_option(c("-i", "--infile"), type="character", default="./datasets/exp1/bug_http.csv",
    help="Input csv file name (with path) [default=%default]"),
  make_option(c("-o", "--outfile"), type="character", default="./out/issue_class_out.txt",
    help="Output file name (with path) [default=%default]"));


#Parse options
parser <- OptionParser(usage="%prog [options] file", option_list=option_list);
opts <- parse_args(parser);



#suppressPackageStartupMessages({
library(RTextTools)
library(tm)
library(plyr)
library(class)
library(kernlab)
library(gam)
library(e1071)
library(elmNN)
library(fRegression)
library(frbs)
library(randomForest)
library(rpart)
library(rpart.plot)
library(C50)
library(nnet)
library(randomForest)
library(MASS)
library(optparse)
library(caTools)
#})



#Create stratified folds for CV
getStratifiedFolds <- function( dataset, k, seed=RANDOM_SEED) {

    test <- list();
    training <- list();
    accm     <- data.frame();
    kcount = nrow(dataset)/k;

    for(i in 1:(k-1)) {
        set.seed(seed);
        #Select rows from remaining rows
        selectedRows  <- sample.split(dataset$CLASSIFIED, SplitRatio=kcount); #returns a logical vector. 
                                      #TRUE == selected row, FALSE == not selected row
        
        test[[i]] <- dataset[selectedRows, ];

        #Initialize training[i]
        training[[i]] <- data.frame();
        #Collect the previous testsets
        if(i>1) { 
             training[[i]] <- rbind(training[[i]], accm);
        }
        training[[i]] <- rbind(training[[i]], dataset[!selectedRows, ]);

        accm          <- rbind(accm, test[[i]]);
        dataset <- dataset[!selectedRows, ];

    }
    i <- k;   

    test[[i]] <- dataset[, ];

    #Initialize training[i]
    training[[i]] <- data.frame();
    #Collect the previous testsets
    if(i>1) {
        training[[i]] <- rbind(training[[i]], accm);
    }
    
    folds <- list(test=test, training=training);
    return (folds);
}


#Get the (TP, FN, FP, TN) from confusion matrix
getConfusionMeasures <- function(tabl) {
    tp <- tabl[1,1];
    fn <- tabl[1,2];  #Type-II error
    fp <- tabl[2,1];  #Type-I  error
    tn <- tabl[2,2];

    return (c(tp, fn, fp, tn));

}


#Compute precision and recall for both classes
computePrecisionRecall <- function(tp, fn, fp, tn) {

    prec_bug <- tp / (tp + fp);
    recl_bug <- tp / (tp + fn);

    prec_nug <- tn / (tn + fn);
    recl_nug <- tn / (tn + fp);
    return (c(prec_bug, recl_bug, prec_nug, recl_nug));

}


#Compute F-measure for both classes
computeFMeasure <- function( tp, fn, fp, tn ) {
    f_bug <- 2*tp / (2*tp + fp + fn);
    f_nug <- 2*tn / (2*tn + fn + fp);
    return (c(f_bug, f_nug));

}


#Write log
writeLog <- function( obj ) {
     write(obj, file=OUTCONN, ncolumns=50, append=TRUE);
}


#Generate values for classifier predictors
analyzeResult <- function( actual, predicted, result ) {
    t <- table(ACTUAL = actual, PREDICTED = predicted);
    cm <- getConfusionMeasures(t) #computePrecisionRecall(t);
    pr <- computePrecisionRecall(cm[1], cm[2], cm[3], cm[4]);
    acc <- recall_accuracy(actual, predicted);

    #Log results
    #log.info <- c("Logging failed cases");
    #writeLog(log.info);
    ##log.info <- c("[N] Key, Priority, Summary, Actual, Predicted");
    #log.info <- c("[N] Key, Summary, Actual, Predicted");
    #writeLog(log.info);
    #r        <- data.frame(Key=testing.data.full[,2], Summary=testing.data.full[,3], ActualType=testing.data.full[,1],  PredictedType=predicted)
    #r        <- r[r$ActualType!=r$PredictedType, ]
    #write.table(r, OUTFILE, append = TRUE);


    log.info <- c("[N] [BUG:] Precision=", pr[1], ", Recall=", pr[2]);
    writeLog(log.info);
    log.info <- c("[N] [NUG:] Precision=", pr[3], ", Recall=", pr[4], ", Accuracy=", acc);
    writeLog(log.info);

    writeLog("[N] Confusion Matrix");
    write.table(t, file=OUTCONN, append = TRUE);

    if(is.nan(pr[1]) || is.nan(pr[2]) || is.nan(pr[3]) || is.nan(pr[4])) {
        # Purge this result
        writeLog("[W] Got NaN for precision or recall");     

    } else {
        result[6] <- result[6] + 1;   #Number of non-NaN runs
    } 

    result[1] <- result[1] + cm[1]; #tp sum 
    result[2] <- result[2] + cm[2]; #fn sum 
    result[3] <- result[3] + cm[3]; #fp sum 
    result[4] <- result[4] + cm[4]; #tn sum
    result[5] <- result[5] + acc;   #accuracy sum
    
    return(result);
}

#Normalize DTM
tfNormalize <- function(m, t=1, alpha=0) {
    if(t==1) {
        #Use raw term frequencies
    } else if (t==2) {
        m <-t(apply(m, 1, function(x) {v <- sum(x*x); if(v==0) {x} else { x/sqrt(v) } })); #normalize by cosine length
    } else if (t==3) {
        m[m != 0] <- 1+log(m[m != 0]); #logarithmic damping of frequencies
    } else if(t==4) {
        m[m != 0] <- 1+log(m[m != 0]); #logarithmic damping of frequencies
        m <-t(apply(m, 1, function(x) {v <- sum(x*x); if(v==0) {x} else { x/sqrt(v) } })); #normalize by cosine length
    } else if(t==5) {
        m <- t(apply(m, 1, function(x){v <- max(x); if(v==0) {x} else {alpha + (1-alpha)*x/v}} )); #normalize by maxtf
    } else if(t==6) {
        m <- t(apply(m, 1, function(x){v <- max(x); if(v==0) {x} else {alpha + (1-alpha)*x/v}} )); #normalize by maxtf
        m <-t(apply(m, 1, function(x) {v <- sum(x*x); if(v==0) {x} else { x/sqrt(v) } })); #normalize by cosine length 
    } else if (t==7) {  #normalize by total unencoded doc. length
        r <- rowSums(m);
        m <- m/r[r!=0];
    } else {
    }
    return(m);
}
#Replace common columns
replaceMatchedColumns <- function(dst, src) {
    indx <- match(colnames(dst), colnames(src), nomatch=0);
    for(col in seq(1:ncol(dst))) {
        x <- indx[col];
        if(x!=0) {
           dst[,col] <- src[,x];
        }
    }
    return(dst);
}


#initialize control parameters (filenames and parameters common to all input csv files)
TOTRUN <- opts$max_iter
MAX_TERMS_IN_DTM <- opts$max_terms_in_dtm 
if(MAX_TERMS_IN_DTM > 1) { 
    MAX_TERMS_IN_DTM <- as.integer(MAX_TERMS_IN_DTM);
} 

CV_FOLD <- opts$cv_fold
TESTING_FRAC <- 1/CV_FOLD   #1 out of CV_FOLDs is used for testing, remaining for training
NORMALIZE <- opts$normalize;
INFILE    <- opts$infile;
FILENAMES <- c(INFILE);
OUTFILE   <- opts$outfile;
#INFILE_PATH <- opts$in_path;
#INFILE_PATH <- paste(INFILE_PATH,"/", sep="");
#OUTFILE_PATH <- opts$out_path;
#OUTFILE_PATH <- paste(OUTFILE_PATH,"/", sep="");

TERMS_DTM <- 0

#Fixed parameters
RANDOM_SEED  <- 643;
SVM_CROSS_VALIDATE <- 0;
REMOVE_STOPWORDS <- TRUE;
#FILENAMES <- c("bug_http.csv", "bug_lucene.csv", "bug_jack.csv");
#FILENAMES <- c("full_all.csv");

#curTime <- format(Sys.time(),"__%a_%b_%d_%H_%M_%S_%Z_%Y");
#OUTFILE_PREFIX <- c("norm_", NORMALIZE, "dtm_", MAX_TERMS_IN_DTM);
#OUTFILE_PREFIX <- capture.output(cat(OUTFILE_PREFIX, sep=""));
#OUTFILE_NAME <- paste(OUTFILE_PREFIX, "_results_", sep="");
#OUTFILE_NAME <- paste(OUTFILE_NAME, curTime, sep="");
#OUTFILE_NAME <- paste(OUTFILE_NAME, ".txt", sep="");
#OUTFILE <- paste(OUTFILE_PATH, OUTFILE_NAME, sep="");

#Prepare the list of stopwords
stopwords.custom <- c(
  "i",          "me",         "my",         "myself",     "we",         
  "our",        "ours",       "ourselves",  "you",        "your",      
  "yours",      "yourself",   "yourselves", "he",         "him",       
  "his",        "himself",    "she",        "her",        "hers",      
  "herself",    "it",         "its",        "itself",     "they",      
  "them",       "their",      "theirs",     "themselves", "who",       
  "whom",       "this",       "that",       "these",      "those",     
  "am",         "is",         "are",        "was",        "were",      
  "be",         "been",       "being",      "have",       "has",       
  "had",        "having",     "do",         "does",       "did",       
  "doing",      "i'm",        "you're",     "he's",       "she's",     
  "it's",       "we're",      "they're",    "i've",       "you've",    
  "we've",      "they've",    "i'd",        "you'd",      "he'd",      
  "she'd",      "we'd",       "they'd",     "i'll",       "you'll",    
  "he'll",      "she'll",     "we'll",      "they'll",    "let's",     
  "that's",     "who's",      "what's",     "here's",     "there's",   
  "when's",     "where's",    "why's",      "how's",      "a",         
  "an",         "the",        "and",        "or",         "because",   
  "as",         "of",         "at",         "by",         "for",       
  "with",       "about",      "against",    "between",    "into",      
  "through",    "above",      "below",      "to",         "from",      
  "up",         "down",       "in",         "out",        "on",        
  "off",        "over",       "under",      "further",    "once",      
  "here",       "there",      "all",        "both",       "each",      
  "few",        "more",       "other",      "such",       "only",      
  "own",        "so",         "than",       "too",        "very"      
);

#List of functions to preprocess corpus
skipWords <- function(x) removeWords(x, stopwords.custom)
funcs <- list(content_transformer(tolower), removePunctuation, removeNumbers, skipWords, stripWhitespace, stemDocument);

#Create output file
OUTCONN <- file(OUTFILE, open="wt", encoding = "UTF-8");

#Put some headers in output file
writeLog(c("[P]", swname));
log.info <- c("[P] CV_FOLD = ", CV_FOLD, ", TESTING_FRAC = ", TESTING_FRAC, ", MAX_TERMS_IN_DTM = ", MAX_TERMS_IN_DTM, ", SVM_CROSS_VALIDATE = ", SVM_CROSS_VALIDATE, ", REMOVE_STOPWORDS = ", REMOVE_STOPWORDS, ", RANDOM_SEED = ", RANDOM_SEED, ", NORMALIZE (1:None, 2:cosine length, 3:log damping, 4:log damping + cosine length, 5:maxtf, 6:maxtf + cosine length, 7:sumtf) =  ", NORMALIZE, ", INFILE_PATH = ", INFILE, ", OUTFILE_PATH", OUTFILE);

writeLog(log.info);

for( filename in FILENAMES) {

    #Create full path to input file
    #filename.full <- paste(INFILE_PATH, filename, sep="");
    filename.full   <- filename;
    filename.proper <- basename(filename);
    writeLog("******************************************************");
    log.info <- c("[P] Bug DB = " , filename.proper, ", Full path = ", filename.full);
    writeLog(log.info);
    writeLog("******************************************************");
    if(!file.exists(filename.full)) {
        log.info <- c("[E] File ", filename.full, " does not exist. Skipping.");
         writeLog(log.info);
         next;
    } 
    
    #initilize outputs
    nb.result <- c(0.0, 0.0, 0.0, 0.0, 0.0,0.0);
    knn.result <- c(0.0, 0.0, 0.0, 0.0, 0.0,0.0);
    lda.result <- c(0.0, 0.0, 0.0, 0.0, 0.,0.0);
    svm.rbf.result <- c(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    svm.linear.result <- c(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    svm.poly2.result <- c(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    svm.poly3.result <- c(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    svm.sigmoid.result <- c(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    tree.result <- c(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    forest.result <- c(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    #prepare input 
    bugDataRaw <- read.csv(file=filename.full,header=TRUE,sep=",", stringsAsFactors = FALSE);
    bugData <- bugDataRaw[ bugDataRaw$CLASSIFIED %in% c("BUG", "NUG"), ]
    num.tot.orig <- nrow(bugData);

    #It may happen that no terms are left in a summary after preprocessing.
    #Such cases are considered as outliers and ignored from input.
    bugData.corpus     <- Corpus(VectorSource(bugData$SUMMARY));
    bugData.corpus     <- tm_map(bugData.corpus, FUN = tm_reduce, tmFuns = funcs);
    bugData.dtm        <- DocumentTermMatrix(bugData.corpus);
    bugData.dtm.m      <- as.matrix(bugData.dtm);
    rsum               <- rowSums(bugData.dtm.m);
    takeRows           <- (rsum>0);
    bugData <- bugData[ takeRows ,  ];
    num.tot <- nrow(bugData);
    if(num.tot.orig != num.tot) {
        num.removed <- num.tot.orig - num.tot;
        info.log <- c("[N] Found empty summary after preprocessing. Removing ", num.removed, "rows");
        writeLog(info.log);
    } 
    
  
    #Find number of bugs and nugs in input data
    num.bugs   <- nrow(bugData[ bugDataRaw$CLASSIFIED == "BUG", ]);
    num.nugs   <- nrow(bugData[ bugDataRaw$CLASSIFIED == "NUG", ]);
    bug.fraction <- (num.bugs / num.tot );
    nug.fraction <- (num.nugs / num.tot );

    info.log <- c("[N] Effective data distribution in Bug DB: TOT_ISSUES = ", num.tot, ", NUM_BUGS = " , num.bugs, ", NUM_NUGS (Non-bugs) = ", num.nugs);
    writeLog(info.log);

    bugData.row.count <- nrow(bugData);

 #Do repeated cross-validation
 for(iter in 1:TOTRUN) {

    #Set seed. Then do a random permutation of row indices for fold creation
    seed <- RANDOM_SEED * iter;
    info.log <- c("[N] Outer iteration: ", iter, ", Seed for CV stratified sampling: ", seed);
    writeLog(info.log);

    #Set seed. Then do a random permutation of row indices for fold creation
    sfolds <- getStratifiedFolds(bugData, CV_FOLD, seed);
    stest  <- sfolds$test;
    straining <- sfolds$training;

    for(i in 1:CV_FOLD) {

        info.log <- c("[N] Inner iteration (CV): ", i);
        writeLog(info.log); 
        testing.data.input  <- stest[[i]];

        training.data.input <- straining[[i]];
        #Exclude data with missing types
        training.data.input <- training.data.input [ training.data.input$CLASSIFIED %in% c("BUG", "NUG"), ]

        #Check for zero-sized training and test sets
        ntest  <- nrow(testing.data.input);
        ntrain <- nrow(training.data.input);
        info.log <- c("[N] Training.nrows = ", ntrain, ", Testing.nrows = ", ntest);
        writeLog(info.log);
        if(ntrain == 0 || ntest == 0) {
           info.log <- c("[W] Zero sized dataset found. Skipping iteration.");
           writeLog(info.log);
           next;
        } 
        

        #Log data distribution in training data
        Train.Type     <- as.factor(training.data.input$CLASSIFIED);  
        #Train.Type     <- as.factor(training.data.input$TYPE);  #Take the JIRA Type for experiment 2
        Key      <- as.factor(training.data.input$ID);
        info.log <- "[N] Data distribution in training data"
        writeLog(info.log);
        t        <- count(Train.Type)
        write.table(t, file=OUTCONN, append = TRUE);

        #Prepare DTM for training data
        training.data.corpus     <- Corpus(VectorSource(training.data.input$SUMMARY)); 
        training.data.corpus     <- tm_map(training.data.corpus, FUN = tm_reduce, tmFuns = funcs);
        training.data.dtm        <- DocumentTermMatrix(training.data.corpus);
        training.data.dtm.m <- as.matrix(training.data.dtm);
        training.data.dtm.m <- tfNormalize(training.data.dtm.m, t=NORMALIZE);
        training.data.dtm.dim    <- dim(training.data.dtm.m);
        info.log <- c("[N] Dimensions of training DTM before pruning: ", training.data.dtm.dim[1], " docs X ", training.data.dtm.dim[2], " terms");
        writeLog(info.log);
        #Prune DTM
        if( MAX_TERMS_IN_DTM <= 1 ) {
            TERMS_DTM = MAX_TERMS_IN_DTM * training.data.dtm.dim[2];
        } else {
            TERMS_DTM = MAX_TERMS_IN_DTM;
        }
        training.data.dtm.freq <- sort(colSums(training.data.dtm.m), decreasing=TRUE);
        training.data.dtm.wf <- data.frame(word=names(training.data.dtm.freq), freq=training.data.dtm.freq);
        training.data.dtm.maxterms <- min(TERMS_DTM, nrow(training.data.dtm.wf));
        training.data.dtm.m <- training.data.dtm.m[, colnames(training.data.dtm.m) %in% training.data.dtm.wf[1:training.data.dtm.maxterms,1]]
        
        training.data.dtm.dim    <- dim(training.data.dtm.m);
        info.log <- c("[N] Dimensions of training DTM after pruning: ", training.data.dtm.dim[1], " docs X ", training.data.dtm.dim[2], " terms");
        writeLog(info.log);
        training.data.dictionary <- colnames(training.data.dtm.m);

        #training.data.full <- data.frame(Train.Type, Key, Priority, training.data.dtm.m);
        #Priority <- as.numeric(Priority)/5;
        training.data.full <- data.frame(Train.Type, Key, training.data.dtm.m);
        training.data <- data.frame(Train.Type, training.data.dtm.m);
        #training.data <- data.frame(Train.Type, Priority, training.data.dtm.m);
        #training.data <- data.frame(Train.Type, training.data.dtm.m);


        #Log data distribution in test data
        Test.Type     <- as.factor(testing.data.input$CLASSIFIED);


        info.log <- "[N] Data distribution in test data"
        writeLog(info.log);
        t        <- count(Test.Type)
        write.table(t, file=OUTCONN, append = TRUE);

        #Prepare DTM for test data
        testing.data.corpus     <- Corpus(VectorSource(testing.data.input$SUMMARY));
        testing.data.corpus     <- tm_map(testing.data.corpus, FUN = tm_reduce, tmFuns = funcs);

        testing.data.dtm.full   <- DocumentTermMatrix(testing.data.corpus);
        testing.data.dtm.full.m <- as.matrix(testing.data.dtm.full);
        testing.data.dtm.full.m <- tfNormalize(testing.data.dtm.full.m, t=NORMALIZE);

        testing.data.dtm        <- DocumentTermMatrix(testing.data.corpus, control = list(dictionary=training.data.dictionary));
        testing.data.dtm.m      <- as.matrix(testing.data.dtm);
        #Now replace the common columns in testing.data.dtm.m with columns in testing.data.dtm.full.m
        testing.data.dtm.m <- replaceMatchedColumns(testing.data.dtm.m, testing.data.dtm.full.m);
        Key      <- as.factor(testing.data.input$ID);
        #Priority <- as.factor(testing.data.input$PRIORITY);
        Summary  <- testing.data.input$SUMMARY
        #testing.data.full       <- data.frame(Test.Type, Key, Priority, Summary);
        testing.data.full       <- data.frame(Test.Type, Key, Summary);
        #Priority <- as.numeric(Priority)/5;
        testing.data            <- data.frame(Test.Type, testing.data.dtm.m);


        #NAIVE BAYES
        writeLog("[N] NAIVE BAYES");
        nb.model <- naiveBayes( formula=Train.Type ~., data=training.data, laplace=1 );
        nb.predicted <- predict( nb.model, testing.data[, -1] );
        nb.result <- analyzeResult( testing.data[,1], nb.predicted, nb.result ); 
        #table(ACTUAL = testing.data[,1], PREDICTED = nb.predicted);

        #K-NEAREST NEIGHBOURS
        #The value for k is generally chosen as the square root of the number of observations.
        writeLog("[N] K-NEAREST NEIGHBOURS");
        train.temp <- training.data;
        #train.temp[,2] <- as.numeric(train.temp[,2]); #Priority
        test.temp  <- testing.data;
        #test.temp[,2]  <- as.numeric(test.temp[,2]); #Priority
        knn.predicted <- knn(train.temp[,-1], test.temp[,-1], cl = training.data[,1], k = 15);
        #knn.predicted <- knn(train = training.data[, c(-1,-2)],test = testing.data[, c(-1,-2)],cl = training.data[,1], k = 15);
        knn.result <- analyzeResult( testing.data[,1], knn.predicted, knn.result );

        #LDA
        writeLog("[N] LDA");
        lda.model <- lda(formula=Train.Type ~., data=training.data);
        lda.predicted <- predict(lda.model, testing.data[, -1])
        lda.result <- analyzeResult( testing.data[,1], lda.predicted$class, lda.result );


        #SVM-RBF
        cost <- 100; gamma <- 1e-04;
        info.log <- c("[N] Best parameters for SVM: cost = ", cost, ", ", "gamma = ", gamma);
        writeLog(info.log);

        writeLog("[N] SVM-RBF");
        svm.rbf.model <- svm(Train.Type ~.,data=training.data[,], kernel="radial", cost=cost, gamma=gamma);
        svm.rbf.predicted <- predict(svm.rbf.model, testing.data[, -1]);
        svm.rbf.result <- analyzeResult( testing.data[,1], svm.rbf.predicted, svm.rbf.result );
        

        #SVM-LINEAR
        writeLog("[N] SVM-LINEAR")
        info.log <- c("[N] Best parameters for SVM-LINEAR: cost = ", cost);
        writeLog(info.log);
        svm.linear.model <- svm(Train.Type ~.,data=training.data, kernel="linear", cost=cost);
        svm.linear.predicted <- predict(svm.linear.model, testing.data[, -1]);
        svm.linear.result <- analyzeResult( testing.data[,1], svm.linear.predicted, svm.linear.result );

        #SVM-POLY-2
        cost <- 100; gamma <- 1e-02;
        writeLog("[N] SVM-POLY-2");
        info.log <- c("[N] Best parameters for SVM-POLY-2: cost = ", cost, ", ", "gamma = ", gamma);
        writeLog(info.log);
        svm.poly2.model <- svm(Train.Type ~.,data=training.data, kernel="polynomial", cost=cost, gamma=gamma, degree=2);
        svm.poly2.predicted <- predict(svm.poly2.model, testing.data[, -1]);
        svm.poly2.result <- analyzeResult( testing.data[,1], svm.poly2.predicted, svm.poly2.result );

        #SVM-PLOY-3      #Too slow to train
        writeLog("[N] SVM-POLY-3");
        info.log <- c("[N] Best parameters for SVM-POLY-3: cost = ", cost, ", ", "gamma = ", gamma);
        writeLog(info.log);
        svm.poly3.model <- svm(Train.Type ~.,data=training.data, kernel="polynomial", cost=cost, gamma=gamma, degree=3);
        svm.poly3.predicted <- predict(svm.poly3.model, testing.data[, -1]);
        svm.poly3.result <- analyzeResult( testing.data[,1], svm.poly3.predicted, svm.poly3.result );

        #SVM-SIGMOID
        writeLog("[N] SVM-SIGMOID");
        cost <- 100; gamma <- 1e-04;
        info.log <- c("[N] Best parameters for SVM-SIGMOD: cost = ", cost, ", ", "gamma = ", gamma);
        writeLog(info.log);
        svm.sigmoid.model <- svm(Train.Type ~.,data=training.data, kernel="sigmoid", cost=cost, gamma=gamma);
        svm.sigmoid.predicted <- predict(svm.sigmoid.model, testing.data[, -1]);
        svm.sigmoid.result <- analyzeResult( testing.data[,1], svm.sigmoid.predicted, svm.sigmoid.result );

        #TREE
        writeLog("[N] TREE-RPART");
        tree.model <- rpart(Train.Type ~.,data=training.data);
        tree.predicted <- predict(tree.model, testing.data[, -1], type="class");
        tree.result <- analyzeResult( testing.data[,1], tree.predicted, tree.result );

        #FOREST
        writeLog("[N] FOREST");
        seed <- RANDOM_SEED * iter +  i;
        info.log <- c("[N] Inner iteration: ", i, ", Seed for RF: ", seed);
        writeLog(info.log);

        set.seed(seed);
        forest.model <- randomForest(Train.Type ~.,data=training.data);
        forest.predicted <- predict(forest.model, testing.data[, -1]);
        forest.result <- analyzeResult( testing.data[,1], forest.predicted, forest.result );


    }
  }
    #compute measure for each classifier
    DIVIDER = CV_FOLD * TOTRUN;  

    results <- list(nb.result, lda.result, knn.result, svm.linear.result, svm.rbf.result, svm.poly2.result, svm.poly3.result, svm.sigmoid.result, tree.result, forest.result);
    resultClassifierNames <- c("nb.result", "lda.result", "knn.result", "svm.linear.result", "svm.rbf.result", "svm.poly2.result", "svm.poly3.result", "svm.sigmoid.result", "tree.result", "forest.result");
    separator <- "   ";    

    iter <- 1;  
    writeLog("******************************************************");
    writeLog(c("[S]               SUMMARY RESULTS: ", filename));
    writeLog("******************************************************");

     
    log.info <- c("[S] Classifier", separator, "BUG F1-score", separator, "Accuracy", separator, "BUG precision", separator, "BUG recall", separator, "AVG F1", separator, "#RUNS");
    writeLog(log.info); 
    cat("\n");
    for( result in results) {
 
        pr              <- computePrecisionRecall(result[1], result[2], result[3], result[4]);
        bug.precision   <- pr[1];
        bug.recall      <- pr[2];
        nug.precision   <- pr[3];
        nug.recall      <- pr[4];

        f               <- computeFMeasure( result[1], result[2], result[3], result[4]);  #average f-measure 
        bug.f           <- f[1];
        nug.f           <- f[2];
        avg.f           <- bug.fraction * bug.f + nug.fraction * nug.f;

        accuracy        <- result[5] / DIVIDER;

        all.result <- c("[S]", resultClassifierNames[iter], separator, bug.f, separator, accuracy, separator, bug.precision, separator, bug.recall, separator, avg.f, separator, result[6]);
        writeLog(all.result);
        iter <- iter + 1;
    }

}

#Log the total time taken
log.runtime <- proc.time() - log.starttime;
log.info <- c("[S] TIME: user: ", log.runtime[1], ", system: ", log.runtime[2], ", elapsed: ", log.runtime[3]);
writeLog(log.info);

#Close the connection to output file
close(OUTCONN);
