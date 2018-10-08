# BugBang
This repository contains an R script to classify an issue report from JIRA as referring to a bug or not. It also contains associated datasets.
They are used in the following paper:

Nitish Pandey, Debarshi Kumar Sanyal, Abir Hudait, and Amitava Sen, “Automated Classification of Software Issue Reports Using Machine Learning Techniques: An Empirical Study,” Innovations in Systems and Software Engineering, Springer. (Accepted, March 2017) (doi:10.1007/s1133)


BugBang_rel_v4.0.R =>    Main R script for issue classification

datasets/exp1   =>       Datasets for experiment 1

datasets/exp2   =>       Datasets for experiment 2

LogsForJournal.zip =>    Logs containing results reported in the journal paper

TablesAndGraphsForJournal.zip => Tables, graphs and scripts (to plot graphs from the tables) for results reported in the journal



Commandline in linux:

Rscript BugBang_rel_v4.0.R --infile="./datasets/exp1/http_client.csv" --outfile="./out/classification_http_client.out" --max_terms_in_dtm=0.25 --normalize=3 --cv_fold=10

Ensure that the input files and the output directory exist. 

The released script is for experiment 1 (data in folder "exp1"). For experiments in exp2, "CLASSIFIED" in the following lines should be replaced with "TYPE":

training.data.input <- training.data.input [ training.data.input$CLASSIFIED %in% c("BUG", "NUG"), ]

Train.Type     <- as.factor(training.data.input$CLASSIFIED);


