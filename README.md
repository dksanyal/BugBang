# BugBang
This repository contains an R script to classify an issue report from JIRA as referring to a bug or not. It also contains associated datasets.
They are used in the following paper:
Nitish Pandey, Debarshi Kumar Sanyal, Abir Hudait, and Amitava Sen, “Automated Classification of Software Issue Reports Using Machine Learning Techniques: An Empirical Study,” Innovations in Systems and Software Engineering, Springer. (Accepted, March 2017) (doi:10.1007/s1133)

SCRIPTS:
File                   Purpose
----                   -------
BugBang_rel_v4.0.R     Main R script for issue classification
datasets/exp1          Datasets for experiment 1
datasets/exp2          Datasets for experiment 2


HOW TO RUN:
Rscript BugBang.R --infile="./datasets/exp1/http_client.csv" --outfile="./out/classification_http_client.out" --max_terms_in_dtm=0.25 --normalize=3 --cv_fold=10