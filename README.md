# ASB-Visualiser
Welcome to my ASB-Visualiser Project! 

**This project is split up into two parts:**
1. The visualisation and compiling of the dataframes
2. Trial of classification models on dataframes

This code takes in a folder of CSV files where each CSV represents a single account in your ASB banking app. To visualise your spendings and setup the Dataframes relevant for classification, please run the _visualization.Rmd_ file. This File is made completely in base R code so no packages are necessary for this part.

Should you wish to see how well your transactions can be classified, direct your attention to the folder called Classifiers. Inside you will find example HTML reports of my own attempts at classifying my example transactions. The classifiers available are:
1. Naive Bayes Multinomial Classifier
