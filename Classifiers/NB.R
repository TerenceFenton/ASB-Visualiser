# Here begins the Naive Bayes (Document-esk) Training
set.seed(83)

# Preprocess so Trans_Type, Payee and Memo are together in one string
trialStr <- c()
wordRemoval <- c("i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", 
                 "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", 
                 "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", 
                 "their", "theirs", "themselves", "what", "which", "who", "whom", "this", 
                 "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", 
                 "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", 
                 "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", 
                 "of", "at", "by", "for", "with", "about", "against", "between", "into", 
                 "through", "during", "before", "after", "above", "below", "to", "from", 
                 "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", 
                 "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", 
                 "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", 
                 "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", 
                 "will", "just", "don", "should", "now", "devonport", "auckland", "-", "takapuna",
                 "uni", "city", "orewa")

for (i in 1:500) {
  row <- dfNoTFRLabelled[i,]
  str1 <- row$Tran_Type
  str2 <- row$Payee
  str3 <- row$Memo
  bigStr <- paste(str1, str2, str3)
  bigStrLower <- tolower(bigStr)
  splitBigStr <- strsplit(bigStrLower, " ", fixed = TRUE)[[1]]
  freshList <- c()
  for (word in splitBigStr) {
    if (!(word %in% wordRemoval)) {
      freshList <- c(freshList, word)
    }
  }
  pastedfreshList <- paste(freshList, collapse = " ")
  trialStr <- c(trialStr, pastedfreshList)
}

dfNoTFRLabelled["trialStr"] <- trialStr

# Sample them down
threeSplit <- sample(c("Train", "Val", "Test"), 500, TRUE, c(0.6, 0.2, 0.2))
trainDF <- dfNoTFRLabelled[threeSplit == "Train",]
valDF <- dfNoTFRLabelled[threeSplit == "Val",]
testDF <- dfNoTFRLabelled[threeSplit == "Test",]

# Determine prob of Category
probC <- c()
for (i in 1:7) {
  probC <- c(probC, sum(trainDF$Classifier == i)/nrow(trainDF))
}

# Collect Counts
# Naive Bayes is essentially one big spreadsheet of probabilities, so:
categories <- c("Word", "Income", "Expenses", "Food", "Wants", "Travel", 
                "Health", "Education", "Investments")
NBCounts <- data.frame("Word"= c("a"),"Income"=c(0), "Expenses"=c(0), "Food"=c(0), 
                      "Wants"=c(0), "Travel"=c(0), "Health"=c(0), "Education"=c(0),
                      "Investments"=c(0))

for (i in 1:nrow(trainDF)) {
  trial <- trainDF[i, "trialStr"]
  classifier <- trainDF[i, "Classifier"] + 1
  splitTrial <- strsplit(trial, " ", fixed = TRUE)[[1]]
  
  for (word in splitTrial) {
    if (!(word %in% NBCounts$Word)) {
      newrow <- data.frame("Word"= c(word),"Income"=c(0), "Expenses"=c(0), "Food"=c(0), 
                 "Wants"=c(0), "Travel"=c(0), "Health"=c(0), "Education"=c(0), "Investments"=c(0))
      NBCounts <- rbind(NBCounts, newrow)
    }
    
    pos <- as.logical(NBCounts["Word"] == word)
    coord <- as.numeric(1:nrow(NBCounts))[pos]
    NBCounts[coord, categories[classifier]] <- NBCounts[coord, categories[classifier]] + 1
  }
}

# Now that we have the counts, it should be fairly straight forward to get every probability sorted.

NBPosteriori <- data.frame("Word"= NBCounts$Word,"Income"=rep(0, nrow(NBCounts)), 
                           "Expenses"=rep(0, nrow(NBCounts)), "Food"=rep(0, nrow(NBCounts)), 
                           "Wants"=rep(0, nrow(NBCounts)), "Travel"=rep(0, nrow(NBCounts)), 
                           "Health"=rep(0, nrow(NBCounts)), "Education"=rep(0, nrow(NBCounts)),
                           "Investments"=rep(0, nrow(NBCounts)))

for (i in 2:dim(NBCounts)[2]) {
  index <- NBCounts[,i] != 0
  xUnique <- nrow(NBCounts)
  numWords <- sum(NBCounts[index, i])
  
  for (j in 1:nrow(NBCounts)) {
    numSpecificWord <- NBCounts[j,i]
    laplace <- (numSpecificWord + 1) / (numWords + xUnique)
    NBPosteriori[j,i] <- log(laplace)
  }
}

# Now that we have a table of log-likelihoods, let's make a function we can call 
# for a speedy classification.

NBClassifier <- function(NBtable, text, probHypo) {
  vmap <- c()
  splitText <- strsplit(text, " ", fixed = TRUE)[[1]]
  
  for (c in 2:dim(NBtable)[2]) {
    logVal <- 0
    for (word in splitText) {
      if (word %in% NBtable$Word) {
        pos <- as.logical(NBtable["Word"] == word)
        coord <- as.numeric(1:nrow(NBtable))[pos]
        logVal <- logVal + NBtable[coord, c]
      }
    }
    probo <- log(probHypo[c-1])
    vmap <- c(vmap, logVal + probo)
  }
  vmap
}

# Lets start collecting preds
trainPreds <- c()
valPreds <- c()
testPreds <- c()

for (text in trainDF$trialStr) {
  vmap <- NBClassifier(NBPosteriori, text, probC)
  pred <- which.max(vmap)
  trainPreds <- c(trainPreds, pred)
}

for (text in valDF$trialStr) {
  vmap <- NBClassifier(NBPosteriori, text, probC)
  pred <- which.max(vmap)
  valPreds <- c(valPreds, pred)
}

for (text in testDF$trialStr) {
  vmap <- NBClassifier(NBPosteriori, text, probC)
  pred <- which.max(vmap)
  testPreds <- c(testPreds, pred)
}

trainAC <- sum(trainDF$Classifier == trainPreds) / nrow(trainDF) # 0.91
valAC <- sum(valDF$Classifier == valPreds) / nrow(valDF) # 0.78
testAC <- sum(testDF$Classifier == testPreds) / nrow(testDF) # 0.69








