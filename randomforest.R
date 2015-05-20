data = read.csv('engineered1train.csv')
labels = data$class
data = data[,2:105]
data = data[,-104]


library(caret)

library(doParallel)
cl = makeCluster(4)
registerDoParallel(cl)


mcLogLoss <- function (data,
                       lev = NULL,
                       model = NULL) {
  
  if (!all(levels(data[, "pred"]) == levels(data[, "obs"])))
    stop("levels of observed and predicted data do not match")
  
  LogLoss <- function(actual, pred, err=1e-15) {
    pred[pred < err] <- err
    pred[pred > 1 - err] <- 1 - err
    -1/nrow(actual)*(sum(actual*log(pred)))
  }
  
  dtest <- dummyVars(~obs, data=data, levelsOnly=TRUE)
  actualClasses <- predict(dtest, data[,-1])
  
  out <- LogLoss(actualClasses, data[,-c(1:2)])  
  names(out) <- "mcLogLoss"
  out
}

mtryGrid = expand.grid(mtry = c(3,5,7,9))

fitControl <- trainControl(method = "cv",
                           number = 5,
                           classProbs = TRUE,
                           summaryFunction = mcLogLoss)

rfTune<- train(x = data,
               y = labels,
               method = "rf",
               trControl = fitControl,
               metric = "mcLogLoss",
               ntree = 1000,
               tuneGrid = mtryGrid, 
               maximize = FALSE,
               importance = TRUE)

Random Forest 

61878 samples
103 predictor
9 classes: 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9' 

No pre-processing
Resampling: Cross-Validated (5 fold) 

Summary of sample sizes: 49503, 49501, 49504, 49502, 49502 

Resampling results across tuning parameters:
  
  mtry  mcLogLoss  mcLogLoss SD
3     0.6832063  0.001371156 
5     0.6280732  0.001462273 
7     0.6019547  0.002529347 
9     0.5862516  0.002805598 

mcLogLoss was used to select the optimal model using  the smallest value.
The final value used for the model was mtry = 9. 