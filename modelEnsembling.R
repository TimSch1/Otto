data = read.csv('engineered1train.csv')
class = data$class

xgbtrain = data[,2:105]
xgbtrain = xgbtrain[,-104]


class = gsub('Class_','',class)
class = as.numeric(class) - 1

xgbtrain = as.matrix(xgbtrain)
xgbtrain = matrix(as.numeric(xgbtrain), nrow = nrow(xgbtrain), ncol=ncol(xgbtrain))

library(caret)
set.seed(400)
idx = createDataPartition(class, p=0.85, list=FALSE)
xgbtest = xgbtrain[-idx,]
xgbtrain = xgbtrain[idx,]
testlabels = class[-idx]
trainlabels = class[idx]

param <- list('objective' = 'multi:softprob',
              'eval_metric' = 'mlogloss',
              'num_class' = 9,
              'nthread' = 4,
              'max.depth' = 7)

library(methods)
library(xgboost)

xgbfinalblendMod = xgboost(param=param, 
                      data = xgbtrain, 
                      label = class, 
                      nrounds=494, 
                      eta=0.08, 
                      colsample.bytree=0.45)


xgb.save(xgbfinalblendMod, 'xgbfinalblendmodel')

datatest = read.csv('engineered1test.csv')
xgbtest = datatest[,2:105]
xgbtest = xgbtest[,-104]

xgbtest = as.matrix(xgbtest)
xgbtest = matrix(as.numeric(xgbtest), nrow = nrow(xgbtest), ncol=ncol(xgbtest))



xgblendfinal_pred = predict(xgbfinalblendMod, xgbtest)
xgblendfinal_pred = matrix(xgblendfinal_pred,9,length(xgblendfinal_pred)/9)
xgblendfinal_pred = t(xgblendfinal_pred)
xgblendfinal_pred = as.data.frame(xgblendfinal_pred)
names(xgblendfinal_pred) = paste0('Class_',1:9)
write.csv(xgblendfinal_pred, file='xgblendfinal_pred.csv', quote=FALSE,row.names=FALSE)


xgb.load('xgblendfinalmodel')

################################################H2O####################################################
library(h2o)
localH2O <- h2o.init(nthread=4, Xmx='8g')

class = data$class
train = data[,2:105]
train = train[,-104]
train$class = class

test = datatest[,2:105]
test = test[,-104]

for(i in 1:(ncol(train)-1)){
  train[,i] <- as.numeric(train[,i])
  train[,i] <- sqrt(train[,i]+(3/8))
}

for(i in 1:(ncol(test))){
  test[,i] <- as.numeric(test[,i])
  test[,i] <- sqrt(test[,i]+(3/8))
}


train.hex <- as.h2o(localH2O,train)
test.hex <- as.h2o(localH2O,test)

predictors <- 1:(ncol(train.hex)-1)
response = ncol(train.hex)


submission <- read.csv("sampleSubmission.csv")
submission[,2:10] <- 0

for(i in 1:20){
  print(i)
  model <- h2o.deeplearning(x=predictors,
                            y=response,
                            data=train.hex,
                            classification=T,
                            activation="RectifierWithDropout",
                            hidden=c(1024,512,256),
                            hidden_dropout_ratio=c(0.5,0.5,0.5),
                            input_dropout_ratio=0.05,
                            epochs=100,
                            l1=1e-5,
                            l2=1e-5,
                            rho=0.99,
                            epsilon=1e-8,
                            train_samples_per_iteration=2000,
                            max_w2=10,
                            seed=1)
  
  submission[,2:10] <- submission[,2:10] + as.data.frame(h2o.predict(model,test.hex))[,2:10]
  print(i)
  write.csv(submission,file="submission.csv",row.names=FALSE) 
}   
submission = read.csv('submission.csv')


submission1 = submission1[,-1]
subSums = rowSums(submission1)
submissionNormed1 = sweep(as.matrix(submission1), 1, subSums, `/`)
colnames(submissionNormed) = paste0(rep('Class_',9),1:9)
submissionNormed1 = as.data.frame(submissionNormed1)
submissionNormed$obs = test$target
LogLoss(submissionNormed)




LogLoss = function(data, lev=NULL, model=NULL){
  pred = data[,-which(colnames(data) == 'obs')]
  eps = 1e-15
  predsnormed = do.call(cbind, lapply(pred, function(x) sapply(x, function(y) max(min(y, 1-eps), eps))))
  logProbs = log(as.matrix(predsnormed))  
  log1minus = log(1-as.matrix(predsnormed))  
  out = rep(NA, nrow(data))
  for(i in 1:length(data$obs)){
    colidx = which(data$obs[i] == colnames(logProbs))
    out[i] = sum(logProbs[i,colidx], log1minus[i,-colidx])
  }
  return(-sum(out)/length(out))
}

##Find overall model weights##

#first i is 1:10 0.7866895

#48% h2o / 52% xgboost 0.7866227
for(i in 40:60){
  weighted_pred = (i*submissionNormed + (100-i)*xgblend_pred)/100
  weighted_pred$obs = paste0('Class_', as.character((testlabels+1)))
  ll = LogLoss(weighted_pred)
  print(i)
  print(ll)
}

weighted_pred = (48*submissionNormed + 52*xgblend_pred)/100
weighted_pred$obs = paste0('Class_', as.character((testlabels+1)))

##Dive deeper beyond model averages

for(j in 1:9){
  for(i in 1:10){
    weighted_pred1 = weighted_pred
    print(paste0('h2o weight = ', i))
    print(paste0('column',j))
    weighted_pred1[,j] = (i*submissionNormed[,j]+(10-i)*xgblend_pred[,j])/10
    ll = LogLoss(weighted_pred1)
    print(ll)
  }
}
  

#class 1; h20 = 8, xgboost = 2, 0.786466
#class 2; h20 = 5, xgboost = 5, 0.7865928
#class 3; h2o = 4, xgboost = 6, 0.7864068
#class 4; h20 = 3, xgboost = 7, 0.7862879
#class 5; h20 = 4, xgboost = 6, 0.786622
#class 6; h20 = 5, xgboost = 5, 0.7866182
#class 7; h20 = 5, xgboost = 5, 0.7866383
#class 8; h20 = 5, xgboost = 5, 0.7866125
#class 9; h20 = 5, xgboost = 5, 0.7866207

weighted_pred2 = weighted_pred1
weighted_preds3[,1] = (8*subNorm[,1] + 2*xgblendfinal_pred[,1])/10
weighted_preds3[,3] = (4*subNorm[,3] + 6*xgblendfinal_pred[,3])/10
weighted_preds3[,4] = (3*subNorm[,4] + 7*xgblendfinal_pred[,4])/10
weighted_preds3[,5] = (4*subNorm[,5] + 6*xgblendfinal_pred[,5])/10

LogLoss(weighted_pred2) #0.7859146

weighted_pred2 = as.matrix(weighted_pred2)
weighted_pred2 = matrix(as.numeric(weighted_pred2), nrow = nrow(weighted_pred2), ncol=ncol(weighted_pred2))
weighted_pred2 = weighted_pred2[,-10]

param <- list('objective' = 'multi:softprob',
              'eval_metric' = 'mlogloss',
              'num_class' = 9,
              'nthread' = 4,
              'max.depth' = 2)

xgbmod.cv = xgb.cv(param=param, data = weighted_pred2, label = testlabels, 
                   nfold = 3, nrounds=1200, eta=0.1)

#max.depth = 1, nround = 52, eta=0.3, colsample.bytree=1, test ll= 0.455617
#max.depth = 1, nround = 147, eta=0.1, colsample.bytree=1, test ll= 0.454446
#max.depth = 1, nround = 236, eta=0.1, colsample.bytree=0.5, test ll= 0.453400
#max.depth = 1, nround = 357, eta=0.1, colsample.bytree=0.25, test ll= 0.458217

#max.depth = 1, nround = 47, eta=0.08, colsample.bytree=0.45, test ll= 0.469667

#max.depth = 2, nround = 29, eta=0.3, colsample.bytree=1, test ll= 0.455414
#max.depth = 2, nround = 87, eta=0.1, colsample.bytree=1, test ll= 0.453182**
#max.depth = 2, nround = 155, eta=0.1, colsample.bytree=0.5, test ll= 0.459024
#max.depth = 2, nround = 207, eta=0.1, colsample.bytree=0.25, test ll= 0.465074

#max.depth = 3, nround = 25, eta=0.3, colsample.bytree=1, test ll= 0.465275
#max.depth = 3, nround = 75, eta=0.1, colsample.bytree=1, test ll= 0.459041
#max.depth = 3, nround = 114, eta=0.1, colsample.bytree=0.5, test ll= 0.464345
#max.depth = 3, nround = 188, eta=0.1, colsample.bytree=0.25, test ll= 0.474762

#max.depth = 4, nround = 21, eta=0.3, colsample.bytree=1, test ll= 0.468915
#max.depth = 4, nround = 69, eta=0.1, colsample.bytree=1, test ll= 0.472718
#max.depth = 4, nround = 100, eta=0.1, colsample.bytree=0.5, test ll= 0.468746
#max.depth = 4, nround = 171, eta=0.1, colsample.bytree=0.25, test ll= 0.484421

#max.depth = 5, nround = 21, eta=0.3, colsample.bytree=1, test ll= 0.478406
#max.depth = 5, nround = 67, eta=0.1, colsample.bytree=1, test ll= 0.478422
#max.depth = 5, nround = 89, eta=0.1, colsample.bytree=0.5, test ll= 0.477553
#max.depth = 5, nround = 155, eta=0.1, colsample.bytree=0.25, test ll= 0.487217

#max.depth = 6, nround = 21, eta=0.3, colsample.bytree=1, test ll= 0.490719
#max.depth = 6, nround = 68, eta=0.1, colsample.bytree=1, test ll= 0.485659
#max.depth = 6, nround = 82, eta=0.1, colsample.bytree=0.5, test ll= 0.481521
#max.depth = 6, nround = 137, eta=0.1, colsample.bytree=0.25, test ll= 0.498963

#max.depth = 7 not competitive

xgbfinalblendMod = xgboost(param=param, data = weighted_pred2, label = class, nrounds=87, eta=0.1)

#######################testing ensembles####################################
test = read.csv('engineered1test.csv')
xgbtest = test[,2:105]
xgbtest = xgbtest[,-104]
xgbtest = as.matrix(xgbtest)
xgbtest = matrix(as.numeric(xgbtest), nrow = nrow(xgbtest), ncol=ncol(xgbtest))

xgbpred_sub = predict(xgblendMod, xgbtest)
