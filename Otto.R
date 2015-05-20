#For Otto Kaggle Competition

setwd("C:/Users/TimBo/Downloads/R docs and scripts/Otto")
train = read.csv('ottotrain.csv', header=T)
targets = train$target
train = train[,-c(1, 95)]
colnames(train) = seq(1, 93, 1)



#exploring features
library(plyr)
library(dplyr)
library(ggplot2)
library(reshape2)
train$target = NULL
eventrates = ddply(train, .(targets), function(x) colSums(x!=0)/nrow(x))
classSums = rowSums(eventrates[,-1])
eventrates.mat = as.matrix(eventrates[,-1])
normalized = sweep(eventrates.mat, 1, classSums, `/`)
normalized = as.data.frame(cbind(targets = seq(1,9,1), normalized))
eventrate_m = melt(eventrates, id='targets')classSu
normalized_m = melt(normalized, id='targets')

#event rate and normalized event-rate by class
ggplot(eventrate_m, aes(x=variable, y=value, fill=targets, label=variable))+
  ylab('Event-rate')+geom_bar(stat='identity',position='dodge')+xlab('Feature')+
  theme_bw()+theme(axis.text.x=element_text(angle=45, hjust=1))+facet_wrap(~targets)+
  geom_text()

ggplot(normalized_m, aes(x=variable, y=value, fill=factor(targets), label=variable))+
  ylab('Normalized Event-rate')+geom_bar(stat='identity',position='dodge')+xlab('Feature')+
  theme_bw()+theme(axis.text.x=element_text(angle=45, hjust=1))+facet_wrap(~targets)+
  geom_text()

#plot feature ranges by class
library(reshape2)
library(ggplot2)
train$id = seq(1, nrow(train), 1)
train$target = targets
train_m = melt(train, id=c('target','id'), class = variable)
feat.ranges = ddply(train_m, .(target, variable), function(x) range(x$value))

ggplot(feat.ranges, aes(x=variable, y=V2, group=target, fill=target, label=variable))+xlab('')+
  geom_bar(stat='identity')+facet_wrap(~target)+ylab('Feature Range')+theme_bw()+geom_text()+
  theme(axis.text.x=element_blank())

#plot typical values (mean & median)
feat.centers = ddply(train_m, .(target, variable), summarize, Mean = mean(value), Median = median(value))
ggplot(feat.centers, aes(x=variable, y=Mean, group=target, fill=target, label=variable))+xlab('')+
  geom_bar(stat='identity')+facet_wrap(~target)+ylab('Feature Mean')+theme_bw()+geom_text()+
  theme(axis.text.x=element_blank())

ggplot(feat.centers, aes(x=variable, y=Median, group=target, fill=target, label=variable))+xlab('')+
  geom_bar(stat='identity')+facet_wrap(~target)+ylab('Feature Median')+theme_bw()+geom_text()+
  theme(axis.text.x=element_blank())

#heat map with all features
library(tidyr)
library(gplots)
Col.Scale = colorRampPalette(colors=c('blue','white','red'))(5)
heat.Med = spread(feat.centers[,-3],variable, Median)[,-1]
heatmap.2(data.matrix(heat.Med), Rowv=TRUE, Colv=FALSE, scale='column', trace='none', 
          col=Col.Scale, margins = c(3,5), ylab='Class', xlab='Feature Median')

#replot heatmap with only features w/ median > 0
idx = which(colSums(heat.Med)!=0)
heat.Med.small = select(heat.Med, idx)
dev.off()
heatmap.2(data.matrix(heat.Med.small), Rowv=TRUE, Colv=FALSE, scale='column', trace='none', 
          col=Col.Scale, margins = c(3,5), ylab='Class', xlab='Feature Median')


#Build LogLoss evaluation metric used by Kaggle
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

##Unable to use caret for tuning parameter search and model eval due to memory requirements/burden/leakage##
##single gbm object ~1gb, only have 8gb##
##Building parallelized function for model evaluation and tuning parameter selection##

#data for competition
train = read.csv('ottotrain.csv')
test = read.csv('ottotest.csv')
train = train[,-1]
test = test[,-1]


#register parallel backend
library(doParallel)
cl = makeCluster(4)
registerDoParallel(cl)


#no cv using logloss fxn, just holdout test set for model evaluation
library(caret)
idx = createDataPartition(train$target, p=0.85, list=FALSE)
train2 = train[idx,]
test = train[-idx,]

#initial gbm tuning parameters
trees = c(100,150,200,250,300)
depth = c(3,5,7)

#tune over grid of parameters
tune = foreach(i = trees, .packages='gbm') %:% 
  foreach(j = depth, .combine = 'rbind') %dopar% {
  gbmMod = gbm(target~., data= train2[,-1], distribution='multinomial', n.trees=i, 
               interaction.depth=j, shrinkage = 0.01, n.cores=3)
  gbmPred = predict(gbmMod, test[,-1], n.trees=i, type='response')
  gbmPreddf = as.data.frame(gbmPred[,,1])
  gbmPreddf$obs = test$target
  LogLoss(gbmPreddf)
}

#first model submission with best parameters chosen, unsurprisingly the are max trees (300) & depth (7)
finalMod = gbm(target~., data= train[,-1], distribution='multinomial', n.trees=300, interaction.depth=7, shrinkage = 0.01, n.cores=3)
##Results: first submission score 0.70854 - beats uniform probability:2.19, and rf benchmarks:1.50
##best logloss was with highest number of trees and deepest depth, going to continue increasing trees/depth
##features 1,3,6,10,12,13,21,27,28,29,31,37,46,49,51,52,61,63,65,66,73,74,80,81,82,87,89 have zero Var Imp.


#new tuning parameters
trees = c(350,400,450,500,550)
depth = c(9,11)

#second search in parameter space
tune2 = foreach(i = trees, .packages='gbm') %:% 
  foreach(j = depth, .combine = 'rbind') %dopar% {
    gbmMod = gbm(target~., data= train2[,-1], distribution='multinomial', n.trees=i, 
                 interaction.depth=j, shrinkage = 0.01, n.cores=3)
    gbmPred = predict(gbmMod, test[,-1], n.trees=i, type='response')
    gbmPreddf = as.data.frame(gbmPred[,,1])
    gbmPreddf$obs = test$target
    LogLoss(gbmPreddf)
  }

finalMod2 = gbm(target~., data= train[,-1], distribution='multinomial', n.trees=550, 
                interaction.depth=11, shrinkage = 0.01, n.cores=4)

#make submissions, only second output shown
submit2 = predict(finalMod2, test[,-1], n.trees=550, type='response')
submit2 = as.data.frame(submit1[,,1])
write.csv(submit2, 'submit2.csv')
finalMod2.varImp = summary(finalMod2)
write.csv(finalMod2.varImp, 'submit2varImp.csv')
##Results: second submission score 0.59317
##best logloss was with highest number of trees and deepest depth, going to continue increasing trees/depth
##6,12,21,27,28,31,37,52,63,82 zero var imp
##gbm quickly becoming too computationally complex for my machine, switched to xgboost.

stopCluster(cl)

##xgboost - tuning. Tuned depth, then n.trees.

library(xgboost)
library(methods)

train = read.csv('ottotrain.csv')
test = read.csv('ottotest.csv')
train = train[,-1]
test = test[,-1]

train$target = gsub('Class_', '', train$target)
class = as.numeric(train$target) - 1
train$target = NULL

train = as.matrix(train)
train = matrix(as.numeric(train), nrow = nrow(train), ncol=ncol(train))
test = as.matrix(test)
test = matrix(as.numeric(test), nrow=nrow(test))

param <- list('objective' = 'multi:softprob',
              'eval_metric' = 'mlogloss',
              'num_class' = 9,
              'nthread' = 4,
              'max.depth' = 5)

xgbmod.cv = xgb.cv(param=param, data = train, label = class, 
                nfold = 3, nrounds=200)

xgbfinalMod = xgboost(param=param, data = train, label = class, nrounds=53)

# Make prediction
submit4 = predict(xgbfinalMod,test)
submit4 = matrix(submit4,9,length(submit4)/9)
submit4 = t(submit4)
submit4 = as.data.frame(submit4)
submit4 = cbind(id = 1:nrow(submit4), submit4)
names(submit4) = c('id', paste0('Class_',1:9))
write.csv(submit4, file='submit4.csv', quote=FALSE,row.names=FALSE)

##submission result using max.depth = 6, nround = 50 was 0.50763
##submission results using max.depth = 10, nround = 57 was 0.47653

##cv 3-fold tuning
##max.depth = 6 0.524514
##max.depth = 7 gave 0.514282
##max.depth = 8 gave 0.508554
##max.depth = 9 gave 0.504817
##max.depth = 10, nrounds=50 gave 0.503991, nrounds = 57 gave 0.503269
##max.depth = 11 gave 0.510535 <- overfitting

importance_matrix <- xgb.importance(colnames(train), model = xgbfinalMod)
xgb.plot.importance(importance_matrix)

##Feature Engineering

train = read.csv('engineered1train.csv')
class = train$class
train = train[,-c(1,95)]


#counts of all unique numbers per row
nums = unique(as.numeric(as.matrix(train))) #leave nums unchanged from train set

p = matrix(ncol=length(nums))
colnames(p) = nums
p = foreach(i = 1:nrow(train), .combine='rbind') %dopar% {
    sapply(nums, function(x) sum(x == train[i,]))
}


colnames(p) = sapply(nums, function(x) paste0('Num', x))

train = cbind(train, rsum = rowSums(train)) #rowsums
train = cbind(train, p)

write.csv(cbind(train, class), 'engineered1train.csv')

test = read.csv('ottotest.csv')
test = test[,-1]

l = matrix(ncol=length(nums))
colnames(l) = nums
l = foreach(i = 1:nrow(test), .combine='rbind') %dopar% {
  sapply(nums, function(x) sum(x == test[i,]))
}


colnames(l) = sapply(nums, function(x) paste0('Num', x))

test = cbind(test, rsum = rowSums(test)) #rowsums
test = cbind(test, l)

write.csv(test, 'engineered1test.csv')

stopCluster(cl)

##New Model w/ features

class = gsub('Class_','',class)
class = as.numeric(train$target) - 1

train = as.matrix(train)
train = matrix(as.numeric(train), nrow = nrow(train), ncol=ncol(train))
colnames(train) = colnames(test)
test = as.matrix(test)
test = matrix(as.numeric(test), nrow=nrow(test))
colnames(test) = colnames(train)

param <- list('objective' = 'multi:softprob',
              'eval_metric' = 'mlogloss',
              'num_class' = 9,
              'nthread' = 4,
              'max.depth' = 6)

xgbmod.cv = xgb.cv(param=param, data = train, label = class, 
                   nfold = 3, nrounds=130) #nround =56, max.depth=10, gives 0.494899

xgbfinalMod = xgboost(param=param, data = train, label = class, nrounds=104)

submit5 = predict(xgbfinalMod,test)
submit5 = matrix(submit5,9,length(submit5)/9)
submit5 = t(submit5)
submit5 = as.data.frame(submit5)
submit5 = cbind(id = 1:nrow(submit5), submit5)
names(submit5) = c('id', paste0('Class_',1:9))
write.csv(submit5, file='submit5.csv', quote=FALSE,row.names=FALSE)
#scored 0.47704 not better than previous score

#using ONLY most frequent feature counts
train = train[,1:105]
train = train[,-104]

#retune -
#max.depth = 11 gave nround(54) = 0.515250
#max.depth = 10 gave nround(53) = 0.508982
#max.depth = 9 gave nround(65) = 0.504933
#max.depth = 8 gave nround(74) = 0.506664 
#max.depth = 7 gave nround(104) = 0.499573 - submission score = 0.47365. new best (renamed this submit5.)
#max.depth = 6 gave nround(130) = 0.500334


#now tune with ALL new features
#retune - 
#max.depth = 10 gave nround(54) = 0.509774
#max.depth = 9 gave nround(72) = 0.505913
#max.depth = 8 gave nround(85) =  0.503530
#max.depth = 7 gave nround(104) = 0.499491 - submit6 score = 0.47365. exactly the same as submit5.
#max.depth = 6 gave nround(128) = 0.500250 (may benefit from more nrounds)

submit6 = predict(xgbfinalMod,test)
submit6 = matrix(submit6,9,length(submit6)/9)
submit6 = t(submit6)
submit6 = as.data.frame(submit6)
submit6 = cbind(id = 1:nrow(submit6), submit6)
names(submit6) = c('id', paste0('Class_',1:9))
write.csv(submit6, file='submit6.csv', quote=FALSE,row.names=FALSE)


xgb.save(xgbfinalMod, 'xgbfinalmodel')
xgb.load('xgbfinalmodel')

#follow code above to get training predictions, repredicted i know its problematic, classes 2&3 most often misclassified
library(caret)
pred_classes = sapply(1:nrow(trainpreds), function(x) which(trainpreds[x,] == max(trainpreds[x,])))
pred_classes = paste0('Class_',as.character(pred_classes))
confusionMatrix(pred_classes, classes)


#plot overlapping classes 2 & 3 to look for differences
p = cbind(p, classes)
p = as.data.frame(p)
class23 = p[p$classes == 2 | p$classes == 3,]
class23 = class23[,order(colSums(class23), decreasing=T)] #order columns by frequency
class23 = class23[,1:41] #only select features with frequencies >= 10, classes change at id = 16122-16123

library(reshape2)
library(ggplot2)
class23$id = 1:nrow(class23)
class23_m = melt(class23, id = c('id', 'classes'))
ggplot(class23_m, aes(x=variable, y=id, fill=value))+ geom_tile()+geom_hline(aes(yintercept=16122))+
  theme(axis.text.x = element_text(angle = 90, hjust=1))+ylab('Observations - Classes 2 & 3')+
  scale_fill_continuous(high='darkred', low='white', name='Frequency')+xlab('')

class23_m = class23_m[class23_m$variable != 'Num0' & class23_m$variable != 'Num1' & class23_m$variable != 'Num2' & class23_m$variable != 'Num3',]


##svm for classes 2&3 - using all engineered features
library(caret)
train = train[train$class=='Class_2'|train$class=='Class_3',]
idx = createDataPartition(train$class, p=0.40, list=F)
svmTrain = train[idx,]
svmTest = train[-idx,]

svmMod = train(class~., data=svmTrain, method='svmRadial', preProc=c('center','scale'), tuneLength=10, trControl = trainControl(method='repeatedcv', repeats=5))





#visualize in Eigenspace
library(psych)
train = read.csv('engineered1train.csv')
pca.train = scale(train[,-c(1,247)], center=T, scale=T)
pca.train = principal(pca.train, nfactors=10, covar=F)
pca.coords = as.data.frame(pca.train$scores)
pca.coords$class = factor(gsub('Class_','',train$class))

library(rgl)
class234 = pca.coords[pca.coords$class %in% c(2,3,4),]
plot3d(class234[,2], class234[,1], class234[,3], col=class234$class)
legend3d("topright", legend = paste('Class_', c('2', '3', '4')), pch=16, col = unique(class234$class), cex=1, inset=c(0.02))

#see if SVM model separates classes
library(doParallel)
cl = makeCluster(4)
registerDoParallel(cl)

sigmaRangeReduced <- sigest(as.matrix(pca.coords[,-ncol(pca.coords)])) 
svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced[1], .C = 2^(seq(-4, 4)))

ctrl = trainControl(method='repeatedcv', repeats = 5, classProbs=TRUE, summaryFunction=mcLogLoss)

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

svmMod = train(pca.coords[,-ncol(pca.coords)], pca.coords$class,
               method = 'svmRadial', metric='mcLogLoss',
               tuneGrid = svmRGridReduced,
               fit = FALSE, trControl=ctrl, maximize=FALSE)

head(pca.coords)

stopCluster(cl)


library(h2o)
localH2O <- h2o.init(nthread=4, Xmx='8g')

train <- read.csv("ottotrain.csv")

#used for blending only
library(caret)
idx = createDataPartition(train$target, p=0.85, list=FALSE)
train = train[idx,]
test = train[-idx,]

for(i in 2:94){
  train[,i] <- as.numeric(train[,i])
  train[,i] <- sqrt(train[,i]+(3/8))
}

#not used for blending
test <- read.csv("ottotest.csv")

for(i in 2:94){
  test[,i] <- as.numeric(test[,i])
  test[,i] <- sqrt(test[,i]+(3/8))
}



train.hex <- as.h2o(localH2O,train)
test.hex <- as.h2o(localH2O,test[,2:94])

predictors <- 2:(ncol(train.hex)-1)
response <- ncol(train.hex)

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

subSums = rowSums(submission[,-1])
submissionNormed = sweep(as.matrix(submission[,-1]), 1, subSums, `/`)
colnames(submissionNormed) = paste0(rep('Class_',9),1:9)
submissionNormed = as.data.frame(submissionNormed)
submissionNormed$obs = test$target
LogLoss(submissionNormed) #0.88

write.csv(train, 'trainforcv.csv')
write.csv(test, 'testforcv.csv')

pred_classes = apply(submissionNormed[,-ncol(submissionNormed)], 1, function(x) which(x == max(x)))
pred_classes = paste0('Class_',as.character(pred_classes))
confusionMatrix(pred_classes, submissionNormed$obs)

#used trainforcv and testforcv. otherwise used script above

param <- list('objective' = 'multi:softprob',
              'eval_metric' = 'mlogloss',
              'num_class' = 9,
              'nthread' = 4,
              'max.depth' = 7)

xgbfinalMod = xgboost(param=param, data = train, label = class, 
                   nfold = 3, nrounds=385, eta=0.1, colsample.bytree=0.5)

#max.depth = 5, nround = 609, eta=0.1, colsample.bytree=1, test ll= 0.490782
#max.depth = 5, nround = 704, eta=0.1, colsample.bytree=0.5, test ll= 0.485484
#max.depth = 5, nround = 871, eta=0.1, colsample.bytree=0.25, test ll= 0.485374


#max.depth = 6, nround = 444, eta=0.1, colsample.bytree=1, test ll= 0.489010
#max.depth = 6, nround = 523, eta=0.1, colsample.bytree=0.5, test ll= 0.482306
#max.depth = 6, nround = 631, eta=0.1, colsample.bytree=0.25, test ll= 0.483093

 
#max.depth = 7, nround = 341, eta=0.1, colsample.bytree=1, test ll= 0.490034
#max.depth = 7, nround = 385, eta=0.1, colsample.bytree=0.5, test ll= 0.481144** submission8 score = 0.45702 score
#max.depth = 7, nround = 429, eta=0.1, colsample.bytree=0.25, test ll= 0.484539 

xgbfinalMod = xgboost(param=param, data = train, label = class, nrounds=385, eta=0.1, colsample.bytree=0.5)#0.45702 
xgbfinalMod = xgboost(param=param, data = train, label = class, nrounds=463, eta=0.085, colsample.bytree=0.45)#0.45671
xgbfinalMod = xgboost(param=param, data = train, label = class, nrounds=494, eta=0.08, colsample.bytree=0.45)# 0.45637


submit10 = predict(xgbfinalMod,test)
submit10 = matrix(submit10,9,length(submit10)/9)
submit10 = t(submit10)
submit10 = as.data.frame(submit10)
submit10 = cbind(id = 1:nrow(submit10), submit10)
names(submit10) = c('id', paste0('Class_',1:9))
write.csv(submit10, file='submit10.csv', quote=FALSE,row.names=FALSE)

xgbmod.cv = xgb.cv(param=param, data = train, label = class, 
                   nfold = 3, nrounds=1200, eta=0.075, colsample.bytree=0.45)

#max.depth = 7, nround = 379, eta=0.11, colsample.bytree=0.35, test ll= 0.482658
#max.depth = 7, nround = 519, eta=0.11, colsample.bytree=0.15, test ll= 0.489333
#max.depth = 7, nround = 461, eta=0.09, colsample.bytree=0.35, test ll= 0.480999
#max.depth = 7, nround = 611, eta=0.09, colsample.bytree=0.15, test ll= 0.486262
#max.depth = 7, nround = 492, eta=0.085, colsample.bytree=0.35, test ll= 0.480797
#max.depth = 7, nround = 463, eta=0.085, colsample.bytree=0.45, test ll= 0.480152**submission9 - scored 0.45671
#max.depth = 7, nround = 469, eta=0.085, colsample.bytree=0.40, test ll= 0.481215
#max.depth = 7, nround = 544, eta=0.075, colsample.bytree=0.35, test ll= 0.480764
#max.depth = 7, nround = 543, eta=0.08, colsample.bytree=0.35, test ll= 0.482171
#max.depth = 7, nround = 494, eta=0.08, colsample.bytree=0.45, test ll= 0.479735** new best cv score
#max.depth = 7, nround = 533, eta=0.075, colsample.bytree=0.45, test ll= 0.480853



submitblendtune = predict(xgbfinalblendMod,test)
submitblendtune = matrix(submitblendtune,9,length(submitblendtune)/9)
submitblendtune = t(submitblendtune)
submitblendtune = as.data.frame(submitblendtune)
names(submitblendtune) = paste0('Class_',1:9)
write.csv(submitblendtune, file='xgbblendmod.csv', quote=FALSE,row.names=FALSE)

xgbpred_classes = apply(submitblendtune, 1, function(x) which(x == max(x)))
xgbpred_classes = paste0('Class_',as.character(xgbpred_classes))

confusionMatrix(xgbpred_classes, labels)
