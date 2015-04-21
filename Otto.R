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
eventrate_m = melt(eventrates, id='targets')
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

#heatmap by class - doesn't work yet
splitdf = split(train[,-94], train$target)
freqbyclass = lapply(splitdf, function(x) lapply(x, table))
freqbyclass = do.call(rbind, freqbyclass)
lapply(freqbyclass, function(x) merge_recurse)


for(j in 1:93){
    freqbyclass = cbind(freqbyclass[[j]], feat = j)
    }

# Not useful, no additional class separation
#library(psych)
#pca.train = scale(train[,-c(94:95)])
#pca.train = principal(pca.train, nfactors=10, covar=F)
#pca.coords = as.data.frame(pca.train$scores)
#subset = pca.coords[idx, ]
#subset$targets = targets[idx]



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
              'max.depth' = 10)

xgbmod.cv = xgb.cv(param=param, data = train, label = class, 
                nfold = 3, nrounds=150)

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

train = read.csv('ottotrain.csv')
class = train$target
train = train[,-c(1,95)]

#rowsums
train = cbind(train, rsum = rowSums(train))

#counts of all unique numbers per row
nums = unique(as.numeric(train)) #leave nums unchanged from train set

p = matrix(ncol=length(nums))
colnames(p) = nums
for(i in 1:nrow(train)){
j = sapply(nums, function(x) sum(x == train[i,]))
p = rbind(p, j)
}
p = p[-1,]
colnames(p) = sapply(colnames(p), function(x) paste0('Num', x))

test = cbind(test, rsum = rowSums(test))
test = cbind(test, p)

write.csv(cbind(train, class), 'engineered1train.csv')

#new model with engineered features made with nrounds = 53, max.depth = 10


