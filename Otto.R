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

library(GGally)
#ggpairs(subset, columns = 1:4, color='targets', alpha=0.4,
 #       upper = list(continuous='density'),
  #      lower = list(continuous='points'))


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
  return(sum(out))
}

##from robert
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



#generate stratified cross validation folds
cv_fit = function(y, data, folds = 10, ...){
  classes = vector('list', length(unique(y)))
  cvfolds = vector('list', length(unique(y)))
  for(i in 1:length(unique(y))){
  classes[[i]] = which(y == unique(y)[i])
  cvfolds[[i]] = sample(1:folds, size=length(classes[[i]]), replace=T)
  }
  scores = vector('integer')
  for(i in 1:folds){
    
  }
}
idx = which(cvfolds[[1]] != 10)
train = classes[[1]][idx]
test = classes[[1]][-idx]

#parallelize and build model using logloss fxn
library(caret)
library(doParallel)
cl = makeCluster(3)
registerDoParallel(cl)
gbmGrid = expand.grid(interaction.depth=1, 
                      n.trees = 50, 
                      shrinkage=0.1)

fitControl = trainControl(method='cv', number=10, 
                          classProbs=TRUE, summaryFunction = mcLogLoss)





library(doParallel)
cl = makeCluster(3)
registerDoParallel(cl)

trees = c(100,150,200,250,300)

tune = foreach(i = c(50,100), .packages='gbm') %dopar% {
gbmMod = gbm(target~., data= train, distribution='multinomial', n.trees=i, interaction.depth=1, shrinkage = 0.01, n.cores=3)
gbmPred = predict(gbmMod, train, n.trees=10, type='response')
gbmPreddf = as.data.frame(gbmPred[,,1])
gbmPreddf$obs = train$target
LogLoss(gbmPreddf)
}


#this one now
library(caret)
idx = createDataPartition(train$target, p=0.75, list=FALSE)
train2 = train[idx,]
test = train[-idx]




trees = c(100,150,200,250,300)
depth = c(3,5,7)

tune = foreach(i = trees, .packages='gbm') %:% 
  foreach(j = depth, .combine = 'rbind') %dopar% {
  gbmMod = gbm(target~., data= train2, distribution='multinomial', n.trees=i, interaction.depth=j, shrinkage = 0.01, n.cores=3)
  gbmPred = predict(gbmMod, test, n.trees=i, type='response')
  gbmPreddf = as.data.frame(gbmPred[,,1])
  gbmPreddf$obs = factor(test$target)
  LogLoss(gbmPreddf)
}






stopCluster(cl)
