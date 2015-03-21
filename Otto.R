setwd("C:/Users/TimBo/Downloads/R docs and scripts/Otto")
train = read.csv('ottotrain.csv', header=T)
targets = sapply(train$target, function(x) strsplit(as.character(x), '_'))
targets = do.call(rbind, lapply(targets, function(x) x[2]))
train = train[,-c(1, 95)]
colnames(train) = seq(1, 93, 1)
train$target = as.factor(targets)
train$id = seq(1, nrow(train), 1)


#exploring responses
library(plyr)
library(dplyr)
classFreq = ddply(train, .(target), function(x) count(x))
classFreq = mutate(classFreq, Freq = n/sum(n))
qplot(target, n, data=classFreq, stat='identity', geom='bar')

#exploring features
library(reshape2)
library(ggplot2)
train_m = melt(train, id=c('target','id'), class = variable)
feat.ranges = ddply(train_m, .(target, variable), function(x) range(x$value))
ggplot(feat.ranges, aes(x=variable, y=V2, group=target, fill=target, label=variable))+xlab('')+
  geom_bar(stat='identity')+facet_wrap(~target)+ylab('Feature Range')+theme_bw()+geom_text()+
  theme(axis.text.x=element_blank())

feat.centers = ddply(train_m, .(target, variable), summarize, Mean = mean(value), Median = median(value))
ggplot(feat.centers, aes(x=variable, y=Mean, group=target, fill=target, label=variable))+xlab('')+
  geom_bar(stat='identity')+facet_wrap(~target)+ylab('Feature Mean')+theme_bw()+geom_text()+
  theme(axis.text.x=element_blank())

ggplot(feat.centers, aes(x=variable, y=Median, group=target, fill=target, label=variable))+xlab('')+
  geom_bar(stat='identity')+facet_wrap(~target)+ylab('Feature Median')+theme_bw()+geom_text()+
  theme(axis.text.x=element_blank())

library(gplots)
train2 = scale(as.matrix(train[,-c(94,95)]), center=T, scale=T)
rownames(train2) = seq(1,nrow(train2),1)
palette = colorRampPalette(c('lightyellow','darkred'), space='rgb')(100)
heatmap.2(train2, Rowv=NA, Colv=NA, col=palette)
