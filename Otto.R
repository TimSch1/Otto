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

#plot range/max values
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

#heat map
Col.Scale = colorRampPalette(colors=c('blue','white','red'))(5)
heat.Med = spread(feat.centers[,-3],variable, Median)[,-1]

#with all features
heatmap.2(data.matrix(heat.Med), Rowv=TRUE, Colv=FALSE, scale='column', trace='none', 
          col=Col.Scale, margins = c(3,5), ylab='Class', xlab='Feature Median')

#get only relevant features w/ median > 0
idx = which(colSums(heat.Med)!=0)
heat.Med.small = select(heat.Med, idx)

#condensed heatmap
dev.off()
heatmap.2(data.matrix(heat.Med.small), Rowv=TRUE, Colv=FALSE, scale='column', trace='none', 
          col=Col.Scale, margins = c(3,5), ylab='Class', xlab='Feature Median')

