##################################################################################################
                    DATACO GLOBAL SUPPLYCHAIN------CAPstone project
#################################################################################################

library(mice)
library(ggplot2)
library(gbm)
library(Ckmeans.1d.dp)
library(xgboost)
library(dmm)
library(xgboost)
library(reshape2)
library(DataExplorer)
library(corrplot)
library(ipred)
library(rpart)
library(DMwR)
library(gridExtra)
library(e1071)
library(GGally)
library(mice)
library(ROCR)
library(ineq)
library(plyr)
library(car)
library(lmtest)
library(pan)
library(corrplot)
library(ggplot2)
library(DataExplorer)
library(reshape)
library(RColorBrewer)
library(class)
library(caTools)
library(caret)
library(psych)
library(car)
library(foreign)
library(MASS)
library(lattice)
library(randomForest)
library(rpart.plot)
##install.packages("nortest")
library(nortest)
library(GPArotation)
library(dplyr)

#################SETTING DIRECTORY AND IMPORTING DATASET#############
setwd("C:/Users/senthilkumar/OneDrive/Documents/senthil")
dataco=read.csv("DataCoSupplyChainDataset.csv")
View(dataco)
#### CHECKING DIMENSIUONALITY################
dim(dataco)
#####CHECKING COLUMN NAMES############
names(dataco)
####CHECKING THE STRUCTURE OF DATA######
str(dataco)
###### CHECKING FOR NAS########################
sum(is.na(dataco))
summary(dataco)
dataco<-as.data.frame(dataco)
for (i in 1:length(dataco)) { print(paste(colnames(dataco[i]),class(dataco[,i])))}
library(DataExplorer)
library(ggplot2)
plot_intro(dataco)
na_counttrain<-sapply(dataco, function(y)sum(length(which(is.na(y)))))
na_counttrain<-data.frame(na_counttrain)
na_counttrain
######## REMOVING CLOMMNS WITH FULL OF NAS AND redundant column#####
Dataconew<-dataco[,-c(20,44,47)]

names(Dataconew)
plot_intro(Dataconew)

head(Dataconew)
str(Dataconew)

##########################Converting categorical variable binary numerical variable for easier analytic process#####
Dataconew$Type=revalue(Dataconew$Type,c("DEBIT"="0", "TRANSFER"="1","CASH"="2","PAYMENT"="3"))

Dataconew$Delivery.Status=revalue(Dataconew$Delivery.Status,c("Advance shipping"="0","Late delivery"="1","Shipping on time"="2","Shipping canceled"="3"))


Dataconew$Market=revalue(Dataconew$Market,c("Pacific Asia"="0", "USCA"="1","Africa"="2","Europe"="3","LATAM"="4"))

Dataconew$Customer.Segment=revalue(Dataconew$Customer.Segment,c("Consumer"="0", "Corporate"="1","Home Office"="2"))

Dataconew$Order.Status=revalue(Dataconew$Order.Status,c("CANCELED"="0", "CLOSED"="1","COMPLETE"="2","ON_HOLD"="3","PAYMENT_REVIEW"="4","PENDING"="5","PENDING_PAYMENT"="6","PROCESSING"="7","SUSPECTED_FRAUD"="8"))

Dataconew$Shipping.Mode=revalue(Dataconew$Shipping.Mode,c("First Class"="0", "Same Day"="1","Second Class"="2","Standard Class"="3"))



View(Dataconew)


############REMOVING REDUNDANT AND DUPLICATE COLUMNS###############################
Dataconewset<-Dataconew[,-c(9,10,12,13,15,16,18,19,21,22,23,25,45,48,11,26,28,40,41,46,49,34,27,44,30)]


View(Dataconewset)
dim(Dataconewset)
str(Dataconewset)

#######################Coverting non numerical vaiable to numerical variable#####################
for (i in 1:length(Dataconewset[1,])){
  if(length(as.numeric(Dataconewset[,i][!is.na(Dataconewset[,i])])[!is.na(as.numeric(Dataconewset[,i][!is.na(Dataconewset[,i])]))])==0){}
  else {
    Dataconewset[,i]<-as.numeric(Dataconewset[,i])
  }
}

###################################CORRELATION#############################################################

cor(Dataconewset)
table(Dataconewset$Late_delivery_risk)


library(DataExplorer)
library(gridExtra)
library(ggplot2)
library(corrplot)

######CHECKING FOR MULTICOLINEARITY##########################################################################
corrplot(cor(Dataconewset))


############OUTLIERS IDENTIFICATION AND TREATMENT############

plot_histogram(Dataconewset,ggtheme = theme_light())

boxplot(dataco$Days.for.shipping..real,col = "orange")
boxplot(dataco$Sales.per.customer,col = "orange")
boxplot(dataco$Benefit.per.order,col = "orange")

boxplot(dataco$Order.Profit.Per.Order,col = "orange")


boxplot(dataco$Order.Item.Product.Price,col = "orange")
boxplot(dataco$Sales,col = "orange")

#########################TREATING OUTLIERS############################################
quantile(Dataconewset$Sales.per.customer,c(0.95))
Dataconewset$Sales.per.customer[which(Dataconewset$Sales.per.customer>569.905)]<-383.98
quantile(Dataconewset$Benefit.per.order,c(0.95))
Dataconewset$Benefit.per.order[which(Dataconewset$Benefit.per.order>132.29)]<-132.29
quantile(Dataconewset$Product.Price,c(0.95))
Dataconewset$Product.Price[which(Dataconewset$Product.Price>399.98)]<-399.98
quantile(Dataconewset$Sales,c(0.95))
Dataconewset$Sales[which(Dataconewset$Sales>399.98)]<-399.98
 View(Dataconewset)

######################################Checking outliers again########################
 boxplot(Dataconewset$Benefit.per.order,col = "orange")
 boxplot(Dataconewset$Product.Price,col = "orange")
 
 #################################### PCA TO HANDLE MULTICOLINEARITY################################
 # As the dataset contain huge variable which is corerelated with one another pca is considered to be the best method to treat collinearity###
 
 
 Dataconewsetcorr = cor(Dataconewset)
 
 corrdf = data.frame(Dataconewsetcorr)
 write.csv(corrdf, "corrmat.csv")
 
 print(cortest.bartlett(Dataconewsetcorr,nrow(dataco)))
 
 #####EIGEN VALUES AND EIGEN VECTORS######
 
 A = eigen(Dataconewsetcorr)
 eigenvalues = A$values
 eigenvectors = A$vectors
 eigenvalues
 eigenvectors
 
 ###PCA#########
 part.pca = eigenvalues/sum(eigenvalues)*100
 part.pca
 plot(eigenvalues,type="lines", xlab="Principal Components",ylab="Eigen Values")
 pc=principal(Dataconewset,nfactors=length(Dataconewset),rotate="none")
 pc
 
 pc2=principal(Dataconewset,nfactors=6,rotate="none")
 pc2
 
 
 print(pc)
 summary(pc)
 attributes(pc)
 pc$communality
 pc$values
 pc$loadings
 ######ROTATION 2######
 pc2
 pc2$communality
 plot(pc2)
 plot(pc2, row.names(pc2$loadings))
 pairs.panels(pc2$loadings)
 pc2$scores
 print(pc2)
 
 
 ######## PC ROTATION 3#########
 
 pc3=principal(Dataconewset,nfactors=6,rotate="varimax")
 pc3
 
 pc3
 pc3$communality
 plot(pc3)
 plot(pc3, row.names(pc3$loadings))
 pairs.panels(pc3$loadings)
 pc3$scores
 print(pc3)
 
 #########six new components are created based the pc3############################
 
 MYdatacoDf=cbind(Dataconewset$Late_delivery_risk,pc3$scores)
 MYdatacoDf=as.data.frame(MYdatacoDf)
 names(MYdatacoDf)=c("Late_delivery_risk","SalesVolume","Productinfo","ROI","shipmentinfo","Customerinfo","Discountinfo")
 names(MYdatacoDf)
 
 
 View(MYdatacoDf)

 ####################Splitting datset in to train and test########################
 
 
 index=createDataPartition(y=MYdatacoDf$Late_delivery_risk,p=0.7,list = FALSE)
 traindata=MYdatacoDf[index,]
 table(traindata$Late_delivery_risk)
 
 testdata=MYdatacoDf[-index,]
 table(testdata$Late_delivery_risk)
 
 View(testdata)
 
 ###########################################################################
 MODEL BULDING
 Logistic regression
 
#################################################################################
 
 lgmodel<-glm(formula = Late_delivery_risk~.,traindata,family = binomial)
 lgmodel
 summary(lgmodel)
##########################################################################
 ###LESS SIGNIFICANT VARIABLES ARE REMOVED#########
 traindatnew<-traindata[,-c(2,3,6)]
 testdatanew<-testdata[,-c(2,3,6)]
 #########################################################################
 
 lgmodel1<-glm(formula = Late_delivery_risk~.,testdatanew,family = binomial)
 lgmodel1
 summary(lgmodel1)
 confint(lgmodel1)
 
 lg_predictions<-predict(lgmodel1,testdatanew,type="response")
 lg_predictions
 
 cm = table(testdatanew$Late_delivery_risk, lg_predictions>0.5)
 cm
 
 TP = cm[2,2]
 FN = cm[2,1]
 FP = cm[1,2]
 TN = cm[1,1]
 
 Accuracy = (TP+TN)/nrow(testdatanew)
 Accuracy
 sensitivity = TP/(TP+FN)
 sensitivity
 Specificity = TN/(TN+FP)
 Specificity 
 Precision = TP/(TP+FP)
 Precision
 F1 = 2*(Precision*sensitivity)/(Precision + sensitivity) 
 F1
 
 #########################################################################
 
 NAIVE BAYES
 #######################
 
 library(e1071)
 testdatanew$Late_delivery_risk=as.factor(testdatanew$Late_delivery_risk)
 traindatnew$Late_delivery_risk=as.factor(traindatnew$Late_delivery_risk)
 
 NBmodel<-naiveBayes(Late_delivery_risk~.,data = traindatnew)

 NBmodel
 
 NB_predictions<-predict(NBmodel,testdatanew)
 NB_predictions
 table(NB_predictions,testdatanew$Late_delivery_risk)
 
 confusionMatrix(NB_predictions,testdatanew$Late_delivery_risk)
 
 
 #########################################################################
 KNN
############################################################################# 
set.seed(1134)
classifier_knn<-knn(train = traindatnew,test = testdatanew,cl=traindatnew$Late_delivery_risk,k=1)
classifier_knn
cm<-table(testdatanew$Late_delivery_risk,classifier_knn)
cm 


######Chooosing K########################
misclasserror<-mean(classifier_knn!=testdatanew$Late_delivery_risk)
print(paste('Accuracy=',1-misclasserror)) 
####K=5
classifier_knn5<-knn(train = traindatnew,test = testdatanew,cl=traindatnew$Late_delivery_risk,k=5)
classifier_knn5


misclasserror5<-mean(classifier_knn5!=testdatanew$Late_delivery_risk)
print(paste('Accuracy=',1-misclasserror5))

#K=7
classifier_knn7<-knn(train = traindatnew,test = testdatanew,cl=traindatnew$Late_delivery_risk,k=7)
classifier_knn7


misclasserror7<-mean(classifier_knn7!=testdatanew$Late_delivery_risk)
print(paste('Accuracy=',1-misclasserror7)) 

confusionMatrix(classifier_knn,testdatanew$Late_delivery_risk)

####CART#####CART MODEL
set.seed(233)
library(rpart.plot)
#defining the seperate variable for training and testing which has splitted above

cart.train=traindatnew
cart.test=testdatanew

#setting the control parameter inputs for Rpart

tree=rpart(formula = Late_delivery_risk~.,data = cart.train,method="class",control=rpart.control(minsplit = 50,minbucket = 16,cp=0.0001))
printcp(tree)
plotcp(tree)
rpart.plot(tree,cex = 0.6)


tree$cptable
print(tree$cptable)
plot(tree$cptable)
tree$cptable[,"xerror"]
min(tree$cptable[,"xerror"])
bestcp=tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"]

#######PRUNNNING TRE#####################
ptree=prune(tree,cp=0.0048643248)
print(ptree)    
rpart.plot(ptree,cex = 0.6)   

##### VARIABLE IMPORTANCE#######
library(caret)
summary(ptree)
ptree$variable.importance
barplot(ptree$variable.importance)


predict.class=predict(ptree,cart.train,type = "class")
predict.class


##########Confusion matrix#############################

tabdev=with(cart.train,table(Late_delivery_risk,predict.class))
cart.train$predict.class=predict(ptree,cart.train,type = "class")
cart.train$predict.score=predict(ptree,cart.train,type = "prob")
cart.train$predict.score
print(tabdev)
confusionMatrix(tabdev)
############PREdiction on test dats##########################



cart.test$predict.class=predict(ptree,cart.test, type="class")
cart.test$predict.score=predict(ptree,cart.test, type="prob")[,"1"]
View(testdata)
tabdev1=with(cart.test,table(Late_delivery_risk,predict.class))
cart.test$predict.score

print(tabdev1)
confusionMatrix(tabdev1)

######################################################################
            RANDOM forest
#####################################################################
library(devtools)
library(randomForest)

x = sqrt(ncol(traindatnew))

x
mtry1 = floor(sqrt(ncol(traindatnew)))
mtry1

names(traindatnew)
x=traindatnew[, c(2:4)]
y=traindatnew$Late_delivery_risk


set.seed(123)


bestmtry = tuneRF(x, y, stepFactor = 1.5, improve = 1e-5, ntree=500)
print(bestmtry)


TRF = tuneRF(x, y,
             mtryStart = 2,
             ntreeTry = 500,
             stepFactor = 1.5,
             improve = 0.0001,
             trace=TRUE,
             plot=TRUE,
             doBest= TRUE,
             nodesize=100,
             importance=TRUE)
print(TRF)
traindatnew$Late_delivery_risk=as.factor(traindatnew$Late_delivery_risk)
testdatanew$Late_delivery_risk=as.factor(testdatanew$Late_delivery_risk)

str(traindatnew)

Dataco_RF = randomForest(Late_delivery_risk~.,data =traindatnew, ntree=200,
                         mtry=2, nodesize=100,
                         importance=TRUE)
print(Dataco_RF)


plot(Dataco_RF, main= "")
title(main="Error rates Random Forest")

######VARIABLE IMPORTANCE############
varImpPlot(Dataco_RF)

library(caret)





traindatnew$predict.classrforest=predict(Dataco_RF, traindatnew, type="class")
traindatnew$prob1=predict(Dataco_RF, traindatnew,type="prob")[, "1"]
testdatanew$predict.classrforest=predict(Dataco_RF, testdatanew, type="class")
testdatanew$prob1=predict(Dataco_RF, testdata,type="prob")[, "1"]
View(traindatnew)


tabdevL=table(traindatnew$Late_delivery_risk, traindatnew$predict.classrforest)
confusionMatrix(tabdevL)

tabdevL2=table(testdatanew$Late_delivery_risk, testdatanew$predict.classrforest)
confusionMatrix(tabdevL2)


##################################################################
set.seed(3311)
BAGGGING

library(gbm)
library(xgboost)
library(caret)
library(ipred)
library(rpart)
mod.bagging<-bagging(Late_delivery_risk~.,data = traindatnew,control=rpart.control(maxdepth = 5,minsplit = 4))
bag.pred<-predict(mod.bagging,testdatanew)
confusionMatrix(bag.pred,testdatanew$Late_delivery_risk)


######################################################
boosting##
set.seed(4431)
str(MYdatacoDf)
MYdatacoDf$Late_delivery_risk=as.factor(MYdatacoDf$Late_delivery_risk)

library(dmm)
index2=createDataPartition(y=MYdatacoDf$Late_delivery_risk,p=0.7,list = FALSE)
Bosstingtraindata=MYdatacoDf[index2,]
table(Bosstingtraindata$Late_delivery_risk)

boostingtestdata=MYdatacoDf[-index2,]
table(boostingtestdata$Late_delivery_risk)
boostingtrainfactor=Bosstingtraindata
boostingtrainfactor$Late_delivery_risk=unfactor(boostingtrainfactor$Late_delivery_risk)
boostingtestfactor=boostingtestdata
boostingtestfactor$Late_delivery_risk=unfactor(boostingtestfactor$Late_delivery_risk)


boost.model<-gbm(Late_delivery_risk~.,data = boostingtrainfactor,distribution = "bernoulli",n.trees = 500,interaction.depth = 4,shrinkage = 0.01)
summary(boost.model)
boost.pred<-predict(boost.model,boostingtestdata,n.trees=500,type="response")
y_pred_num<-ifelse(boost.pred>0.5,1,0)
y_pred<-factor(y_pred_num,levels=c(0,1))
table(y_pred,boostingtestdata$Late_delivery_risk)               
confusionMatrix(y_pred,boostingtestdata$Late_delivery_risk)

###################################################################
            MODEL performance METRIC

######################################################################
LOGISTIC REGRESSION

library(ROCR)
pred.lg<-prediction(lg_predictions,testdata$Late_delivery_risk)
perf.lg<-performance(pred.lg,"tpr","fpr")
plot(perf.lg)


KS<-max((attr(perf.lg,'y.values')[[1]]-attr(perf.lg,'x.values')[[1]]))
KS 
AREA UNDER curve
########################

auc<-performance(pred.lg,"auc");
auc<-as.numeric(auc@y.values)
auc

GINI Coefficients
###################
library(ineq)
gini=ineq(lg_predictions,type = "Gini")  
gini


##########random forest###


predobjtrain = prediction(traindatnew$prob1, traindatnew$Late_delivery_risk)
preftrain = performance(predobjtrain, "tpr", "fpr")
plot(preftrain)

predobjtest = prediction(testdatanew$prob1, testdatanew$Late_delivery_risk)
preftest = performance(predobjtest, "tpr", "fpr")
plot(preftest)


#AUC
auctrain = performance(predobjtrain, "auc")
auctrain= as.numeric(auctrain@y.values)
auctrain

auctest = performance(predobjtest, "auc")
auctest= as.numeric(auctest@y.values)
auctest

#Gini Coefficient train
Ginitrain= (2*auctrain) - 1
Ginitrain

Ginitrainnew = ineq(traindatnew$prob1, "gini")
Ginitrainnew

Ginitest= (2*auctest) - 1
Ginitest

Ginitestnew = ineq(testdatanew$prob1, "gini")
Ginitestnew

###KS-randomforest
KStrain=max(preftrain@y.values[[1]]- preftrain@x.values[[1]])
KStrain

KStest=max(preftest@y.values[[1]]- preftest@x.values[[1]])
KStest











