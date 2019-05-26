#####################  ERS_Baskaran.R        ##################
#Code for the project "Emotion Recognition System" for the Data Science: Capstone course of the HarvardX Professional Certificate Program in Data Science
# Code written by: T Baskaran
#Ensure that the downloaded file is: "ANAD_Normalized.csv"
#This is the data file which this project uses for analysis
#The total time taken may be about 900 seconds
#R version used: 3.5.3 (2019-03-11)
#I experienced compatibility problems of R version 3.6.0 with Rmd for generating report. Hence used R 3.5.3 version.
#My Github repository: https://github.com/tbaskaran/edX-Capstone-CYO-Project/upload

########################  Initial Settings   ################
# List of packages required for this analysis

pkg <- c("tidyverse", "knitr", "tictoc", "caret", "caTools", "kernlab", "e1071", "randomForest", "nnet")

# Check if packages are not installed and assign the names of the packages not installed to the variable new.pkg

new.pkg <- pkg[!(pkg %in% installed.packages())]

# If there are any packages in the list that aren't installed, install them

if (length(new.pkg)) {
  install.packages(new.pkg, repos = "http://cran.rstudio.com")
}

#Load the libraries
library(tictoc)
library(caret)
library(tidyverse)
library(caTools)
library(kernlab)
library(e1071)
library(randomForest)
library(nnet)
library(knitr)

#############      Data Preparation         #####################
tic()
#Download the data
if (!file.exists("ANAD_Normalized.csv")) {download.file("https://data.mendeley.com/datasets/xm232yxf7t/1/files/e535362e-ce98-4729-8c7a-f32fec55cc30/ANAD_Normalized.csv?dl=1", "ANAD_Normalized.csv")}

#Read the data set
anad <- read.csv("ANAD_Normalized.csv")
dim(anad)

#Check for the near zero variance variables
nzv <- nearZeroVar(anad)
dim(anad)
length(nzv)

#Remove the nzv columns
data <- anad[,-nzv]
dim(data)

#Remove the "name" and "Emotion" variables as they are not going to be used in the analysis 
data <- data[,-c(1,2)]

# Make the integer variable "Type" as factor
data$Type <- as.factor(data$Type) 
str(data)
toc()

#################             EDA              ###################

# Look at the frequency distribution of the Type of emotion
class(data$Type)
table((data$Type))

# Visual display of the frequency distribution of the "Emotion"
data %>% ggplot(aes(x=Type, fill=Type))+
  geom_bar()+
  scale_fill_discrete(name = "Emotion", labels = c("Angry", "Surprised", "Happy"))+
  ggtitle("Distribution of Emotions")+
  scale_x_discrete(name = "Type")+
  theme_bw()

#Check for the missing values. There is no missing observation in the entire dataset
sum(is.na(anad))


################ Model Development ########################

tic()
#Partition the data set "data" in to training (80%) and test (20%) data sets
set.seed(1)
test_index <- createDataPartition(data$Type, times = 1, p = 0.2, list = FALSE)
train_set <- data[-test_index,]
test_set <-  data[test_index,]
dim(train_set)
dim(test_set)
time_part <- toc()
toc()

###############  k Nearest Neighbours     ###############

#Model 1: kNN model after cross validation

tic()
set.seed(2)
control <- trainControl(method = "cv", 
                        number = 5, p = .9)

train_knn_cv <- train(Type ~ .,
                      method = "knn",
                      tuneGrid = data.frame(k = c(3,5,7)),
                      trControl = control,
                      data = train_set)

pred_knn_cv <- predict(train_knn_cv, test_set, type = "raw")

acc_knn_cv <- confusionMatrix(pred_knn_cv, 
                              test_set$Type)$overall[["Accuracy"]]
acc_knn_cv

time_knn_cv <- toc()

#####################   Logistic Regression    ###################
#Model 2: Boosted Logistic Regression with caret::train using the method "LogitBoost" after Cross Validation

tic()
set.seed(2)
control <- trainControl(method = "cv", 
                        number = 5, p = .9)
train_logit_cv <- train(Type ~ ., 
                        method = "LogitBoost", 
                        tuneGrid = data.frame(nIter=5:15),
                        trControl = control,
                        data = train_set)

plot(x=train_logit_cv$results$nIter,
     y=train_logit_cv$results$Accuracy, 
     type='l')

fit_logit_cv <- lm(train_logit_cv$results$Accuracy ~ train_logit_cv$results$nIter)
abline(fit_logit_cv)

pred_logit_cv <- predict(train_logit_cv, test_set, type = "raw")

acc_logit_cv <- confusionMatrix(pred_logit_cv, 
                                test_set$Type)$overall[["Accuracy"]]

acc_logit_cv

time_logit_cv <- toc()

#Model 3: Logistic Regression with PCA

tic()
fit_logit_pca <- train(Type ~ ., 
                       data = train_set, 
                       method = 'LogitBoost', 
                       tuneGrid = data.frame(nIter=50),
                       preProcess = "pca")

pred_logit_pca <- predict(fit_logit_pca,
                          test_set,
                          type = "raw")

acc_logit_pca<- confusionMatrix(pred_logit_pca,
                                test_set$Type)$overall[["Accuracy"]]

acc_logit_pca

time_logit_pca<- toc()

#####################   Support Vector Machines    ###################

#Model 4: SVM model using train function of the caret package 

tic()
train_svm_caret <- train(Type ~ ., 
                         method = "svmLinear", 
                         data = train_set)

pred_svm_caret <- predict(train_svm_caret, 
                          test_set, 
                          type = "raw")

acc_svm_caret <- confusionMatrix(pred_svm_caret, 
                                 test_set$Type)$overall["Accuracy"]
acc_svm_caret

time_svm_caret <- toc()

#Model 5: SVM model using the svm function of the e1071 package

tic()
colname <- names(train_set)
strpred <- paste(colname[!colname %in% "Type"], collapse = " + ")
formula_svm_e1071 <- as.formula(paste("Type ~ ", strpred))

fit_svm_e1071 <- svm(formula_svm_e1071, 
                     data = train_set)

pred_svm_e1071 <-  predict(fit_svm_e1071, 
                           newdata = test_set)

test_set$Type <- factor(test_set$Type)

acc_svm_e1071 <- confusionMatrix(pred_svm_e1071,
                                 test_set$Type)$overall["Accuracy"]
acc_svm_e1071

time_svm_e1071 <- toc()

#####################   Random Forests    ###################

#Model 6: Random Forest model with randomForest package

tic()
fit_rf <- randomForest(Type~., 
                       data = train_set)

plot(fit_rf)

pred_rf <-  predict(fit_rf, 
                    newdata = test_set)

acc_rf <- confusionMatrix(pred_rf,
                          test_set$Type)$overall["Accuracy"]

acc_rf

time_rf <- toc()

#Model 7: Random Forest after Cross Validation

tic()
set.seed(2)
control <- trainControl(method="cv", number = 5)
grid <- data.frame(mtry = c(1, 5, 10, 25, 50, 100))
train_rf_caret_cv <- train(Type ~ ., 
                           method = "rf", 
                           ntree = 150,
                           trControl = control, 
                           tuneGrid = grid,
                           data = train_set)

ggplot(train_rf_caret_cv)

train_rf_caret_cv$bestTune

pred_rf_caret_cv <- predict(train_rf_caret_cv, 
                            test_set, 
                            type = "raw")

acc_rf_caret_cv <- confusionMatrix(pred_rf_caret_cv, 
                                   test_set$Type)$overall["Accuracy"]

acc_rf_caret_cv

time_rf_caret_cv <- toc()

###########################   Neural Network    ###############

#Model 8: Neural Network Model using caret::train with "nnet" method

tic()
set.seed(2)
library(nnet)

train_nn <- train(Type ~ ., 
                  method = "nnet", 
                  data = train_set)

pred_nn <- predict(train_nn, 
                   test_set, 
                   type = "raw")

acc_nn <- confusionMatrix(pred_nn, 
                          test_set$Type)$overall[["Accuracy"]]

acc_nn

time_nn <- toc()

#################            Results     ##########################
#Printing the results
library(knitr)
df <- data.frame(Model= c("kNN Using CARET after CV",
                          "Logistic Regression Using CARET after CV",
                          "Logistic Regression after PCA",
                          "Support Vector Machines with CARET",
                          "Support Vector Machines with e1071",
                          "Random Forests Using randomForest",
                          "Random Forests Using CARET after CV",
                          "Neural Network Using CARET"),
                 Accuracy=c(acc_knn_cv,
                            acc_logit_cv,
                            acc_logit_pca,
                            acc_svm_caret,
                            acc_svm_e1071,
                            acc_rf,
                            acc_rf_caret_cv,
                            acc_nn))                    

kable(df, caption = "Results of Analysis")
###################     Conclusion            ############

paste0("Best Model: ", df[df$Accuracy==max(df$Accuracy),1])

paste0("Accuracy of the best model: " ,round(max(df$Accuracy)*100,2), "%")
#####################      End    ####################################