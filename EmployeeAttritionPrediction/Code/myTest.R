
# libraries ---------------------------------------------------------------


library(data.table)
library(ggplot2)
library(scales)
library(caret)
library(DMwR)
library(caretEnsemble)
library(pROC)

# data wrangling

library(dplyr)
library(magrittr)
# library(stringr)
# library(stringi)
library(readr)

# machine learning and advanced analytics

# library(DMwR)
# library(caret)
# library(caretEnsemble)
# library(pROC)

# natural language processing

# library(msLanguageR)
library(tm)
# library(jiebaR)

# tools

# library(httr)
# library(XML)
# library(jsonlite)

# data visualization

# library(ggplot2)
# library(scales)
# library(wordcloud)


# init ----------------

DATA1 <- "../Data/DataSet1.csv"
DATA2 <- "../Data/DataSet2.csv"

# read data ------------------

df <- read_csv(DATA1)
head(df)
dim(df)
names(df)
str(df)

dt <- fread(DATA1, stringsAsFactors = TRUE)
head(dt)
dim(dt)
names(dt)
str(dt)

# visualization 1 -----
ggplot(dt, aes(JobRole, fill=Attrition)) +
  geom_bar(aes(y=(..count..)/sum(..count..)), position="dodge") +
  scale_y_continuous(labels=percent) +
  xlab("Job Role") +
  ylab("Percentage")


# corrplot 1 ------
# dev.new()
dt2 <- lapply(dt, as.numeric)
dt2 <- as.data.table(dt2)
M <- cor(dt2)
corrplot(M, tl.cex = 0.75, type="lower")

# visualization 2 --------
# ggplot(filter(df, (YearsAtCompany >= 2) & (YearsAtCompany <= 5) & (JobLevel < 3)),
#        aes(x=factor(JobRole), y=MonthlyIncome, color=factor(Attrition))) +
#   geom_boxplot() +
#   xlab("Department") +
#   ylab("Monthly income") +
#   scale_fill_discrete(guide=guide_legend(title="Attrition")) +
#   theme_bw() +
#   theme(text=element_text(size=13), legend.position="top")


ggplot(dt[(YearsAtCompany >= 2) & (YearsAtCompany <= 5) & (JobLevel < 3), ],
       aes(x=JobRole, y=MonthlyIncome, color=Attrition)) +
  geom_boxplot() +
  xlab("Department") +
  ylab("Monthly income") +
  scale_fill_discrete(guide=guide_legend(title="Attrition")) +
  theme_bw() +
  theme(text=element_text(size=13), legend.position="top")

# vis 3 -------------
# ggplot(filter(df, as.character(Attrition) == "Yes"), aes(x=YearsSinceLastPromotion)) +
#   geom_histogram(binwidth=0.5) +
#   aes(y=..density..) +
#   xlab("Years since last promotion.") +
#   ylab("Density") +
#   # scale_fill_discrete(guide=guide_legend(title="Attrition")) +
#   facet_grid(Department ~ JobLevel)

ggplot(dt[Attrition == "Yes", ], aes(x=YearsSinceLastPromotion)) +
  geom_histogram(binwidth=0.5) +
  aes(y=..density..) +
  xlab("Years since last promotion.") +
  ylab("Density") +
  # scale_fill_discrete(guide=guide_legend(title="Attrition")) +
  facet_grid(Department ~ JobLevel)

# preprocess ------

# get predictors that has no variation.
pred_no_var <- names(df[, nearZeroVar(df)]) %T>% print()
# remove the zero variation predictor columns.
df %<>% select(-one_of(pred_no_var))

pred_no_var <- names(dt[, nearZeroVar(dt), with = FALSE])
print(pred_no_var)
dt[, c(pred_no_var) := NULL]

# convert certain integer variable to factor variable.
int_2_ftr_vars <- c("Education", "EnvironmentSatisfaction", "JobInvolvement", "JobLevel", "JobSatisfaction", "NumCompaniesWorked", "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel")

df[, int_2_ftr_vars] <- lapply((df[, int_2_ftr_vars]), as.factor)
dt[, (int_2_ftr_vars) := lapply(.SD, as.factor), .SDcols = int_2_ftr_vars]

# The variables of character type are converted to categorical type. 
df %<>% mutate_if(is.character, as.factor)

is.factor(df$Attrition)
is.factor(dt$Attrition)

# feature selection --------

# set up the training control.
control <- trainControl(method="repeatedcv", number=3, repeats=1)

# train the model
names(getModelInfo())

model <- train(dplyr::select(df, -Attrition), 
               df$Attrition,
               data=df, 
               method="rf", 
               preProcess="scale", 
               trControl=control)


modeldt <- train(dt[, -c("Attrition")], 
               dt$Attrition,
               data=dt, 
               method="rf", 
               preProcess="scale", 
               trControl=control)

m2 <- train(dt[, -c("Attrition")], 
               dt$Attrition,
               data=dt, 
               method="gbm", 
               preProcess="scale", 
               trControl=control)

# var importance -----------

# estimate variable importance
imp <- varImp(model, scale=FALSE)

# plot
plot(imp)

# estimate variable importance
impdt <- varImp(modeldt, scale=FALSE)

# plot
plot(impdt)


# remove vars ----------

# select the top-ranking variables.
imp_list <- rownames(imp$importance)[order(imp$importance$Overall, decreasing=TRUE)]

# drop the low ranking variables. Here the last 3 variables are dropped. 
top_var <- 
  imp_list[1:(ncol(df) - 3)] %>%
  as.character() 
top_var

# select the top-ranking variables.
imp_list_dt <- rownames(impdt$importance)[order(impdt$importance$Overall, decreasing=TRUE)]

# drop the low ranking variables. Here the last 3 variables are dropped. 
top_var_dt <- imp_list[1:(ncol(dt) - 3)]
top_var_dt

# select the top ranking variables 
df %<>% select(., one_of(c(top_var, "Attrition")))

dt[, (setdiff(names(dt), c(top_var_dt, "Attrition"))) := NULL]
setcolorder(dt, c(top_var_dt, "Attrition"))


# resampling --------

train_index <- createDataPartition(df$Attrition, times=1, p=.7, list = FALSE)
df_train <- df[train_index, ]
df_test <- df[-train_index, ]
table(df_train$Attrition)


train_index_dt <- createDataPartition(dt$Attrition, times=1, p=.7, list = FALSE)
dt_train <- dt[train_index_dt, ]
dt_test <- dt[-train_index_dt, ]
table(dt_train$Attrition)


# note DMwR::SMOTE does not handle well with tbl_df. Need to convert to data frame.

df_train %<>% as.data.frame()

df_train <- SMOTE(Attrition ~ .,
                  df_train,
                  perc.over=300,
                  perc.under=150)
table(df_train$Attrition)

dt_train <- SMOTE(Attrition ~ .,
                  dt_train,
                  perc.over=300,
                  perc.under=150)
table(dt_train$Attrition)


# model building ----------

# initialize training control. 
tc <- trainControl(method="boot", 
                   number=3, 
                   repeats=3, 
                   search="grid",
                   classProbs=TRUE,
                   savePredictions="final",
                   summaryFunction=twoClassSummary)

# SVM model.
time_svm <- system.time(
  model_svm <- train(Attrition ~ .,
                     df_train,
                     method="svmRadial",
                     trainControl=tc)
)

# random forest model

time_rf <- system.time(
  model_rf <- train(Attrition ~ .,
                    df_train,
                    method="rf",
                    trainControl=tc)
)

# xgboost model.

time_xgb <- system.time(
  model_xgb <- train(Attrition ~ .,
                     df_train,
                     method="xgbLinear",
                     trainControl=tc)
)


# ensemble of models ------------

# ensemble of the three models.
time_ensemble <- system.time(
  model_list <- caretList(Attrition ~ ., 
                          data = df_train,
                          trControl = tc,
                          methodList = c("svmRadial", "rf", "xgbLinear"))
)

# stack of models. Use glm for meta model.
model_stack <- caretStack(
  model_list,
  metric = "ROC",
  method = "glm",
  trControl=trainControl(
    method = "boot",
    number = 10,
    savePredictions = "final",
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  )
)


# model validation ----------

models <- list(model_svm, model_rf, model_xgb, model_stack)
predictions <- lapply(models, 
                      predict, 
                      newdata = select(df_test, -Attrition))

# confusion matrix evaluation results.
cm_metrics <- lapply(predictions,
                     confusionMatrix, 
                     reference=df_test$Attrition, 
                     positive="Yes")

# accuracy
acc_metrics <- 
  lapply(cm_metrics, `[[`, "overall") %>%
  lapply(`[`, 1) %>%
  unlist()

# recall
rec_metrics <- 
  lapply(cm_metrics, `[[`, "byClass") %>%
  lapply(`[`, 1) %>%
  unlist()

# precision
pre_metrics <- 
  lapply(cm_metrics, `[[`, "byClass") %>%
  lapply(`[`, 3) %>%
  unlist()

algo_list <- c("SVM RBF", "Random Forest", "Xgboost", "Stacking")
time_consumption <- c(time_svm[3], time_rf[3], time_xgb[3], time_ensemble[3])

df_comp <- 
  data.frame(Models=algo_list, 
             Accuracy=acc_metrics, 
             Recall=rec_metrics, 
             Precision=pre_metrics,
             Time=time_consumption) %T>%
             {head(.) %>% print()}


# inizio sentiment analysis -----------
ggplot(df, aes(JobSatisfaction, fill=Attrition)) +
  geom_bar(aes(y=(..count..)/sum(..count..)), position="dodge") +
  scale_y_continuous(labels=percent) +
  xlab("Job Satisfaction") +
  ylab("Percentage")

ggplot(df, aes(RelationshipSatisfaction, fill=Attrition)) +
  geom_bar(aes(y=(..count..)/sum(..count..)), position="dodge") +
  scale_y_continuous(labels=percent) +
  xlab("Relationship Satisfaction") +
  ylab("Percentage")

ggplot(df, aes(EnvironmentSatisfaction, fill=Attrition)) +
  geom_bar(aes(y=(..count..)/sum(..count..)), position="dodge") +
  scale_y_continuous(labels=percent) +
  xlab("Environment Satisfaction") +
  ylab("Percentage")



ggplot(df, aes(x=factor(Attrition), fill=factor(JobSatisfaction))) +
  geom_bar(width=0.5, position="fill") +
  coord_flip() +
  xlab("Attrition") +
  ylab("Proportion") +
  scale_fill_discrete(guide=guide_legend(title="Score of\n job satisfaction")) 

ggplot(df, aes(x=factor(Attrition), fill=factor(RelationshipSatisfaction))) +
  geom_bar(width=0.5, position="fill") +
  coord_flip() +
  xlab("Attrition") +
  ylab("Proportion") +
  scale_fill_discrete(guide=guide_legend(title="Score of\n relationship satisfaction")) 

ggplot(df, aes(x=factor(Attrition), fill=factor(EnvironmentSatisfaction))) +
  geom_bar(width=0.5, position="fill") +
  coord_flip() +
  xlab("Attrition") +
  ylab("Proportion") +
  scale_fill_discrete(guide=guide_legend(title="Score of\n environment satisfaction")) 


# review comments --------------
df <-
  read_csv(DATA2) %>%
  mutate(Feedback=as.character(Feedback))

head(df$Feedback, 10)

# create a corpus based upon the text data.

corp_text <- Corpus(VectorSource(df$Feedback))
corp_text

# the transformation functions can be checked with 
getTransformations()

# transformation on the corpus.

corp_text %<>%
  tm_map(removeNumbers) %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removeWords, stopwords("english")) %>%
  tm_map(removePunctuation) %>%
  tm_map(stripWhitespace) 

corp_text


# transform corpus to document term frequency.

dtm_txt_tf <- 
  DocumentTermMatrix(corp_text, control=list(wordLengths=c(1, Inf), weighting=weightTf)) 

inspect(dtm_txt_tf[1:10, 1:10])

dtm_txt <-
  removeSparseTerms(dtm_txt_tf, 0.99) %>%
  print()

df_txt <- 
  inspect(dtm_txt) %>%
  as.data.frame()

head(df_txt, 20)



# Sentiment analysis on review comments -------

# form the data set
df_txt %<>% cbind(Attrition=df$Attrition)

# split data set into training and testing set.
train_index <- 
  createDataPartition(df_txt$Attrition,
                      times=1,
                      p=.7) %>%
  unlist()

df_txt_train <- df_txt[train_index, ]
df_txt_test <- df_txt[-train_index, ]

# model building

model_svm <- train(Attrition ~ .,
                   df_txt_train,
                   method="svmRadial",
                   trainControl=tc)

# model evaluation

prediction <- predict(model_svm, newdata=select(df_txt_test, -Attrition))

confusionMatrix(prediction,
                reference=df_txt_test$Attrition,
                positive="Yes")


# msLanguageR ------------
# Install devtools
if(!require("devtools")) install.packages("devtools")
devtools::install_github("yueguoguo/Azure-R-Interface/utils/msLanguageR")
library(msLanguageR)

senti_score <- cognitiveSentiAnalysis(text=df[-train_index, ]$Feedback, apiKey="your_api_key")

df_senti <- mutate(senti_score$documents, Attrition=ifelse(score < 0.5, "Yes", "No"))

confusionMatrix(df_senti$Attrition,
                reference=df[-train_index, ]$Attrition,
                positive="Yes")
