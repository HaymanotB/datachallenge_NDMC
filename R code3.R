
# Load necessary packages
library(dplyr)
library(readr)
require(corrplot)
library(graphics)
library(grDevices)
library(caret)
library(randomForest)
library(xgboost)
library(e1071)
library(ggplot2)

cat("then read CHAMPS.CSV data  from ", getwd())
df <- read.csv("CHAMPS.CSV")
dimensions <- dim(df)
num_rows <- dimensions[1]
num_cols <- dimensions[2]
dimensions
rows <- nrow(df)
rows

cols <- ncol(df)
cols
cat("The dataset has", rows, "rows and", cols, "columns.\n")

columns <- colnames(df)
columns
cat("Columns:", columns, "\n")
# To extract only three columns such as, "dp_013", "dp_108", and "dp_118"

df<-df[,c(16,23,125)]
df
columns <- colnames(df)
columns
##### to rename columns
df<-df%>%rename(case_type = dp_013,
                   underlying_cause = dp_108,
                   maternal_condition = dp_118)
#### to rename values

df <- df %>%
  mutate(case_type = recode(case_type,
                            "CH00716" = "Stillbirth",
                            "CH01404" = "Death in the first 24 hours",
                            "CH01405" = "Early Neonate (1 to 6 days)",
                            "CH01406" = "Late Neonate (7 to 27 days)",
                            "CH00718" = "Infant (28 days to less than 12 months)",
                            "CH00719" = "Child (12 months to less than 60 months)"
  ))

### to rename single error value under maternal conditions

df <- df %>%
  mutate(maternal_condition = recode(maternal_condition,
                                     "Abruptio placenta" = "Abruption placenta"))
### to generate Proportion of null values in each column

null_proportions <- colSums(is.na(df)) / nrow(df)
cat("Proportion of null values in each column:\n")

print(null_proportions)

## to calculate magnitude and proportion of each of the infant underlying cause 

underlying_cause_counts <- df %>%
  group_by(underlying_cause) %>%
  summarise(count = n()) %>%
  mutate(proportion = count / sum(count))

# Print the result

print(underlying_cause_counts)

### or

underlying_cause_counts <- df %>%
  filter(!is.na(underlying_cause)) %>%
  group_by(underlying_cause) %>%
  summarize(count = n()) %>%
  mutate(proportion = count / sum(count))

# Print the result

print(underlying_cause_counts)

## to sort by count from largest to smallest

sort_underlying_cause_counts <- head(underlying_cause_counts[order(-underlying_cause_counts$count), ], 97)

## to remove third rows b/c it is not cause

sort_underlying_cause_counts <- sort_underlying_cause_counts[c(1,2,4:97),]

## to calculate magnitude and proportion of each of the maternal factors 

maternal_condition_counts <- df %>%
  group_by(maternal_condition) %>%
  summarise(count = n()) %>%
  mutate(proportion = count / sum(count))

print(maternal_condition_counts)

## or

maternal_condition_counts <- df %>%
  filter(!is.na(maternal_condition_counts)) %>%
  group_by(maternal_condition_counts) %>%
  summarize(count = n()) %>%
  mutate(proportion = count / sum(count))

# Print the result

print(maternal_condition_counts)

## to sort out largest to smallest 

sort_maternal_condition_counts<- head(maternal_condition_counts[order(-maternal_condition_counts$count), ], 97)

## to remove row 1 b/c it is not maternal factor
sort_maternal_condition_counts <- sort_maternal_condition_counts[2:97,]

## to calculate magnitude and proportion of each of the infant case case type 

case_type_counts <- df %>%
  group_by(case_type) %>%
  summarise(count = n()) %>%
  mutate(proportion = count / sum(count))

case_type_counts

## or

case_types <- df %>%
  filter(!is.na(case_type)) %>%
  group_by(case_type) %>%
  summarize(count = n()) %>%
  mutate(proportion = count / sum(count))

# Print the result
print(case_types)

## to sort out

sort_case_types<- head(case_types[order(-case_types$count), ], 6)

### to extreact only top underlying cause and maternal factors

top_causes<-sort_underlying_cause_counts[1:3,1:2]
print(top_causes)

top_factor<-sort_maternal_condition_counts[1:3,1:2]

top_causes_list <- top_causes$underlying_cause

df_top_causes <- df %>%
  filter(underlying_cause %in% top_causes_list)
df_top_causes_numeric <- df_top_causes %>%
  mutate(
    case_type = as.numeric(factor(case_type)),
    underlying_cause = as.numeric(factor(underlying_cause)),
    maternal_condition = as.numeric(factor(maternal_condition)))

### correlation analysis
cor_matrix <- cor(df_top_causes_numeric[, c('case_type', 'underlying_cause', 'maternal_condition')], use = "complete.obs")
cor_matrix

heatmap_plot<- heatmap(cor_matrix, col = cm.colors(256), margins = c(10,10),
                       xlab = NULL, ylab =  NULL,
                       main = "heatmap(<Mtcars data, ..., scale = \"column\")")


### or
corrplot(cor_matrix, method = "number", type = "lower", tl.cex = 0.7, tl.col = "blue",bg ="red" ,diag = TRUE,)
top_infant_causes<- head(underlying_cause_counts[order(-underlying_cause_counts$count),],6)

###############################


df_model <- df_numeric %>%
  select(case_type, underlying_cause, maternal_condition) %>%
  na.omit()
 
# Split data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(df_model$case_type, p = 0.7, list = FALSE)
trainData <- df_model[trainIndex, ]
testData <- df_model[-trainIndex, ]

# Train models using caret
control <- trainControl(method = "cv", number = 10)
models <- list()

# Logistic Regression
models$logistic <- train(case_type ~ ., data = trainData, method = "glm", family = "binomial", trControl = control)

# SVM
models$svm <- train(case_type ~ ., data = trainData, method = "svmRadial", trControl = control)

# Random Forest
models$random_forest <- train(case_type ~ ., data = trainData, method = "rf", trControl = control)

# Gradient Boosting
models$gbm <- train(case_type ~ ., data = trainData, method = "gbm", trControl = control, verbose = FALSE)

# XGBoost
models$xgboost <- train(case_type ~ ., data = trainData, method = "xgbTree", trControl = control)

# Evaluate models on test data
results <- lapply(models, predict, newdata = testData)
accuracy <- sapply(results, function(pred) {
  mean(pred == testData$case_type)
})

print(accuracy)

# Plot AUC and ROC curves
library(pROC)
roc_data <- lapply(results, function(pred) {
  roc(testData$case_type, as.numeric(pred))
})

# Plot ROC curves
plot(roc_data[[1]], col = "red", main = "ROC Curves for Different Models")
for (i in 2:length(roc_data)) {
  plot(roc_data[[i]], add = TRUE, col = i + 1)
}
legend("bottomright", legend = names(models), col = 1:length(models), lty = 1)


# Feature importance visualization for Random Forest
varImpPlot(models$random_forest$finalModel, main = "Feature Importance for Random Forest")

# Plot top five infant underlying causes of child death

top_infant_causes<-top_infant_causes[c(1,2,4,5,6),]

ggplot(top_infant_causes, aes(x = reorder(underlying_cause, -count) , y = count )) +
  geom_bar(stat = "identity", fill = "blue" , colour = "red") +
  coord_flip() + 
  labs(title = "Top 5 Infant Underlying Causes of Child Death",
       x = "Underlying_Cause",
       y = "Count")



# Plot top five maternal factors contributing to child death

top_maternal_factors <- head(maternal_condition_counts[order(-maternal_condition_counts$count), ], 6)
top_maternal_factors<-top_maternal_factors[c(2,4,3,5,6),]

top_maternal_factors<- top_maternal_factors %>%
  mutate(maternal_condition = recode(maternal_condition,
                                     "Fetus and newborn affected by other forms of placental separation and hemorrhage (Abruption placenta)" = 
                                     "(Abruption placenta)"))
top_maternal_factors<- top_maternal_factors %>%
  mutate(maternal_condition = recode(maternal_condition, "Fetus and newborn affected by other forms of placental separation and hemorrhage" = "(F & N affected by PSH)"))

ggplot(top_maternal_factors, aes(x = reorder(maternal_condition, -count), y  
                                 = count)) +
  geom_bar(stat = "identity", fill = "#00abaa" , colour = "blue") +
  coord_flip() + 
  labs(title = "Top 5 Maternal Factors Contributing to Child Death",
       x = "Maternal Condition",
       y = "Count")

# Plot distribution of child deaths based on case types
 
ggplot(case_types, aes(x = reorder(case_type, -count), y = count)) +
  geom_bar(stat = "identity" , fill = "red" , colour = "blue") +
  coord_flip() +
  labs(title = "Distribution of Child Death by Case Type",
       x = "Case Type",
       y = "Count")

# or
ggplot(case_types, aes(x = reorder(case_type, -count), y = count)) +
  geom_bar(stat = "identity" , fill = "#00abcc" , colour = "red") +
  coord_flip() +
  labs(title = "Distribution of Child Death by Case Type",
       x = "Case Type",
       y = "Count")

