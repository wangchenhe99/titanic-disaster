# Load libraries
library(tidyverse)
library(dplyr)

# Load data
df <- read.csv("src/data/train.csv", na.strings = c("", "NA"))

print("Initial data summary:")
print(head(df))
print(summary(df))
print(sapply(df, function(x) sum(is.na(x))))


# Handle missing values
df$Age[is.na(df$Age)] <- mean(df$Age, na.rm = TRUE)

# Cabin: too many missing, drop the column
df <- subset(df, select = -Cabin)

# Embarked: fill with most common value (mode)
mode_emb <- names(sort(table(df$Embarked), decreasing = TRUE))[1]  
df$Embarked[is.na(df$Embarked)] <- mode_emb

print("Missing values after cleaning:")
print(sapply(df, function(x) sum(is.na(x))))

# Encode categorical features
df$Sex <- ifelse(df$Sex == "male", 0, 1)
df$Embarked <- recode(df$Embarked, "C" = 0, "Q" = 1, "S" = 2)

# Build logistic regression model
model <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
             data = df, family = binomial)

print("Model summary:")
print(summary(model))

# Measure accuracy on training set
pred_probs <- predict(model, type = "response")
pred <- ifelse(pred_probs > 0.5, 1, 0)
train_acc <- mean(pred == df$Survived)
print(paste("Training accuracy:", round(train_acc, 4)))

# Predict on test.csv
test_df <- read.csv("src/data/test.csv", na.strings = c("", "NA"))

# Apply same preprocessing
test_df$Age[is.na(test_df$Age)] <- mean(test_df$Age, na.rm = TRUE)
df <- subset(test_df, select = -Cabin)
# Embarked: fill with mode, same as training
mode_emb <- names(sort(table(test_df$Embarked), decreasing = TRUE))[1]
test_df$Embarked[is.na(test_df$Embarked)] <- mode_emb


test_df$Sex <- ifelse(test_df$Sex == "male", 0, 1)
test_df$Embarked <- recode(test_df$Embarked, "C" = 0, "Q" = 1, "S" = 2)
test_df <- test_df %>% select(PassengerId, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)

# Predict survivability
test_pred_probs <- predict(model, newdata = test_df, type = "response")
test_pred <- ifelse(test_pred_probs > 0.5, 1, 0)


print("First 10 Predictions on test.csv:")
print(head(test_pred, 10))

output <- data.frame(PassengerId = test_df$PassengerId, Survived = test_pred)
# Save predictions to CSV
write.csv(output, "src/data/predictions_r.csv", row.names = FALSE)
print("Predictions saved to src/data/predictions_r.csv")
