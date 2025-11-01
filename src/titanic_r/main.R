# Load data
df <- read.csv("data/train.csv", na.strings = c("", "NA"))

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
df$Embarked[df$Embarked == "C"] <- 0
df$Embarked[df$Embarked == "Q"] <- 1
df$Embarked[df$Embarked == "S"] <- 2
df$Embarked <- as.numeric(df$Embarked)


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
test_df <- read.csv("data/test.csv", na.strings = c("", "NA"))

# Apply same preprocessing
test_df$Age[is.na(test_df$Age)] <- mean(test_df$Age, na.rm = TRUE)
test_df <- subset(test_df, select = -Cabin)

# Embarked: fill with mode, same as training
mode_emb <- names(sort(table(test_df$Embarked), decreasing = TRUE))[1]
test_df$Embarked[is.na(test_df$Embarked)] <- mode_emb


test_df$Sex <- ifelse(test_df$Sex == "male", 0, 1)

test_df$Embarked[test_df$Embarked == "C"] <- 0
test_df$Embarked[test_df$Embarked == "Q"] <- 1
test_df$Embarked[test_df$Embarked == "S"] <- 2
test_df$Embarked <- as.numeric(test_df$Embarked)

test_df <- test_df[, c("PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked")]


# Predict survivability
test_pred_probs <- predict(model, newdata = test_df, type = "response")
test_pred <- ifelse(test_pred_probs > 0.5, 1, 0)


print("First 10 Predictions on test.csv:")
print(head(test_pred, 10))

output <- data.frame(PassengerId = test_df$PassengerId, Survived = test_pred)
# Save predictions to CSV

write.csv(output, "data/predictions_r.csv", row.names = FALSE)
cat("Predictions saved to data/predictions_r.csv\n")
quit(save = "no")

