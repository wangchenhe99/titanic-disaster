
import pandas as pd

# Load dataset from local data folder
df = pd.read_csv("src/data/train.csv")
print(print(df.head()))

# Data Exploration and cleaning
print("\nChecking for missing values and data info.")
print(df.info())
print(df.describe())
print(df.isnull().sum())



df['Age'] = df['Age'].fillna(df['Age'].mean())
df = df.drop(columns=['Cabin'])
df = df.dropna(subset=["Embarked"])
print("\nAfter cleaning:")
print(df.isnull().sum())

# 15. Build a logistic regression model to predict survivability on the training set using any features that you see fit.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Sex"]
df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})
df["Embarked"]


features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

X = df[features]
print("\nFeatures selected:")
print(X)

y = df["Survived"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression(max_iter=1000)
print(model.fit(X_train, y_train))


y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# 16. Measure the accuracy of your model on the training set.
print()
print(f"Training model accuracy: {acc:.4f}")

# 17. Load `test.csv` and predict your model on the test set.
# Load test data
test_df = pd.read_csv("src/data/test.csv")
test_df.head()
print("\nMissing values in test.csv:")
print(test_df.isnull().sum())
print(test_df.info())

# Apply same preprocessing
test_df["Sex"] = test_df["Sex"].map({"male": 0, "female": 1})
test_df["Embarked"] = test_df["Embarked"].map({"C": 0, "Q": 1, "S": 2})
test_df["Age"] = test_df["Age"].fillna(df["Age"].mean())
test_df["Fare"] = test_df["Fare"].fillna(df["Fare"].mean())
test_df = test_df.dropna(subset=["Embarked"])
test_df = test_df.drop(columns=['Cabin'])

print("Missing values after cleaning test.csv:")
print(test_df.isnull().sum())

# Select same features
X_test_final = test_df[features]
y_test_pred = model.predict(X_test_final)


# 18. Measure the accuracy of your model on the test set.
print("\n18. first 10 Predictions on test.csv:")
print(y_test_pred[:10]) 