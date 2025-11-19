import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

#load dataset
file = "creditcard.csv"
df = pd.read_csv(file)

# display datatset info
print("Dataset Info:\n")
print(df.info())
print("\n Class Distribution:\n")
print(df['Class'].value_counts())

#define features and target variable
X = df.drop(columns=['Class'],)
y = df['Class']

#split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#initialize k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42) # 5-fold cross-validation | shuffle the data before splitting | set random state for reproducibility

# train a RandomForestClassifier with cross-validation
rf_model = RandomForestClassifier(random_state=42)
score_kfold = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='accuracy')

print("\nK-Fold Cross-Validation Accuracy Scores:", score_kfold)
print("Mean K-Fold CV Accuracy:", score_kfold.mean())

#initialize stratified k-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # 5-fold stratified cross-validation | shuffle the data before splitting | set random state for reproducibility

# train a RandomForestClassifier with stratified cross-validation
score_stratified = cross_val_score(rf_model, X_train, y_train, cv=skf, scoring='accuracy')

print("\nStratified K-Fold Cross-Validation Accuracy Scores:", score_stratified)
print("Mean Stratified K-Fold CV Accuracy:", score_stratified.mean())

