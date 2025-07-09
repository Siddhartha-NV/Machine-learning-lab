import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, precision_score, accuracy_score
from sklearn.metrics import PrecisionRecallDisplay

from imblearn.over_sampling import SMOTE

df = pd.read_csv("creditcard.csv")

print("--- Dataset Information ---")
print(df.info())

plt.figure(figsize=(7, 5))
sns.countplot(x='Class', data=df)
plt.title('Class Distribution (0: Not Fraud, 1: Fraud)')
plt.xlabel('Class')
plt.ylabel('Number of Transactions')
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))

sns.histplot(df['Time'], bins=50, kde=True, ax=ax1)
ax1.set_title('Distribution of Transaction Time')
ax1.set_xlabel('Time (in seconds)')

sns.histplot(df['Amount'], bins=50, kde=True, ax=ax2)
ax2.set_title('Distribution of Transaction Amount')
ax2.set_xlabel('Amount (in Euros)')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Class', y='Amount', data=df)
plt.title('Transaction Amount by Class')
plt.xlabel('Class (0: Not Fraud, 1: Fraud)')
plt.ylabel('Transaction Amount')
plt.ylim(0, 500)
plt.show()

sample_df = df.sample(frac=0.1, random_state=42)
plt.figure(figsize=(20, 15))
sns.heatmap(sample_df.corr(), cmap='coolwarm')
plt.title('Correlation Heatmap of Features (on a 10% sample)')
plt.show()



scaler = StandardScaler()
df['scaled_Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

df.drop(['Time', 'Amount'], axis=1, inplace=True)
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

print("\nApplying SMOTE to the training data...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\n--- Class distribution after SMOTE ---")
print(y_train_resampled.value_counts())


print("\n--- Training Logistic Regression Model ---")
log_reg = LogisticRegression(random_state=42, max_iter=200)
log_reg.fit(X_train_resampled, y_train_resampled)
y_pred_lr = log_reg.predict(X_test)

print("\n--- Logistic Regression Results ---")
print(classification_report(y_test, y_pred_lr))

print("\n--- Training Random Forest Model ---")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_clf.fit(X_train_resampled, y_train_resampled)
y_pred_rf = rf_clf.predict(X_test)

print("\n--- Random Forest Results ---")
print(classification_report(y_test, y_pred_rf))


cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(7, 5))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(7, 5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

fig, ax = plt.subplots(figsize=(10, 7))

display_lr = PrecisionRecallDisplay.from_estimator(log_reg, X_test, y_test, name='Logistic Regression', ax=ax)

display_rf = PrecisionRecallDisplay.from_estimator(rf_clf, X_test, y_test, name='Random Forest', ax=ax)

plt.title('Precision-Recall Curve Comparison')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='best')
plt.grid(True)
plt.show()