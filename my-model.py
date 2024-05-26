import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings('ignore')
sns.set_style("whitegrid", {'axes.grid': False})

# Load Dataset
df = pd.read_csv(r"C:\Users\Tarun\SmartCrop\SmartCrop-Dataset.csv")

# Select only numeric columns for outlier removal
numeric_df = df.select_dtypes(include=[np.number])

# Remove Outliers
Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1
outlier_condition = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)
df = df[~outlier_condition]

# Split Data
target = 'label'
X = df.drop(target, axis=1)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train Model
pipeline = make_pipeline(StandardScaler(), GaussianNB())
model = pipeline.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
print(f"Training Accuracy Score: {model.score(X_train, y_train) * 100:.1f}%")
print(f"Validation Accuracy Score: {model.score(X_test, y_test) * 100:.1f}%")
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap='YlGnBu', fmt='g')
plt.title('Confusion Matrix', fontsize=20)
plt.ylabel('Actual label', fontsize=15)
plt.xlabel('Predicted label', fontsize=15)
plt.show()

print(classification_report(y_test, y_pred))

# Save Model
with open('my-model.pkl', 'wb') as file:
    pickle.dump(model, file)
