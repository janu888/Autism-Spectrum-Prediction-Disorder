import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
file_path = "autism_data.csv"
df = pd.read_csv(file_path)
df.dropna(inplace=True)

features = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
            'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
            'gender', 'jundice', 'austim']
target = 'Class/ASD'

df = df[features + [target]]

encoder = LabelEncoder()
df['gender'] = encoder.fit_transform(df['gender'])
df['jundice'] = encoder.fit_transform(df['jundice'])
df['austim'] = encoder.fit_transform(df['austim'])
df[target] = encoder.fit_transform(df[target])

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

if not os.path.exists("static"):
    os.makedirs("static")

# Feature importance plot
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]

plt.figure(figsize=(10, 5))
plt.bar(range(len(features)), feature_importance[sorted_idx], align="center")
plt.xticks(range(len(features)), np.array(features)[sorted_idx], rotation=45)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("static/plot.png")
plt.close()

# Accuracy
accuracy = accuracy_score(y_test, model.predict(X_test)) * 100
with open("accuracy.txt", "w") as f:
    f.write(f"{accuracy:.2f}")

print("✅ Model trained and saved as model.pkl")
print("✅ Accuracy saved to accuracy.txt")
print("✅ Feature importance saved to static/plot.png")
