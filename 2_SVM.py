import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib


data = pd.read_csv("isl_2hands_LM.csv", low_memory=False)


data.iloc[:, -1] = data.iloc[:, -1].astype(str)


X = data.iloc[:, :-1]
data = data[X.sum(axis=1) != 0]


X = data.iloc[:, :-1]
y = data.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = SVC(kernel="rbf")
model.fit(X_train, y_train)


accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# save model
joblib.dump(model, "isl_2hand_model.pkl")
joblib.dump(scaler, "isl_2hand_scaler.pkl")

print("Model saved successfully")