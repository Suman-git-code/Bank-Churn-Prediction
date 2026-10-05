"""Train the churn model and save it as churn_model.joblib (same steps as the notebook)."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

folder = Path(__file__).parent

df = pd.read_csv(folder / "Churn_Modelling.csv")
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
df["ZeroBalance"] = (df["Balance"] == 0).astype(int)

numeric_features = ["CreditScore", "Age", "Tenure", "Balance",
                    "EstimatedSalary", "NumOfProducts"]
categorical_features = ["Geography", "Gender"]
binary_features = ["HasCrCard", "IsActiveMember", "ZeroBalance"]

X = df[numeric_features + categorical_features + binary_features]
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(drop="first"), categorical_features),
    ("bin", "passthrough", binary_features),
])

ratio = (y_train == 0).sum() / (y_train == 1).sum()

xgb = Pipeline([
    ("prep", preprocessor),
    ("model", XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                            scale_pos_weight=ratio, eval_metric="logloss",
                            random_state=42)),
])
xgb.fit(X_train, y_train)

joblib.dump(xgb, folder / "churn_model.joblib")
print("Saved:", folder / "churn_model.joblib")
