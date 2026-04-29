import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("data/vgsales.csv")

df.head()
#getting the necessary columns
columns = [
    "Platform",
    "Genre",
    "Publisher",
    "NA_Sales",
    "EU_Sales",
    "JP_Sales",
    "Other_Sales",
    "Global_Sales"
]

df = df[columns]

#dropping missing values if they exist
df = df.dropna()

X = df.drop("Global_Sales", axis=1)
y = df["Global_Sales"]

categorical_features = ["Platform", "Genre", "Publisher"]
numeric_features = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Model trained successfully!")
print("MAE:", mae)
print("R2 Score:", r2)

joblib.dump(model, "model/game_sales_model.pkl")

print("Model saved to model/game_sales_model.pkl")