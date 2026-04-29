import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("data/vgsales.csv")

print("Original shape:", df.shape)


df = df[[
    "Platform",
    "Genre",
    "Publisher",
    "Year",
    "Global_Sales"
]]

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

print("Missing values before cleaning:\n", df.isnull().sum())

# X = df.drop("Global_Sales", axis=1)
# y = df["Global_Sales"]

# # Feature types
# categorical_features = ["Platform", "Genre", "Publisher"]
# numeric_features = ["Year"]

# # Pipelines
# categorical_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="most_frequent")),
#     ("encoder", OneHotEncoder(handle_unknown="ignore"))
# ])

# numeric_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="median"))
# ])

# # Combine preprocessing
# preprocessor = ColumnTransformer([
#     ("cat", categorical_pipeline, categorical_features),
#     ("num", numeric_pipeline, numeric_features)
# ])

# # Model pipeline
# model = Pipeline([
#     ("preprocessor", preprocessor),
#     ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
# ])

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Train
# model.fit(X_train, y_train)

# # Evaluate
# predictions = model.predict(X_test)

# mae = mean_absolute_error(y_test, predictions)
# r2 = r2_score(y_test, predictions)

# print("Model V2 Results:")
# print("MAE:", mae)
# print("R2 Score:", r2)

# # Save model
# joblib.dump(model, "model/game_sales_model_v2.pkl")

# print("Improved model saved!")