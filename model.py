import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
    StackingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


df = pd.read_csv("CarPrice_Assignment.csv")


df.drop_duplicates(inplace=True)

df = df.drop(columns=["car_ID"])

df["Brand"] = df["CarName"].apply(lambda x: x.split()[0])

df = df.drop(columns=["CarName"])


x = df.drop(columns=["price"])
y = df["price"]

cat_col = x.select_dtypes(include=["object"]).columns
num_col = x.select_dtypes(include=["int64", "float64"]).columns



cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])

preprocessing = ColumnTransformer([
    ("cat", cat_pipeline, cat_col),
    ("num", num_pipeline, num_col)
])


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.25,
    random_state=42
)


lr_model = LinearRegression()

rf_model = RandomForestRegressor(
    n_estimators=50,
    max_depth=2,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

xgb_model = GradientBoostingRegressor(
    learning_rate=0.01,
    n_estimators=100,
    max_depth=2,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

voting_model = VotingRegressor([
    ("lr", lr_model),
    ("rf", rf_model),
    ("xgb", xgb_model)
])

stacking_model = StackingRegressor(
    estimators=[
        ("rf", rf_model),
        ("xgb", xgb_model)
    ],
    final_estimator=Ridge()
)



models = {
    "LinearRegression": lr_model,
    "RandomForestRegressor": rf_model,
    "GradientBoostingRegressor": xgb_model,
    "VotingRegressor": voting_model,
    "StackingRegressor": stacking_model
}

results = []

for name, model in models.items():

    pipe = Pipeline([
        ("preprocessing", preprocessing),
        ("model", model)
    ])

    pipe.fit(x_train, y_train)

    pred = pipe.predict(x_test)

    results.append({
        "Model": name,
        "R2": r2_score(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred))
    })

result_df = pd.DataFrame(results)

best_model_name = result_df.sort_values(
    by="R2",
    ascending=False
).iloc[0]["Model"]

print("Best Model:", best_model_name)

best_model = models[best_model_name]


final_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("model", best_model)
])

final_pipeline.fit(x_train, y_train)

pred = final_pipeline.predict(x_test)

print("Final R2 Score:", r2_score(y_test, pred))


with open("model.pkl", "wb") as file:
    pickle.dump(final_pipeline, file)

print("Model Saved Successfully!")