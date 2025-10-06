# insurance_experiments.py
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

if __name__ == '__main__':

    df = pd.read_csv("insurance.csv")
    print(df.head())
    print(df.info())
    print(df.describe())

    # Encode categorical
    df['sex'] = df['sex'].map({'male': 1, 'female': 0})
    df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
    df = pd.get_dummies(df, columns=['region'], drop_first=True)

    X = df.drop("charges", axis=1)
    y = df["charges"]

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

    # Models
    linear = LinearRegression()
    poly = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    models = {"Linear": linear, "Polynomial (deg=2)": poly, "Random Forest": rf}

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    results_test = []
    results_cv = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_r2 = r2_score(y_train, y_pred_train)

        cv_mae = -cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv).mean()
        cv_rmse = -cross_val_score(model, X_train, y_train, scoring='neg_root_mean_squared_error', cv=cv).mean()
        cv_r2 = cross_val_score(model, X_train, y_train, scoring='r2', cv=cv).mean()

        results_test.append([name, train_mae, train_rmse, train_r2])
        results_cv.append([name, cv_mae, cv_rmse, cv_r2])

    result_test_df = pd.DataFrame(results_test, columns=["Model", "Train MAE", "Train RMSE", "Train R2"])
    result_df = pd.DataFrame(results_cv, columns=["Model", "CV MAE", "CV RMSE", "CV R2"])
    print("\nTest results:\n", result_test_df)
    print("\nCross-validation results:\n", result_df)

    # Select best (by CV MAE)
    best = result_df.loc[result_df["CV MAE"].idxmin(), "Model"]
    print("\nBest model by CV MAE:", best)

    final_model = models[best]
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nFinal test performance:")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R²:", round(r2, 3))

    # Plot predicted vs actual
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual Charges")
    plt.ylabel("Predicted Charges")
    plt.title(f"Predicted vs Actual Charges ({best})")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.show()
