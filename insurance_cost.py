import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    df = pd.read_csv("insurance.csv")
    print(df.head())
    print(df.info())
    print(df.describe())

    df['sex'] = df['sex'].map({'male': 1, 'female': 0})
    df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
    df = pd.get_dummies(df, columns=['region'], drop_first=True)

    sns.boxplot(x="smoker",y="charges", data=df)
    plt.title("Charges by Smoking Status")
    plt.show()

    sns.scatterplot(x="bmi", y="charges", data=df)
    plt.title("BMI vs Charges")
    plt.show()

    X = df.drop("charges", axis=1)
    y = df["charges"]

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual Charges")
    plt.ylabel("Predicted Charges")
    plt.title("Predicted vs Actual Charges")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.show()

    mae = mean_absolute_error(y_test, y_pred)

    print("MAE:", round(mae, 2))
    print("R²:", round(r2_score(y_test, y_pred),3))