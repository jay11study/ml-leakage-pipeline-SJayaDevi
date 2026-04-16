# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# Task 1: Create Synthetic Dataset
# -------------------------------

np.random.seed(42)

n = 60  # at least 50 records

area_sqft = np.random.randint(500, 3000, n)
num_bedrooms = np.random.randint(1, 5, n)
age_years = np.random.randint(0, 30, n)

# Create price with some logic + noise
price_lakhs = (
    area_sqft * 0.05 +
    num_bedrooms * 10 -
    age_years * 0.3 +
    np.random.normal(0, 5, n)
)

# Create DataFrame
df = pd.DataFrame({
    "area_sqft": area_sqft,
    "num_bedrooms": num_bedrooms,
    "age_years": age_years,
    "price_lakhs": price_lakhs
})

print("Dataset Preview:\n", df.head())

# -------------------------------
# Build Regression Model
# -------------------------------

X = df[["area_sqft", "num_bedrooms", "age_years"]]
y = df["price_lakhs"]

model = LinearRegression()
model.fit(X, y)

# Print intercept and coefficients
print("\nIntercept:", model.intercept_)
print("Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

# Predictions
y_pred = model.predict(X)

# Show first 5 actual vs predicted
comparison = pd.DataFrame({
    "Actual": y.head(),
    "Predicted": y_pred[:5]
})

print("\nFirst 5 Actual vs Predicted:\n", comparison)


# -------------------------------
# Task 2: Model Evaluation
# -------------------------------

mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print("\nModel Evaluation Metrics:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)

# Explanation:
# MAE shows the average absolute error between predicted and actual prices.
# RMSE penalizes larger errors more heavily, giving insight into model accuracy.
# R² indicates how well the model explains variance (closer to 1 = better fit).


# -------------------------------
# Task 3: Residual Analysis
# -------------------------------

residuals = y - y_pred

plt.figure()
plt.hist(residuals, bins=15)
plt.title("Residuals Distribution")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

# Explanation:
# Residuals are the differences between actual and predicted values.
# A symmetric, bell-shaped histogram suggests the model errors are normally distributed,
# indicating a well-fitted regression model without major bias.
