import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("=== Sales Prediction ===")

# 1. Load dataset
df = pd.read_csv("Advertising.csv")

# 2. Select features and target
features = ['TV', 'Radio', 'Newspaper']
target = 'Sales'

# 3. Drop missing values
df_clean = df[features + [target]].dropna()
X = df_clean[features]
y = df_clean[target]

# 4. Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the model
model = LinearRegression()
print("\n1. Training the Model...")
model.fit(X_train, y_train)

# 6. Evaluate model performance
y_pred = model.predict(X_test)
print("\n2. Model Evaluation:")
print(f"Accuracy (R2 Score): {r2_score(y_test, y_pred):.4f}")
print(f"Error (MSE): {mean_squared_error(y_test, y_pred):.2f}")

# 7. Make a sample prediction
print("\n3. Sample Prediction:")
sample_input = pd.DataFrame([[150.0, 25.0, 10.0]], columns=features)
prediction = model.predict(sample_input)
print(f"Predicted Sales: {prediction[0]:.2f}")
