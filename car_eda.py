import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("=== Car Data Analysis ===")

# 1. Load dataset
df = pd.read_csv("car_data.csv")

# 2. Check shape of data
print("\n1. Dataset Shape:")
print(df.shape)

# 3. Check data types
print("\n2. Column Data Types:")
df.info()

# 4. Check for missing values
print("\n3. Missing Values Count:")
print(df.isnull().sum())

# 5. Check statistical summary
print("\n4. Statistical Summary:")
print(df.describe())

# 6. Generate price distribution plot
print("\n5. Generating Plot...")
plt.figure(figsize=(8, 5))
sns.histplot(df['Selling_Price'], kde=True, color='purple')
plt.title('Car Selling Price Distribution')
plt.xlabel('Selling Price')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
