# Module 7: Data Analysis with Python - NumPy and Pandas
# Complete Combined Code for Jupyter Notebook

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("All libraries imported successfully!")

# =============================================================================
# NUMPY EXERCISES
# =============================================================================

print("\n" + "="*60)
print("NUMPY EXERCISES")
print("="*60)

# Q.1 Convert a 1D array to a 2D array with 2 rows
print("\nQ.1 Convert 1D array to 2D array with 2 rows")
arr_1d = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print("Original 1D array:", arr_1d)
arr_2d = arr_1d.reshape(2, -1)
print("2D array with 2 rows:")
print(arr_2d)

# Q.2 Get the common items between a and b
print("\nQ.2 Common items between arrays a and b")
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
print("Array a:", a)
print("Array b:", b)
common_items = np.intersect1d(a, b)
print("Common items:", common_items)

# Q.3 Get all items between 5 and 10 from a
print("\nQ.3 Items between 5 and 10 from array a")
a = np.array([2, 6, 1, 9, 10, 3, 27])
print("Array a:", a)
items_between = a[(a >= 5) & (a <= 10)]
print("Items between 5 and 10 (inclusive):", items_between)

# Q.4 Limit the number of items printed in python NumPy array
print("\nQ.4 Limit printed items in NumPy array")
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
print("Original array:", arr)
np.set_printoptions(threshold=6)
print("Array with limited display:")
print(arr)
np.set_printoptions(threshold=None)

# =============================================================================
# PANDAS EXERCISES
# =============================================================================

print("\n" + "="*60)
print("PANDAS EXERCISES")
print("="*60)

# 1. Compute the minimum, 25th percentile, median, 75th, and maximum of series
print("\n1. Series Statistics")
ser = pd.Series([12, 15, 18, 22, 25, 28, 31, 35, 40, 45, 50, 55, 60, 65, 70])
print("Series:", ser.values)
stats = {
    'Minimum': ser.min(),
    '25th Percentile': ser.quantile(0.25),
    'Median': ser.median(),
    '75th Percentile': ser.quantile(0.75),
    'Maximum': ser.max()
}
for stat, value in stats.items():
    print(f"{stat}: {value}")

# 2. Creating A Pandas Data Frame and Using Sample Data Sets
print("\n2. Creating DataFrames")
data_dict = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'City': ['New York', 'London', 'Tokyo', 'Paris'],
    'Salary': [50000, 60000, 70000, 55000]
}
df_dict = pd.DataFrame(data_dict)
print("DataFrame from dictionary:")
print(df_dict)

# 3. Using NumPy, create a Pandas Data Frame with five rows and three columns
print("\n3. DataFrame from NumPy array")
np_array = np.random.rand(5, 3) * 100
df_from_np = pd.DataFrame(np_array, columns=['Column_A', 'Column_B', 'Column_C'])
print("DataFrame from NumPy array:")
print(df_from_np)

# 4. Default behavior for DataFrame labels
print("\n4. Default DataFrame Labels")
sample_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
sample_df = pd.DataFrame(sample_array)
print("Sample DataFrame:")
print(sample_df)
print("\nDefault column labels:", sample_df.columns.tolist())
print("Default row labels (index):", sample_df.index.tolist())

# 5-8. Working with CSV File and DataFrame Operations
print("\n5-8. Working with Large Dataset")

# Create sample dataset with 10,000+ rows and 12 columns
np.random.seed(42)
n_rows = 10000
sample_large_data = {
    'id': range(1, n_rows + 1),
    'numeric_1': np.random.normal(100, 15, n_rows),
    'numeric_2': np.random.exponential(2, n_rows),
    'numeric_3': np.random.randint(1, 100, n_rows),
    'numeric_4': np.random.uniform(0, 1, n_rows),
    'numeric_5': np.random.poisson(5, n_rows),
    'text_1': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
    'text_2': np.random.choice(['Yes', 'No'], n_rows),
    'text_3': np.random.choice(['Low', 'Medium', 'High'], n_rows),
    'text_4': np.random.choice(['Type1', 'Type2', 'Type3', 'Type4'], n_rows),
    'category_1': np.random.choice(['Cat1', 'Cat2', 'Cat3'], n_rows),
    'category_2': np.random.choice(['GroupA', 'GroupB'], n_rows)
}

large_df = pd.DataFrame(sample_large_data)
print("Sample large DataFrame created!")

# 6. Show number of rows and columns
print(f"\n6. DataFrame shape - Rows: {large_df.shape[0]}, Columns: {large_df.shape[1]}")

# 7. Show first few rows
print("\n7. First few rows of DataFrame:")
print(large_df.head())

# 8. Type when selecting single column
single_column = large_df['numeric_1']
print(f"\n8. Type when selecting single column: {type(single_column)}")

# =============================================================================
# VISUALIZATION EXERCISES
# =============================================================================

print("\n" + "="*60)
print("VISUALIZATION EXERCISES")
print("="*60)

# 9. Create a line plot
print("\n9. Line Plot")
plt.figure(figsize=(8, 6))
x = [3, 4, 5, 6]
y = [1.5, 2, 2.5, 3]
plt.plot(x, y, marker='o', linewidth=2, markersize=8)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Line Plot: Y vs X')
plt.grid(True, alpha=0.3)
plt.show()

# 10. Plot multiple functions
print("\n10. Multiple Function Plot")
x = np.arange(0, 6.1, 0.3)
plt.figure(figsize=(10, 8))
plt.plot(x, x, 'ro', label='x', markersize=4)
plt.plot(x, x**2, 'bs', label='x²', markersize=4)
plt.plot(x, x**3, 'g', label='x³', linewidth=2)
plt.plot(x, x**4, ':', label='x⁴', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Multiple Functions Plot')
plt.legend()
plt.xlim(0, 6)
plt.ylim(0, 125)
plt.grid(True, alpha=0.3)
plt.show()

# 11. Create a bar plot
print("\n11. Bar Plot - Height Comparison")
height = [179, 155, 191, 152, 188, 177]
names = ['QA', 'WB', 'EC', 'RD', 'TE', 'YF']
plt.figure(figsize=(10, 6))
bars = plt.bar(names, height, color=['skyblue', 'lightcoral', 'lightgreen', 
                                    'gold', 'lightpink', 'lightsteelblue'])
plt.xlabel('Individuals')
plt.ylabel('Height (cm)')
plt.title('Height Comparison')
plt.ylim(150, 200)
for bar in bars:
    height_val = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height_val,
             f'{int(height_val)}', ha='center', va='bottom')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# 12. Plot histograms with different bin sizes
print("\n12. Histograms with Normal Distribution")
np.random.seed(42)
x = np.random.randn(100000)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].hist(x, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
axes[0].set_title('Histogram with 10 Bins')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')
axes[0].grid(True, alpha=0.3)
axes[1].hist(x, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
axes[1].set_title('Histogram with 20 Bins')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Frequency')
axes[1].grid(True, alpha=0.3)
axes[2].hist(x, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
axes[2].set_title('Histogram with 50 Bins')
axes[2].set_xlabel('Value')
axes[2].set_ylabel('Frequency')
axes[2].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*60)
print("SUMMARY AND ADDITIONAL ANALYSIS")
print("="*60)

print("\nData Types in Large DataFrame:")
print(large_df.dtypes)

print("\nBasic Statistics for Numerical Columns:")
print(large_df.select_dtypes(include=[np.number]).describe())

print("\nMemory usage of large DataFrame:")
print(large_df.memory_usage(deep=True))

print("\nAssignment completed successfully!")