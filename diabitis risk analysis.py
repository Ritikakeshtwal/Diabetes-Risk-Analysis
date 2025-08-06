import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# data load
df = pd.read_csv("diabetes.csv")
print(df.head())

#Basic information 
print(df.info())
print(df.describe())
print(df.isnull().sum())

# check distribution 
sns.countplot(x='Outcome', data=df)
plt.title("Diabetes Outcome (0 = No, 1 = Yes)")
plt.show()

#Glucose vs Age
plt.figure(figsize=(8,6))
sns.scatterplot(x='Age', y='Glucose', hue='Outcome', data=df)
plt.title('Glucose Level vs Age')
plt.show()


#BMI patterns 
plt.figure(figsize=(8,6))
sns.histplot(df['BMI'], kde=True, bins=30)
plt.title('BMI Distribution')
plt.show()

# heatmap for correlation 

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()