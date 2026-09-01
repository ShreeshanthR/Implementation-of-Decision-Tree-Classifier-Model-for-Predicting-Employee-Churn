# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess the employee dataset.
2. Split the data into training and testing sets.
3. Initialize and train the Decision Tree Classifier.
4. Predict employee churn and calculate accuracy.
5. Display the trained Decision Tree.

## Program:
```python
'''
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Shreeshanth R
RegisterNumber: 212225040411
'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


data = pd.read_csv("Employee.csv")

data = pd.get_dummies(data, drop_first=True)


X = data.iloc[:, :-1]   
y = data.iloc[:, -1]    


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = DecisionTreeClassifier(criterion='gini', random_state=42)

model.fit(X_train, y_train)


plt.figure(figsize=(25,12))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=[str(i) for i in model.classes_],
    filled=True
)

plt.title("Decision Tree Classifier")
plt.show()


```

## Output:

<img width="1758" height="820" alt="image" src="https://github.com/user-attachments/assets/36d1d007-5473-4a4d-bc18-dc0741532100" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
