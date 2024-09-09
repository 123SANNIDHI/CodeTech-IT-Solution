
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler

data = pd.read_csv("creditcard.csv")
data.head()
pd.options.display.max_columns = None
data.head()

data.shape

print("Number of columns: {}".format(data.shape[1]))
print("Number of rows: {}".format(data.shape[0]))

data.info()

data.isnull().sum()
sc = StandardScaler()
data['Amount'] = sc.fit_transform(pd.DataFrame(data['Amount']))

data.head()

data = data.drop(['Time'], axis =1)
data.head()

data.duplicated().any()

data = data.drop_duplicates()
data.shape

data['Class'].value_counts()

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
sns.countplot(data['Class'])
plt.show()
X = data.drop('Class', axis = 1)
y=data['Class']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
classifier = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree Classifier": DecisionTreeClassifier()
}

for name, clf in classifier.items():
    print(f"\n=========={name}===========")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"\n Accuaracy: {accuracy_score(y_test, y_pred)}")
    print(f"\n Precision: {precision_score(y_test, y_pred)}")
    print(f"\n Recall: {recall_score(y_test, y_pred)}")
    print(f"\n F1 Score: {f1_score(y_test, y_pred)}")
    normal = data[data['Class']==0]
fraud = data[data['Class']==1]
normal.shape

fraud.shape
normal_sample = normal.sample(n=473)
normal_sample.shape
new_data = pd.concat([normal_sample,fraud], ignore_index=True)
new_data.head()