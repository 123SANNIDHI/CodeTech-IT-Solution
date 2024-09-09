import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#loading a dataset to pandas library
credit_card_data = pd.read_csv('creditcard.csv')
credit_card_data.info() #information about credit_card_data
credit_card_data.isnull().sum()#CHECKING THE NUMBER OF MISSING VALUES IN THE COLUMN
#distrubution of legit transcation and fraudulent transaction
credit_card_data['Class'].value_counts()
#this dataset is highly unbalanced 0-->normal transcation 1-->fraudulent transcation

#seperate data for analysis
legit=credit_card_data[credit_card_data.Class==0]
fraud=credit_card_data[credit_card_data['Class']==1]
fraud['Class']
legit.shape
fraud.shape
#stastical measures of data
legit.Amount.describe()
fraud.Amount.describe()
#comapare the values for two difeerent classes
credit_card_data.groupby('Class').mean()
#under sampling
#Build a sample dataset conatining similar distrubution of normal transaction and fraudulent transaction
#number of fraudulent trascation--492
#take out the random sample from legit ,then legit and fraudulent become balanced
legit_sample=legit.sample(n=492)
print(legit_sample)
#conactinating the dataframe 
new_df=pd.concat([legit_sample,fraud],axis=0)
new_df.head()
new_df['Class'].value_counts()
new_df.groupby('Class').mean()
#splitting data into features and targets
X=new_df.drop(columns='Class',axis=1)
Y=new_df['Class']

#spilt the data into training and testing data

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
#X.shape,X_train.shape,X_test.shape

#model training
#linear regression
model=LogisticRegression()
#training the logistic regression model and training data
model.fit(X_train,Y_train) 
X_train_prediction=model.predict(X_test)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on Test Data : ', test_data_accuracy)
