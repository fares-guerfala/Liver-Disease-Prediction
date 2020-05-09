# -*- coding: utf-8 -*-
"""
Created on Sat May  9 10:13:57 2020

@author: Fares Guerfala
"""
##Import all required libraries
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from pandas_profiling import ProfileReport


#Read the training & test data and create a report file englobe all details about our data as a html file
df=pd.read_csv('liver_patient.csv', encoding= 'unicode_escape')
profile = ProfileReport(df,title='Pandas EDA')
profile.to_file(output_file='my_report.html')

#------------------------data analysis------------------------------------------------------
print(df.columns)
print(df.info())#Only gender is non-numeric veriable and all others are numerical variables
print(df.describe(include='all'))
#Check for any null values
print(df.isnull().sum())
df=df.dropna()#drop missing values
print(df.shape)

#-------------------Data Visualization--------------------------------------------------------

sns.countplot(data=df, x = 'Dataset', label='Count')
plt.show()
LD, NLD = df['Dataset'].value_counts()
print('The number of patients diagnosed with liver disease: ',LD)
print('The number of patients not diagnosed with liver disease: ',NLD)
#Create new categorical column 
print('Number of people in Dataset 1',df[df['Dataset'] == 1].Age.count())
print('Number of people in Dataset 2',df[df['Dataset'] == 2].Age.count())

# Gender Distribution of 2 Dataset
sns.countplot(data=df, x = 'Gender', label='Count',palette="Set1")
plt.title('Distribution of Datasets by Gender')
plt.show()


gender = sns.FacetGrid(df, col="Dataset", row="Gender", margin_titles=True)
gender.map(plt.hist, "Age", color="red")
plt.subplots_adjust(top=0.9)
gender.fig.suptitle('Disease by Agefor each Gender')
plt.show()

#Distribution of features in the whole dataset
columns=list(df.columns[:10])
columns.remove('Gender')
plt.subplots(figsize=(18,15))
length=len(columns)
for i,j in itertools.zip_longest(columns,range(length)):
    plt.subplot((length/2),3,j+1)
    plt.subplots_adjust(wspace=0.2,hspace=0.5)
    df[i].hist(bins=10,edgecolor='black')#,range=(0,0.3))
    plt.title(i)
plt.show()


sns.pairplot(
    data=df,
    hue='Dataset',
    vars=['Age', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
       'Alamine_Aminotransferase', 'Aspartate_Aminotransferase',
       'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio']
)
plt.subplots_adjust(top=0.9)

#From the above jointplots and scatterplots, we find direct relationship between the following features:
#Direct_Bilirubin & Total_Bilirubin
#Aspartate_Aminotransferase & Alamine_Aminotransferase
#Total_Protiens & Albumin
#Albumin_and_Globulin_Ratio & Albumin
#So, we can very well find that we can omit one of the features. I'm going to keep the follwing features:
#Total_Bilirubin/Alamine_Aminotransferase/Total_Protiens/Albumin_and_Globulin_Ratio/Albumin

#Convert categorical variable "Gender" to indicator variables
pd.get_dummies(df['Gender'], prefix = 'Gender')
df = pd.concat([df,pd.get_dummies(df['Gender'], prefix = 'Gender')], axis=1)
print(df.head())
#we find that the numbers of Albumin_and_Globulin_Ratio are ubnormal so we should clean it
df[df['Albumin_and_Globulin_Ratio'].isnull()]
df["Albumin_and_Globulin_Ratio"] = df.Albumin_and_Globulin_Ratio.fillna(df['Albumin_and_Globulin_Ratio'].mean())

X = df.drop(['Gender','Dataset'], axis=1)
y = df['Dataset'] 
#correlation
liver_corr = X.corr()
plt.figure(figsize=(30, 30))
sns.heatmap(liver_corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
           cmap= 'coolwarm')
plt.title('Correlation between features');



#-----------------------------Machine Learning---------------------------------------------------------

# Importing modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn.metrics import classification_report,confusion_matrix


scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(X)
#Split the dataset
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)
#Train the model
model=XGBClassifier()
model.fit(x_train,y_train)
model_predicted = model.predict(x_test)

Rf=RandomForestClassifier(n_estimators=100)
Rf.fit(x_train,y_train)
Rf_predicted = Rf.predict(x_test)

Lr=LogisticRegression()
Lr.fit(x_train,y_train)
Lr_predicted = Lr.predict(x_test)

clf=neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)
clf_predicted = clf.predict(x_test)

svc_linear=SVC()
svc_linear.fit(x_train,y_train)
svc_linear_predicted = svc_linear.predict(x_test)

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
gauss_predicted = gaussian.predict(x_test)

#Calculate the accuracy
print ("XGB Classifier accuracy :",model.score(x_test,y_test),confusion_matrix(y_test,model_predicted),
classification_report(y_test,model_predicted))
print ("Random Forest Classifier accuracy :",Rf.score(x_test,y_test),confusion_matrix(y_test,Rf_predicted),
classification_report(y_test,Rf_predicted))
print ("Logistic Regression accuracy :",Lr.score(x_test,y_test),confusion_matrix(y_test,Lr_predicted),
classification_report(y_test,Lr_predicted))
print ("KNeighborsClassifier accuracy :",clf.score(x_test,y_test),confusion_matrix(y_test,clf_predicted),
classification_report(y_test,clf_predicted))
print ("SVC accuracy :",svc_linear.score(x_test,y_test),confusion_matrix(y_test,svc_linear_predicted),classification_report(y_test,gauss_predicted))
print ("GaussianNB accuracy :",gaussian.score(x_test,y_test),confusion_matrix(y_test,gauss_predicted),
classification_report(y_test,gauss_predicted))


