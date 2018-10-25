# coding: utf-8
# In[31]:
#data source: uci machine learning repository: https://archive.ics.uci.edu/ml/datasets/Absenteeism+at+work
#This is a logistic regression used to predict whether or not an employee will be absent from work
#import modules necessary for analysis
import pandas as pd #used to create dataframes and manipulate data
import statsmodels.api as smf #used to run regressions
import numpy as np #used to process data
from sklearn import preprocessing
import matplotlib.pyplot as plt #used to visualize data
plt.rc("font", size=16)#set font and size for matplotlib to visualize
from sklearn.linear_model import LogisticRegression #used for logistic regressions
from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
#load data from .csv file
data = pd.read_csv("/Users/NickSummers/Desktop/DataScience/Work_Absenteeism/Absenteeism/Absenteeism_at_work.csv")
#see values of variable of interest
data['Absenteeism time in hours'].unique()
#set certain values in our analysis to 0 or 1 to perform logistic regression
#0 hours absent will be 0 and anything else will be 1
#nonzero abseentee values: 4,   0,   2,   8,  40,   1,   7,   3,  32,   5,  16,  24,  64, 56,  80, 120, 112, 104,  48
data['Absenteeism time in hours']=np.where(data['Absenteeism time in hours'] ==0, 0, data['Absenteeism time in hours'])
data['Absenteeism time in hours']=np.where(data['Absenteeism time in hours'] ==1, 1, data['Absenteeism time in hours'])
data['Absenteeism time in hours']=np.where(data['Absenteeism time in hours'] ==4, 1, data['Absenteeism time in hours'])
data['Absenteeism time in hours']=np.where(data['Absenteeism time in hours'] ==2, 1, data['Absenteeism time in hours'])
data['Absenteeism time in hours']=np.where(data['Absenteeism time in hours'] ==8, 1, data['Absenteeism time in hours'])
data['Absenteeism time in hours']=np.where(data['Absenteeism time in hours'] ==40, 1, data['Absenteeism time in hours'])
data['Absenteeism time in hours']=np.where(data['Absenteeism time in hours'] ==7, 1, data['Absenteeism time in hours'])
data['Absenteeism time in hours']=np.where(data['Absenteeism time in hours'] ==3, 1, data['Absenteeism time in hours'])
data['Absenteeism time in hours']=np.where(data['Absenteeism time in hours'] ==32, 1, data['Absenteeism time in hours'])
data['Absenteeism time in hours']=np.where(data['Absenteeism time in hours'] ==5, 1, data['Absenteeism time in hours'])
data['Absenteeism time in hours']=np.where(data['Absenteeism time in hours'] ==16, 1, data['Absenteeism time in hours'])
data['Absenteeism time in hours']=np.where(data['Absenteeism time in hours'] ==24, 1, data['Absenteeism time in hours'])
data['Absenteeism time in hours']=np.where(data['Absenteeism time in hours'] ==64, 1, data['Absenteeism time in hours'])
data['Absenteeism time in hours']=np.where(data['Absenteeism time in hours'] ==56, 1, data['Absenteeism time in hours'])
data['Absenteeism time in hours']=np.where(data['Absenteeism time in hours'] ==80, 1, data['Absenteeism time in hours'])
data['Absenteeism time in hours']=np.where(data['Absenteeism time in hours'] ==120, 1, data['Absenteeism time in hours'])
data['Absenteeism time in hours']=np.where(data['Absenteeism time in hours'] ==112, 1, data['Absenteeism time in hours'])
data['Absenteeism time in hours']=np.where(data['Absenteeism time in hours'] ==104, 1, data['Absenteeism time in hours'])
data['Absenteeism time in hours']=np.where(data['Absenteeism time in hours'] ==48, 1, data['Absenteeism time in hours'])
#now, we can see that abseentee is binary (0: not absent, 1:absent)
data['Absenteeism time in hours'].unique()
data['Absenteeism time in hours'].value_counts()
#visualize the data
sns.countplot(x='Absenteeism time in hours',data=data,palette='hls') #create bar graph of absenteeism
plt.show() #show figure
plt.savefig('absent_0_1')#save figure
#calculate some statistics for our data
count_not_absent = len(data[data['Absenteeism time in hours']==0])
count_absent = len(data[data['Absenteeism time in hours']==1])
pct_of_not_absent = count_not_absent/(count_not_absent+count_absent)
print("percentage of people not absent is", pct_of_not_absent*100)
pct_of_absent = count_absent/(count_not_absent+count_absent)
print("percentage of absent", pct_of_absent*100)
#this shows us the averages of other variables for not absent and absent people
data.groupby('Absenteeism time in hours').mean()
#visualize absent outcome by variable
#this allows us to see which variables may be good predictors of our outcome variable
#age vs. absenteeism
%matplotlib inline
pd.crosstab(data.Age,data.AbsenteeBinary).plot(kind='bar')
plt.title('Age vs. Absenteeism')
plt.xlabel('Age')
plt.ylabel('Absent')
plt.savefig('bargraph_age_absenteeism')
#month vs absenteeism
pd.crosstab(data.Son,data.AbsenteeBinary).plot(kind='bar')
plt.title('Son vs. Abseenteeism')
plt.xlabel('Son')
plt.ylabel('Absent')
plt.savefig('bargraph_month_absenteeism')
#this code create dummy variables (either 0 or 1) for our logistic regression analysis
category_vars=['Reason for absence','Month of absence','Day of the week','Seasons','Transportation expense','Distance from Residence to Work','Service time','Age','Work load Average/day ','Hit target','Disciplinary failure','Education','Social drinker','Social smoker','Pet','Weight','Height','Body mass index','Absenteeism time in hours','AbsenteeBinary']
for var in category_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1
category_vars=['Reason for absence','Month of absence','Day of the week','Seasons','Transportation expense','Distance from Residence to Work','Service time','Age','Work load Average/day ','Hit target','Disciplinary failure','Education','Social drinker','Social smoker','Pet','Weight','Height','Body mass index','Absenteeism time in hours','AbsenteeBinary']
for var in category_vars:
    data_vars=data.columns.values.tolist()
    to_keep=[i for i in data_vars if i not in category_vars]
data_final=data[to_keep]
data_final.columns
#Synthetic Minority Oversampling Technique
#this will balance our dataset
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
X = data['Reason for absence'].values.reshape(-1,1) #need to convert dataframe column to array and resize it so SMOTE will work on it
y = data['AbsenteeBinary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)
print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))
dataArray = data.values
dataArray.reshape(-1, 1)
sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
#use recursive feature elimination (finds best features in model) to get better model
data_final_vars=data_final.columns.values.tolist()
y=['AbsenteeBinary']
X=[i for i in data_final_vars if i not in y]
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(X_train, y_train.values.ravel())
print(rfe.support_)
print(rfe.ranking_)
#perform logistic regression
import statsmodels.api as sm
logit_model=sm.Logit(data['AbsenteeBinary'],data['Seasons'])
result=logit_model.fit()
print(result.summary2())
#fit our model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
#use our model to predict
prediction = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print('Confusion matrix',confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
