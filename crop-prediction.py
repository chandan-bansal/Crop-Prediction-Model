#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as skl
from ipywidgets import interact
import matplotlib.pyplot as plt
import time


# In[34]:



from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import cross_val_score


# In[3]:


data =pd.read_csv('crop.csv')
data.head()


# In[4]:


#Shape of Dataset
print('Shape of Dataset : ',data.shape)


# In[5]:


data.describe()


# In[6]:


data.nunique()


# In[7]:


#Check Miising Values
data.isnull().sum()


# In[8]:


#Check Crops in Dataset
data['label'].value_counts()


# In[9]:


fig_dims = (18, 12)
fig, ax = plt.subplots(figsize=fig_dims)

plt.subplot(2,4,1)
sns.boxplot(data['N'])
plt.xlabel('Nitrogen')

plt.subplot(2,4,2)
sns.boxplot(data['P'])
plt.xlabel('Phosphorus')

plt.subplot(2,4,3)
sns.boxplot(data['K'])
plt.xlabel('Potassium')

plt.subplot(2,4,4)
sns.boxplot(data['temperature'])
plt.xlabel('Temperature')

plt.subplot(2,4,5)
sns.boxplot(data['humidity'])
plt.xlabel('Humidity')

plt.subplot(2,4,6)
sns.boxplot(data['ph'])
plt.xlabel('PH')

plt.subplot(2,4,7)
sns.boxplot(data['rainfall'])
plt.xlabel('Rainfall')


# In[10]:


first = data['humidity'].quantile(.25)
third = data['humidity'].quantile(.75)
print(first)
IQR = third-first
print(IQR)
print(third)

range = third + 1.5*IQR
print(range)

range1 = third + 3*IQR
print(range1)


# In[11]:


temp_data = data[data['humidity'] > range1]


# In[12]:


temp_data.shape


# In[13]:


#Check Summary for all Crops
print('Average Ratio of Nitrogen in the soil : {0:.2f}'.format(data['N'].mean()))
print('Average Ratio of Phosphorous in the soil : {0:.2f}'.format(data['P'].mean()))
print('Average Ratio of Potassium in the soil : {0:.2f}'.format(data['K'].mean()))
print('Average Temperature in Celsius : {0:.2f}'.format(data['temperature'].mean()))
print('Average Ratio of Humidity in the soil : {0:.2f}'.format(data['humidity'].mean()))
print('Average PH value of the soil : {0:.2f}'.format(data['ph'].mean()))
print('Average Rainfall in mm : {0:.2f}'.format(data['rainfall'].mean()))


# In[14]:


#lets Check the Summary Statistics for each of the Crops

@interact
def summary(crops = list(data['label'].value_counts().index)):
    x = data[data['label'] == crops]
    print('--------------------------------------------------')
    print('Statistics for Nitrogen')
    print('Minimum Nitrogen required :',x['N'].min())
    print('Average Nitrogen required :',x['N'].mean())
    print('Maximum Nitrogen required :',x['N'].max())
    print('--------------------------------------------------')
    print('Statistics for Phosphorus')
    print('Minimum Phosphorus required :',x['P'].min())
    print('Average Phosphorus required :',x['P'].mean())
    print('Maximum Phosphorus required :',x['P'].max())
    print('--------------------------------------------------')
    print('Statistics for Potassium')
    print('Minimum Potassium required :',x['K'].min())
    print('Average Potassium required :',x['K'].mean())
    print('Maximum Potassium required :',x['K'].max())
    print('--------------------------------------------------')
    print('Statistics for Temperature')
    print('Minimum Temperature required :',x['temperature'].min())
    print('Average Temperature required :',x['temperature'].mean())
    print('Maximum Temperature required :',x['temperature'].max())
    print('--------------------------------------------------')
    print('Statistics for Humidity')
    print('Minimum Humidity required :',x['humidity'].min())
    print('Average Humidity required :',x['humidity'].mean())
    print('Maximum Humidity required :',x['humidity'].max())
    print('--------------------------------------------------')
    print('Statistics for PH')
    print('Minimum PH required :',x['ph'].min())
    print('Average PH required :',x['ph'].mean())
    print('Maximum PH required :',x['ph'].max())
    print('--------------------------------------------------')
    
    print('Statistics for Rainfall')
    print('Minimum Rainfall required :',x['rainfall'].min())
    print('Average Rainfall required :',x['rainfall'].mean())
    print('Maximum Rainfall required :',x['rainfall'].max())


# In[15]:


#Distribution of Agriculture Conditions

fig_dims = (18, 12)
fig, ax = plt.subplots(figsize=fig_dims)

plt.subplot(2,4,1)
sns.distplot(data['N'])
plt.xlabel('Nitrogen')

plt.subplot(2,4,2)
sns.distplot(data['P'])
plt.xlabel('Phosphorus')

plt.subplot(2,4,3)
sns.distplot(data['K'])
plt.xlabel('Potassium')

plt.subplot(2,4,4)
sns.distplot(data['temperature'])
plt.xlabel('Temperature')

plt.subplot(2,4,5)
sns.distplot(data['humidity'])
plt.xlabel('Humidity')

plt.subplot(2,4,6)
sns.distplot(data['ph'])
plt.xlabel('PH')

plt.subplot(2,4,7)
sns.distplot(data['rainfall'])
plt.xlabel('Rainfall')


# In[16]:


sns.relplot(x='temperature', y='humidity' , hue='rainfall' ,data=data)


# In[17]:


#Season wise Crops

print('\n')
print('Summer Crops')
print(data[(data.temperature > 30) & (data.humidity > 50)]['label'].unique())
print('\n')
print('Winter Crops')
print(data[(data.temperature < 20) & (data.humidity > 30)]['label'].unique())
print('\n')
print('Rainy Crops')
print(data[(data.rainfall > 200) & (data.humidity > 30)]['label'].unique())


# In[19]:


#to make Groups we use KMeans Clustering Algorithm

from sklearn.cluster import KMeans

#removing the labels column
x = data.drop(['label'],axis=1)

#Selecting all the values of the data
x = x.values

#Checking the Shape
print(x.shape)


# In[20]:


#Lets implement the K Means algorithm to perform Clustering analysis


kn = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10)
y_means = kn.fit_predict(x)

#Results

a = data['label']
y_means = pd.DataFrame(y_means)
z = pd.concat([y_means, a], axis = 1)
z = z.rename(columns = {0:'cluster'})

#Lets Check the Cluster of each Crop

print("Lets Check the Results After applying the K Means Clustering Analysis \n")
print("Crops in First Cluster:",z[z['cluster'] == 0]['label'].unique())
print('\n')
print("Crops in Second Cluster:",z[z['cluster'] == 1]['label'].unique())
print('\n')
print("Crops in Third Cluster:",z[z['cluster'] == 2]['label'].unique())
print('\n')
print("Crops in Fourth Cluster:",z[z['cluster'] == 3]['label'].unique())


# In[21]:


#Corelation
data.corr()


# In[22]:


#Visualize Correlation

plt.figure(figsize=(14,14))
sns.heatmap(data.corr(), annot =True, fmt= '.0%')


# In[23]:


c=data.label.astype('category')
targets = dict(enumerate(c.cat.categories))
data['target']=c.cat.codes

y=data.target
X=data[['N','P','K','temperature','humidity','ph','rainfall']]


# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# we must apply the scaling to the test set as well that we are computing for the training set
X_test_scaled = scaler.transform(X_test)


# In[25]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)
knn.score(X_test_scaled, y_test)


# In[26]:


from sklearn.metrics import confusion_matrix
mat=confusion_matrix(y_test,knn.predict(X_test_scaled))
df_cm = pd.DataFrame(mat, list(targets.values()), list(targets.values()))
sns.set(font_scale=1.0) # for label size
plt.figure(figsize = (12,8))
sns.heatmap(df_cm, annot=True, annot_kws={"size": 12},cmap="terrain")


# In[30]:


features = data[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = data['label']


# In[31]:


acc = []
model = []


# In[32]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features,target,test_size = 0.2,random_state =2)


# # Decision Tree

# In[35]:


from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)

DT.fit(x_train,y_train)

predicted_values = DT.predict(x_test)
x = metrics.accuracy_score(y_test, predicted_values)
acc.append(x)
model.append('Decision Tree')
print("Decision Tree's Accuracy is: ", x*100)

print(classification_report(y_test,predicted_values))


# In[36]:


score = cross_val_score(DT, features, target,cv=5)
print('Cross validation score: ',score)


# In[37]:


#Print Train Accuracy
dt_train_accuracy = DT.score(x_train,y_train)
print("Training accuracy = ",DT.score(x_train,y_train))
#Print Test Accuracy
dt_test_accuracy = DT.score(x_test,y_test)
print("Testing accuracy = ",DT.score(x_test,y_test))


# In[38]:


y_pred = DT.predict(x_test)
y_true = y_test

from sklearn.metrics import confusion_matrix

cm_dt = confusion_matrix(y_true,y_pred)

f, ax = plt.subplots(figsize=(15,10))
sns.heatmap(cm_dt, annot=True, linewidth=0.5, fmt=".0f",  cmap='viridis', ax = ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title('Predicted vs actual')
plt.show()


# # Random Forest

# In[39]:


from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(x_train,y_train)

predicted_values = RF.predict(x_test)

x = metrics.accuracy_score(y_test, predicted_values)
acc.append(x)
model.append('RF')
print("Random Forest Accuracy is: ", x)

print(classification_report(y_test,predicted_values))


# In[40]:


score = cross_val_score(RF,features,target,cv=5)
print('Cross validation score: ',score)


# In[41]:


#Print Train Accuracy
rf_train_accuracy = RF.score(x_train,y_train)
print("Training accuracy = ",RF.score(x_train,y_train))
#Print Test Accuracy
rf_test_accuracy = RF.score(x_test,y_test)
print("Testing accuracy = ",RF.score(x_test,y_test))


# In[42]:


y_pred = RF.predict(x_test)
y_true = y_test

from sklearn.metrics import confusion_matrix

cm_rf = confusion_matrix(y_true,y_pred)

f, ax = plt.subplots(figsize=(15,10))
sns.heatmap(cm_rf, annot=True, linewidth=0.5, fmt=".0f",  cmap='viridis', ax = ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title('Predicted vs actual')
plt.show()


# # Naive Bayes Classifier

# In[43]:


from sklearn.naive_bayes import GaussianNB
NaiveBayes = GaussianNB()

NaiveBayes.fit(x_train,y_train)

predicted_values = NaiveBayes.predict(x_test)
x = metrics.accuracy_score(y_test, predicted_values)
acc.append(x)
model.append('Naive Bayes')
print("Naive Bayes Accuracy is: ", x)

print(classification_report(y_test,predicted_values))


# In[44]:


score = cross_val_score(NaiveBayes,features,target,cv=5)
print('Cross validation score: ',score)


# In[45]:


#Print Train Accuracy
nb_train_accuracy = NaiveBayes.score(x_train,y_train)
print("Training accuracy = ",NaiveBayes.score(x_train,y_train))
#Print Test Accuracy
nb_test_accuracy = NaiveBayes.score(x_test,y_test)
print("Testing accuracy = ",NaiveBayes.score(x_test,y_test))


# In[46]:


y_pred = NaiveBayes.predict(x_test)
y_true = y_test

from sklearn.metrics import confusion_matrix

cm_nb = confusion_matrix(y_true,y_pred)

f, ax = plt.subplots(figsize=(15,10))
sns.heatmap(cm_nb, annot=True, linewidth=0.5, fmt=".0f",  cmap='viridis', ax = ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title('Predicted vs actual')
plt.show()


# In[47]:


plt.figure(figsize=[14,7],dpi = 100, facecolor='white')
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('ML Algorithms')
sns.barplot(x = acc,y = model,palette='viridis')
plt.savefig('plot.png', dpi=300, bbox_inches='tight')


# In[52]:


label = ['Decision Tree','Random Forest','Naive Bayes']
Test = [dt_test_accuracy,rf_test_accuracy,
        nb_test_accuracy]
Train = [dt_train_accuracy, rf_train_accuracy,
         nb_train_accuracy]

f, ax = plt.subplots(figsize=(20,7)) # set the size that you'd like (width, height)
X_axis = np.arange(len(label))
plt.bar(X_axis - 0.2,Test, 0.4, label = 'Test', color=('midnightblue'))
plt.bar(X_axis + 0.2,Train, 0.4, label = 'Train', color=('mediumaquamarine'))

plt.xticks(X_axis, label)
plt.xlabel("ML algorithms")
plt.ylabel("Accuracy")
plt.title("Testing vs Training Accuracy")
plt.legend()
#plt.savefig('train vs test.png')
plt.show()


# In[ ]:




