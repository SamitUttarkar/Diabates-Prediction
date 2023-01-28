#!/usr/bin/env python
# coding: utf-8

# In[342]:


# First we import all the necessary libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mno
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection  import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import mean_squared_error 
from termcolor import colored
get_ipython().run_line_magic('matplotlib', 'inline')


# In[343]:


# we analysise the data and see what values are missing 
df = pd.read_csv('PimaDiabetes.csv')


# In[344]:


df.describe()


# In[345]:


df.info()


# In[346]:


# here we can see that there are no null values in our data
df.isnull().any()


# In[347]:


# Let's take a look at minimum values
df.min()


# In[348]:


# ok so here we can see that many of these variables cannot have zero as it is not possible
# let's count the number of zeros in each column


# In[349]:


df[df == 0.0].count(0)
# here as we can see Insulin has almost half of the values zeros and Skin thickness has 30% values zeros which are a lot


# In[130]:


# # lets replace zeros with NaN so that we can better visualise it 
# df.loc[df["Glucose"] == 0.0, "Glucose"] = np.NAN
# df.loc[df["BloodPressure"] == 0.0, "BloodPressure"] = np.NAN
# df.loc[df["SkinThickness"] == 0.0, "SkinThickness"] = np.NAN
# df.loc[df["Insulin"] == 0.0, "Insulin"] = np.NAN
# df.loc[df["BMI"] == 0.0, "BMI"] = np.NAN


# In[131]:


#mno.matrix(df, figsize = (20, 6))
# as we can see skinthickness and insulin has many zeros in them 


# In[350]:


# first let's replace 'glucose' bloodpressure and bmi with medians
df.Glucose.replace(0,df.Glucose.median(),inplace=True)
df.BloodPressure.replace(0,df.BloodPressure.median(),inplace=True)
df.BMI.replace(0,df.BMI.median(),inplace=True)


# In[351]:


# as we can see there are no zeros left 
df['Glucose'].min()
df['BloodPressure'].min()
df['BMI'].min()


# In[352]:


df.drop(['Insulin','SkinThickness'],axis=1,inplace=True)


# In[353]:


df.head()


# In[354]:


df.isnull().any()


# ## EDA

# In[357]:


plt.figure(figsize=(15,8))
sns.heatmap(df.corr(),annot=True)


# In[358]:


sns.countplot(x=df.Pregnancies,
             palette="pastel").set_title('No. of Pregnacies')
plt.xlabel("Pregnancy count")
plt.ylabel("Total count")


# In[359]:


sns.histplot(x=df.Age).set_title("Histograph of Age")


# In[360]:


sns.countplot(x=df.Outcome).set_title("BarPlot on Outcome")


# In[362]:


fig, axes = plt.subplots(2, 2, sharey=False,figsize=(18, 10))
  
fig.suptitle('Distribution of Continuous Variable of data')
  
sns.histplot(ax=axes[0, 0], x=df.Glucose)
sns.histplot(ax=axes[0, 1], x=df.BloodPressure)
sns.histplot(ax=axes[1, 0], x=df.BMI)
sns.histplot(ax=axes[1, 1], x=df.DiabetesPedigree)


# In[363]:


fig, axes = plt.subplots(nrows = 3, ncols = 3)   
axes = axes.flatten()         
fig.set_size_inches(15, 15)

for ax, col in zip(axes, df.columns):
    sns.distplot(df[col], ax = ax)
    ax.set_title(col)


# In[365]:


fig, axes = plt.subplots(2, 2, sharey=False,figsize=(18, 10))
  
fig.suptitle('BoxPlot Continuous Variable of data')
  
sns.boxplot(ax=axes[0, 0], x=df.Glucose)
sns.boxplot(ax=axes[0, 1], x=df.BloodPressure)
sns.boxplot(ax=axes[1, 0], x=df.BMI)
sns.boxplot(ax=axes[1, 1], x=df.DiabetesPedigree)


# In[366]:


sns.pairplot(df,hue="Outcome",corner=True)


# In[368]:


plt.figure(figsize=(20, 6))
plt.subplot(1,2,1)
sns.histplot(x="Pregnancies",data = df, hue="Outcome")

plt.subplot(1,2,2)
sns.boxplot(x=df.Outcome, y=df.Pregnancies).set_title("Boxplot for Pregnancies by Outcome")


df[["Outcome","Pregnancies"]].groupby(by="Outcome").agg(("mean","median","min","max","skew","count"))


# In[369]:


plt.figure(figsize=(20, 6))
plt.subplot(1,2,1)
sns.histplot(x="Glucose",data = df, hue="Outcome")

plt.subplot(1,2,2)
sns.boxplot(x=df.Outcome, y=df.Glucose).set_title("Boxplot for Glucose by Outcome")


df[["Outcome","Glucose"]].groupby(by="Outcome").agg(("mean","median","min","max","skew","count"))


# In[370]:


plt.figure(figsize=(20, 6))
plt.subplot(1,2,1)
sns.histplot(x="BloodPressure",data = df, hue="Outcome")

plt.subplot(1,2,2)
sns.boxplot(x=df.Outcome, y=df.BloodPressure).set_title("Boxplot for BloodPressure by Outcome")


df[["Outcome","BloodPressure"]].groupby(by="Outcome").agg(("mean","median","min","max","skew","count"))


# In[371]:


plt.figure(figsize=(20, 6))
plt.subplot(1,2,1)
sns.histplot(x="BMI",data = df, hue="Outcome")

plt.subplot(1,2,2)
sns.boxplot(x=df.Outcome, y=df.BMI).set_title("Boxplot for BMI by Outcome")


df[["Outcome","BMI"]].groupby(by="Outcome").agg(("mean","median","min","max","skew","count"))


# In[379]:


plt.figure(figsize=(20, 6))
plt.subplot(1,2,1)
sns.histplot(x="DiabetesPedigree",data = df, hue="Outcome")
plt.subplot(1,2,2)
sns.boxplot(x=df.Outcome, y=df.DiabetesPedigree).set_title("Boxplot for Age by Outcome")


# In[378]:


plt.figure(figsize=(20, 6))
plt.subplot(1,2,1)
sns.histplot(x="Age",data = df, hue="Outcome")

plt.subplot(1,2,2)
sns.boxplot(x=df.Outcome, y=df.Age).set_title("Boxplot for Age by Outcome")


df[["Outcome","Age"]].groupby(by="Outcome").agg(("mean","median","min","max","skew","count"))


# In[280]:


df1 = df.copy()


# In[281]:


new_lis = []
for i in df['Pregnancies']:
    if i<3:
        new_lis.append(0)
    else:
        new_lis.append(1)
        
df['ThreeOrMore'] = new_lis


# In[ ]:





# In[282]:


df


# ## Fitting regression model and finding probability

# In[283]:


X = df['ThreeOrMore'].values
X = X.reshape(-1,1)
y = df['Outcome'].values
y = y.reshape(-1,1)


# In[284]:


from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=11)


# In[285]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)


# In[286]:


accuracy = classifier.score(X,y)
print(accuracy)


# In[287]:


from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,y_pred)
score


# In[288]:


y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test,y_pred)
print(conf_mat)


# In[289]:


#dont run
# Creating a new dataframe with ThreeOrMoreKids and Outcome column:
Out_df = df[['ThreeOrMore','Outcome']]

# Calculating pivot_table
pd.pivot_table(Out_df, values='Outcome', index=['ThreeOrMore'],
                    columns=['Outcome'], aggfunc=len)


# In[290]:


# here we can see that 82 women have diabates with less than 3 children and 178 have diabates with more than 3 children
# We need to use the formula of conditional probability                                                                                                                                                                                                                                              
# first lets calculate the probability of getting diabities and having 2 or less children
P_D_2 = 82/len(Out_df)
# now let's calculate probability of getting diabities and having 3 or more children
P_D_3 = 178/len(Out_df)
# probability of having 2 more less children
P_2 = (258+82)/len(Out_df)
# probabioity of having 3 or more children
P_3 = (232+178)/len(Out_df)

# now we will calculate the conditional probability 
# we will use the formula p(a|b) = p(a∩b)/p(b)
P_D_Given_2 = (P_D_2)/(P_2)
# probability of getting diabates given 3 or more children
P_D_Given_3 = (P_D_3)/(P_3)

print("The probability that getting diabetes, given that you have two or fewer children", P_D_Given_2 )
print("The probability that getting diabetes, given that you have three or more children",P_D_Given_3)


# In[291]:


#Alternatively we can also use predict_proba function to calculate the probability
pred = classifier.predict_proba(X)
print(pred)


# In[292]:


# here first column represents not getting diabates and second column represents getting diabates
print(pred)


# In[293]:


df.insert(loc = 8,
          column = 'Diabates_prob',
          value = pred[:,1])


# In[294]:


#df.drop('col1', inplace=True, axis=1)
df.head()


# ## Part 4 - Using new data for testing

# In[295]:


ToPredict = pd.read_csv('ToPredict.csv')


# In[296]:


ToPredict.head()


# In[297]:


# let's drop SkinThickenss and Insulin from this data as well
ToPredict.drop(['Insulin','SkinThickness'],axis=1,inplace=True)


# In[298]:


# first let's decide which model should we use to predict the values in the second data
# we shall experiment on 3 models
# Logistic Regression
# Decision Tree regressor
# Random Forest Regressor
ToPredict1 = ToPredict.copy()
ToPredict2 = ToPredict.copy()
ToPredict3 = ToPredict.copy()


# In[299]:


# Let's split the first data into training and testing excluding the out come column
X = df1.drop(['Outcome'],axis=1)
y = df1[['Outcome']]


# In[300]:


X_train, X_test, y_train, y_test = train_test_split(X,y , test_size = 0.2, random_state = 42)
print(f"The shape of X_train is      {colored(X_train.shape,'yellow')}")
print(f"The shape of X_test is       {colored(X_test.shape,'yellow')}")
print(f"The shape of y_train is      {colored(y_train.shape,'yellow')}")
print(f"The shape of y_test is       {colored(y_test.shape,'yellow')}")


# In[301]:


X.isnull().any()


# In[302]:


#LEt's use all the column as variables to predict out come 
models = [
    ('DecisionTreeRegressor',DecisionTreeRegressor()),
    ('LogisticRegression',LogisticRegression()),
    
]

print("The accuracy scores of the models are :")
for model_name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{colored(model_name,'blue')}")
    print(f"{colored(accuracy_score(y_test,y_pred), 'yellow')}")
    
    rmse = float(format(np.sqrt(mean_squared_error(y_test, y_pred)), '.3f'))
    print("\nRMSE: ", rmse)
    print('MSE test: %.3f\n' % mean_squared_error(y_test, y_pred))
    


# In[303]:


#lets use logictic regression and see which variables are useful for the outcome
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)


# In[304]:


y_pred = classifier.predict(X_test)
print(accuracy_score(y_test,y_pred))


# In[305]:


from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test,y_pred)


# In[306]:


conf_mat


# In[307]:


df_cm = pd.DataFrame(conf_mat, range(2), range(2))
# plt.figure(figsize=(10,7))
sns.set(font_scale=1.4) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

plt.show()


# In[308]:


print(classification_report(y_test, y_pred))


# In[309]:


importance = classifier.coef_[0]
#importance is a list so you can plot it. 
feat_importances = pd.Series(importance)
feat_importances.nlargest(20).plot(kind='barh',title = 'Feature Importance')


# In[310]:


df = pd.DataFrame(df.columns[:-1])
df['Values'] = feat_importances
df
# we can see that DiabatesPedigree, BMI and Pregnancies have the higest feature values so we will select those


# In[311]:


# Now let's choose one vartiable at a time
# let's choose these 3 variable to see their accuray and confustion matrix
X = df1[['Pregnancies','BMI','DiabetesPedigree']]
y = df1[['Outcome']]


# In[312]:


X_train, X_test, y_train, y_test = train_test_split(X,y , test_size = 0.2, random_state = 42)


# In[313]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
print(accuracy_score(y_test,y_pred))


# In[ ]:





# In[314]:


from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test,y_pred)
df_cm = pd.DataFrame(conf_mat, range(2), range(2))
# plt.figure(figsize=(10,7))
sns.set(font_scale=1.4) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

plt.show()


# In[315]:


# let's choose a single variable which is glucose and diabatesPedigree
X = df1[['Glucose','BloodPressure']]
y = df1[['Outcome']]


# In[316]:


X_train, X_test, y_train, y_test = train_test_split(X,y , test_size = 0.2, random_state = 42)


# In[317]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
print(accuracy_score(y_test,y_pred))


# In[318]:


from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test,y_pred)
df_cm = pd.DataFrame(conf_mat, range(2), range(2))
# plt.figure(figsize=(10,7))
sns.set(font_scale=1.4) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

plt.show()


# ## Now let's perdict the outcome for the Topredict dataset and also their probabilities

# In[319]:


ToPredict.head()


# In[320]:


# we will take this dataset as test set 
X_train = df1.drop(['Outcome'],axis=1)
X_test = ToPredict
y_train = df1['Outcome']


# In[321]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)


# In[322]:


y_pred = classifier.predict(X_test)


# In[323]:


y_pred


# In[324]:


prob = classifier.predict_proba(X_test)


# In[325]:


prob


# In[326]:


ToPredict1['Predicted Outcome'] = y_pred


# In[327]:


ToPredict1['Probability of getting Diabetes'] = prob[:,1]


# In[328]:


ToPredict1


# In[383]:


# now let's used glucose and blood pressure as variables 
X_test = df1[['Glucose','BloodPressure']]
X_test = ToPredict
y_train = df1['Outcome']


# In[384]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
print(y_pred)


# In[385]:


prob = classifier.predict_proba(X_test)


# In[386]:


ToPredict2['Predicted Outcome'] = y_pred


# In[387]:


ToPredict2['Probability of getting Diabetes'] = prob[:,1]


# In[388]:


ToPredict2


# In[389]:


# now let's used glucose and blood pressure as variables 
X_test = df1['Glucose']
X_test = ToPredict
y_train = df1['Outcome']


# In[390]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
print(y_pred)


# In[391]:


prob = classifier.predict_proba(X_test)


# In[392]:


ToPredict3['Predicted Outcome'] = y_pred


# In[393]:


ToPredict3['Probability of getting Diabetes'] = prob[:,1]


# In[394]:


ToPredict3


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




