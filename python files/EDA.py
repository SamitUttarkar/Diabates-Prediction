'''EDA'''

plt.figure(figsize=(15,8))
sns.heatmap(df.corr(),annot=True)



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

