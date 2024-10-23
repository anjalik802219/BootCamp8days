#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df1=pd.read_csv("C:\\Users\\Anjali Kumari\\Downloads\\archive (2)\\car_data.csv")


# In[3]:


df1.head()


# In[4]:


df1.info()


# In[5]:


df1.columns


# In[6]:


df1.dtypes


# In[7]:


df1.tail


# In[8]:


df1.describe()


# In[9]:


df1.isnull().sum()


# In[10]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
df1[['cylinders', 'displacement']] = imputer.fit_transform(df1[['cylinders', 'displacement']])


# In[11]:


df1.isnull().sum()


# In[12]:


df1_encoded = pd.get_dummies(df1, drop_first=True)


# In[13]:


correlation_matrix_encoded = df1_encoded.corr()
correlation_matrix_encoded


# In[14]:


sns.histplot(df1['city_mpg'], kde=True,color='skyblue')
plt.title('Distribution of City MPG')
plt.show()


# In[15]:


sns.histplot(df1['cylinders'], kde=True, color='lightgreen')
plt.title('Distribution of Cylinders')
plt.show()


# In[16]:


sns.histplot(df1['displacement'], kde=True, color='salmon')
plt.title('Distribution of Displacement')
plt.show()


# In[17]:


X = df1_encoded.drop('city_mpg', axis=1)
y = df1_encoded['city_mpg']


# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[19]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[20]:


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)


# In[21]:


from sklearn.metrics import accuracy_score 
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print(f"Random Forest Accuracy: {rf_accuracy*100:.2f}")


# In[22]:


X = df1_encoded.drop('city_mpg', axis=1)
y = df1_encoded['city_mpg']


# In[24]:


from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)


# In[25]:


y_pred_gb = gb_model.predict(X_test)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print(f'Accuracy: {accuracy_gb*100:.2f}')


# In[30]:


pip install catboost


# In[31]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
cat_model = CatBoostClassifier(random_seed=42, verbose=0)
cat_model.fit(X_train, y_train)

cat_predictions = cat_model.predict(X_test)


# In[32]:


y_pred_cat = cat_model.predict(X_test)
accuracy_cat = accuracy_score(y_test, y_pred_cat)
print(f'Accuracy: {accuracy_cat*100:.2f}')


# In[33]:


df1.describe(include = "object")


# In[36]:


plt.figure(figsize = (20, 30))
for i, col in enumerate(df1.columns, 1):
    plt.subplot(4, 3, i)
    sns.histplot(x = df1[col])
    plt.title(f"Histogram of {col} Data")
    plt.xticks(rotation = 30)
    plt.plot()


# In[38]:


ax1 = df1.pivot_table(index = "fuel_type", columns = "class", values = "city_mpg", aggfunc = "mean").plot.bar()
ax1.legend(loc = "best", bbox_to_anchor = (1, 1))
plt.show()


# In[40]:


ax2 = df1.pivot_table(index = "fuel_type", columns = "drive", values = "city_mpg", aggfunc = "mean").plot.bar()
ax2.legend(loc = "best", bbox_to_anchor = (1, 1))
plt.show()


# In[41]:


ax3 = df1.pivot_table(index = "fuel_type", columns = "make", values = "city_mpg", aggfunc = "mean").plot.bar()
ax3.legend(loc = "best", bbox_to_anchor = (1, 1))
plt.show()


# In[42]:


df1_corr = df1.corr()
plt.figure(figsize = (12, 12))
sns.heatmap(df1_corr, fmt = ".3f", annot = True, cmap = "Blues")
plt.show()


# In[43]:


sns.scatterplot(x = df1["city_mpg"], y = df1["combination_mpg"])
plt.show()


# In[46]:


X = df1.iloc[:, :-1]
y = df1.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[48]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = X_train.select_dtypes(include=['object']).columns

X_train_encoded = pd.get_dummies(X_train, columns=categorical_columns, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_columns, drop_first=True)

X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

sc = StandardScaler()

X_train_encoded[numerical_columns] = sc.fit_transform(X_train_encoded[numerical_columns])
X_test_encoded[numerical_columns] = sc.transform(X_test_encoded[numerical_columns])

print(X_train_encoded.head())


# In[50]:


pip install xgboost


# In[55]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def cat_analyser(data, col: str, freq_limit: int = 36):
    df1_ = data.copy()
    sns.set(rc={'axes.facecolor': 'gainsboro', 'figure.facecolor': 'gainsboro'})
    
    # Limit the number of unique categories if it exceeds the freq_limit
    if df1_[col].nunique() > freq_limit:
        df1_ = df1_.loc[df1_[col].isin(df1_[col].value_counts().keys()[:freq_limit].tolist())]
    
    # Create subplots for countplot and pie chart
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    fig.suptitle(col, fontsize=16)
    
    # Countplot
    sns.countplot(data=df1_, x=col, ax=ax[0], palette='coolwarm', 
                  order=df1_[col].value_counts().index)
    ax[0].set_xlabel('')

    # Pie chart
    pie_cmap = plt.get_cmap('coolwarm')
    normalize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    df1_[col].value_counts().plot.pie(autopct='%1.1f%%', textprops={'fontsize': 10},
                                      ax=ax[1], colors=pie_cmap(normalize(df1_[col].value_counts())))
    ax[1].set_ylabel('')
    
    plt.show()
    matplotlib.rc_file_defaults()  
    sns.reset_orig()  
cat_cols = df1.select_dtypes(include=['object', 'category']).columns.tolist()

# Analyze each categorical column
for col in cat_cols:
    cat_analyser(df1, col)


# In[ ]:




