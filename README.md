# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```py
import pandas as pd 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler,RobustScaler,Normalizer
data=pd.read_csv("bmi.csv")
data
```
<img width="328" height="457" alt="image" src="https://github.com/user-attachments/assets/19e9a5aa-5185-40ac-99df-a2986255ec26" />

```py
data.isnull()
```
<img width="359" height="458" alt="image" src="https://github.com/user-attachments/assets/7dd066c2-bce1-43a0-96b5-80786f5f1d9b" />

```py
data2=data.copy()
scalar=StandardScaler()
data2[['New_height','New_weight']]=scalar.fit_transform(data2[['Height','Weight']])
data2
```
<img width="504" height="453" alt="image" src="https://github.com/user-attachments/assets/8c57a78b-7a00-436e-9824-717fa7ae3af1" />
```py
scaler=MinMaxScaler()
data3=data.copy()
data3[['New_height','New_weight']]=scaler.fit_transform(data3[['Height','Weight']])
data3
```

<img width="520" height="449" alt="image" src="https://github.com/user-attachments/assets/b7604e3b-dddb-49c5-baec-86c5f1250368" />

```py
scale=MaxAbsScaler()
data4=data.copy()
data4[['New_height','New_weight']]=scale.fit_transform(data4[['Height','Weight']])
data4
```
<img width="516" height="447" alt="image" src="https://github.com/user-attachments/assets/3a139203-f1e8-4083-9944-92fbd5d1a638" />

```py
data5=data.copy()
scaler=RobustScaler()
data5[['New_height','New_weight']]=scaler.fit_transform(data5[['Height','Weight']])
data5
```
<img width="519" height="449" alt="image" src="https://github.com/user-attachments/assets/f95dadce-f3a2-4bb4-b9ce-8a7ede6925e5" />

```py
data6=data.copy()
scaler=Normalizer()
data6[['New_height','New_weight']]=scaler.fit_transform(data6[['Height','Weight']])
data6
```

<img width="547" height="455" alt="image" src="https://github.com/user-attachments/assets/3f7abf30-20fb-4d72-a291-b97c8b5270a7" />

```py
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
df=pd.read_csv("income(1) (1).csv",na_values=[ " ?"])
df
```
<img width="1249" height="777" alt="image" src="https://github.com/user-attachments/assets/b32df4de-d1a0-487f-a170-36f1181062ce" />

```py
 df.isnull().sum()
```

<img width="230" height="312" alt="image" src="https://github.com/user-attachments/assets/171e447d-5564-4bac-ac8b-515793292813" />

```py
 missing=df[df.isnull().any(axis=1)]
 missing
```

<img width="1251" height="761" alt="image" src="https://github.com/user-attachments/assets/a30aca3c-98a5-4975-b6d2-79409034979f" />

```py
df2=df.dropna(axis=0)
df2
```
<img width="1256" height="783" alt="image" src="https://github.com/user-attachments/assets/519e02bc-6264-4237-81b7-b3c2dd736045" />

```py
sal=df["SalStat"]
df2["SalStat"]=df["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(df2['SalStat'])
```
<img width="419" height="266" alt="image" src="https://github.com/user-attachments/assets/00ebe2c3-4bba-48b2-9a41-95ae57906d38" />

```py
 sal2=df2['SalStat']
 dfs=pd.concat([sal,sal2],axis=1)
 dfs
```
<img width="363" height="455" alt="image" src="https://github.com/user-attachments/assets/8b7829c2-0cf2-4390-96df-a2a14bccff9a" />

```py
 new_data=pd.get_dummies(df2, drop_first=True)
 new_data
```
<img width="1254" height="491" alt="image" src="https://github.com/user-attachments/assets/26feed07-00d7-48f9-befa-de324541d51c" />

```py
columns_list=list(new_data.columns)
print(columns_list)
```
<img width="1243" height="420" alt="image" src="https://github.com/user-attachments/assets/c06107df-f1af-45fa-8d65-2e5a7dc2fcf0" />

```py
 features=list(set(columns_list)-set(['SalStat']))
 print(features)
```
<img width="1243" height="415" alt="image" src="https://github.com/user-attachments/assets/21026cc7-e677-4450-8483-2ca997cf8999" />

```py
y=new_data['SalStat'].values
y
```
<img width="435" height="35" alt="image" src="https://github.com/user-attachments/assets/2777ec15-8fe7-4f49-a9bc-346b2f1dec4e" />

```py
 x=new_data[features].values
 print(x)
```
<img width="220" height="169" alt="image" src="https://github.com/user-attachments/assets/661388c8-66d8-4e79-b9aa-9e946e865348" />

```py
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
<img width="218" height="67" alt="image" src="https://github.com/user-attachments/assets/64124646-dd02-4939-9747-69aedd713e94" />

```py
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
<img width="141" height="61" alt="image" src="https://github.com/user-attachments/assets/4b6abc0d-4879-4f2d-9c7c-4cdaee5d1c85" />

```py
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
<img width="202" height="28" alt="image" src="https://github.com/user-attachments/assets/0d8e7836-7c42-49f2-b47c-e141a688a5dc" />

```py
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
<img width="283" height="25" alt="image" src="https://github.com/user-attachments/assets/ea026db5-2f7d-4b98-a5f5-d674811005ad" />

```py
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
'Feature1': [1,2,3,4,5],
'Feature2': ['A','B','C','A','B'],
'Feature3': [0,1,1,0,1],
'Target'  : [0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
<img width="352" height="60" alt="image" src="https://github.com/user-attachments/assets/416f4d06-207b-4383-bb03-f9e97cf24626" />

```py
 import pandas as pd
 import numpy as np
 from scipy.stats import chi2_contingency
 import seaborn as sns
 tips=sns.load_dataset('tips')
 tips.head()
```
<img width="438" height="221" alt="image" src="https://github.com/user-attachments/assets/8b7245a5-c7cc-4575-9619-15dccfcb02cd" />

```py
tips.time.unique()
```
<img width="441" height="59" alt="image" src="https://github.com/user-attachments/assets/63d6272c-b1c5-466d-8400-1ff4d270dee2" />

```py
contingency_table=pd.crosstab(tips['sex'],tips['time'])
contingency_table
```
<img width="207" height="153" alt="image" src="https://github.com/user-attachments/assets/e32207a8-b652-41f1-a939-35661d0d286c" />

```py
 chi2,p,_,_=chi2_contingency(contingency_table)
 print(f"Chi-Square Statistics: {chi2}")
 print(f"P-Value: {p}")
```
<img width="400" height="53" alt="image" src="https://github.com/user-attachments/assets/b404ed10-548c-4b51-8ac4-5e593537ef1b" />

# RESULT:
From the experiment i explored the different types of scaling methods which will be usefull for scale data into a range of values which helps in improving accuracy.Then comes to performing Feature Selection which helps to select only the most important and relavent columns from a dataset to reduce Overfitting
