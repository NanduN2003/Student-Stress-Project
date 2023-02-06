#!/usr/bin/env python
# coding: utf-8

# # ATTRIBUTE DESCRIPTION:
# 
# This is an virtually desined data related to students pursuing Undergraduation Degree in Bachelor's of Technology(B.Tech) domain.
# 
# 1.St.Name : Name of the student.
# 
# 2.Age : An numerical which specifies the age within limit of (18,23).
# 
# 3.Gender : An Nominal Variable which indicates sex of the student Male/Female.
# 
# 4.Year_of_study : An numerical which encloses the year in which the student is studying like {1,2,3,4}.
# 
# 5.Course : Branch/Department in which student is pursuing his study like amongst list of branches colleges offers:
# {   Computer Science and Engineering(CSE) ,
#     Electronics and Communication Engineering(ECE),
#     Mechanical Engineering(ME) ,
#     Electrical and Electronics Engineering(EEE) , 
#     Aerospace Engineering(AS) , 
#     Information Technology(IT) , 
#     Aeronautical Engineering(AE) , 
#     Data Science(DS) , 
#     Artificial Intelligence(AI) , 
#     Machine Learning(ML) , 
#     Internet of Things (IOT) , 
#     Computer Technology and Information Security(CTIS)
# }
# 
# 6.No.of.subjects : An numerical which specifies no.of subjects a student have in running semester; in a range of (4,8).
# 
# 7.Active.backlogs : An numerical indicates no.of active backlogs a student hold; in a range of (0,4)
# 
# 8.Cum.Percentage : An Ordinal variable where the it shows up the average percentage of student till running semester; in limits (70,95)
# 
# 9.Residential_type : An nominal variable where it displays where the student is living.
# Like we have options of { With Parents , University Accomodation , In Rentals }
# 
# 10.Sleeping_hrs : An numerical which indiactes no.of hours a student rests for a day; like in a range of (4,10)
# 

# # Creating a Student Stress Dataset 

# In[3]:


import pandas as pd
import numpy as np
import random
import names 

#One of the package amongst Python Packages which generates random names
no_students=1000

stress=pd.DataFrame(columns=['St.Name','Age','Gender','Year_of_Study','Course','No.of.subjects','Active.Backlogs','Cum.Percentage','ResidentialType','Sleeping_hrs','Target'])


# In[4]:


for i in range(no_students):
    
    name=names.get_full_name()
    
    age=random.randint(18,23)
    
    gender=random.choice(["Male","Female"])
    
    year_of_study=random.choice([1,2,3,4])
    
    course=random.choice(['CSE','ECE','ME','EEE','AS','IT','AE','DS',"AI",'ML','IOT','CTIS'])
    
    no_of_subjects=random.randint(4,8)
    
    active_backlogs=random.randint(0,4)
    
    cum_percentage=random.randint(70,95)
    
    residentialtype=random.choice(['With Parents','University Accomodation','In Rentals'])
    
    sleeping_hrs=random.randint(4,10)
    
    target=random.choice([0,1])
    
    
    #Appending randomly generated records into dataset 
    
    stress=stress.append({'St.Name':name,'Age':age,'Gender':gender,'Year_of_Study':year_of_study,'Course':course,'No.of.subjects':no_of_subjects,'Active.Backlogs':active_backlogs,'Cum.Percentage':cum_percentage,'ResidentialType':residentialtype,'Sleeping_hrs':sleeping_hrs,'Target':target},ignore_index=True)


# In[ ]:





# In[5]:


stress.to_csv('student_stress.csv',index=False)

#converting the dataset into csv file format.


# In[6]:


#Loading the Data

stress_df=pd.read_csv('student_stress.csv')


# In[7]:


stress_df.head(50)


# In[8]:


stress_df.columns


# In[9]:


stress_df.shape


# In[10]:


stress_df.describe()


# In[11]:


stress_df.corr()


# In[12]:


#Duplicating the dataframe and using it for Train-Test split 

df=stress_df.copy()


# In[13]:


df["Gender"].replace(["Male", "Female"],[0,1], inplace=True)

# Otherwise;we even can use getdummies() method to convert categorical into discrete
## df1= df.get_dummies(df, columns = ['Gender'], drop_first=True)

df["Course"].replace(['CSE','ECE','ME','EEE','AS','IT','AE','DS',"AI",'ML','IOT','CTIS'],[1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012],inplace=True)
df['ResidentialType'].replace(['With Parents','University Accomodation','In Rentals'],[1,2,3],inplace=True)

df.head()


# In[14]:


from sklearn.model_selection import train_test_split
from sklearn import preprocessing

X=df.drop(['Target','St.Name'],axis=1)
y=df['Target']


# # Evaluting using SVM with Linear Kernel
# 
# 

# In[107]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.28,random_state=42)


# In[108]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cls=svm.SVC(kernel='linear')

cls.fit(X_train,y_train)


# In[109]:


y_pred_svc = cls.predict(X_test)
y_pred_svc


# In[110]:


accuracy = cls.score(X_test,y_test)


# In[111]:


confmatrix_linear = confusion_matrix(y_test, y_pred_svc)
print(confmatrix_linear)

#Out of 280,123 were predicted wrongly i.e; aprox 44% predictions


# In[112]:


sns.heatmap(confmatrix_linear, annot=True, fmt='d', cmap='cool')
plt.show()


# In[113]:


print(classification_report(y_test, y_pred_svc))


# In[114]:


print(accuracy)


# # Evaluating using SVM with rbf Kernel

# In[115]:


cls1=svm.SVC(kernel='rbf')

cls1.fit(X_train,y_train)


# In[116]:


y_pred_svc1 = cls1.predict(X_test)
y_pred_svc1


# In[117]:


confmatrix_rbf = confusion_matrix(y_test, y_pred_svc1)
print(confmatrix_rbf)


# In[118]:


sns.heatmap(confmatrix_rbf, annot=True, fmt='d' )
plt.show()


# In[119]:


print(classification_report(y_test, y_pred_svc))


# In[120]:


accuracy = cls1.score(X_test,y_test)


# In[121]:


print(accuracy)


# In[ ]:




