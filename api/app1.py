# from flask import Flask,request,jsonify
# import pandas as pd
# import numpy as np
# # from random import shuffle
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
# from sklearn.svm import SVC
# from sklearn.metrics import f1_score, accuracy_score, confusion_matrix,classification_report,precision_score,roc_curve
# import seaborn as sns
# from sklearn.utils import shuffle
# # from pandas_profiling import ProfileReport
# from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier 
# from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier 
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn import metrics
# import statistics
# from sklearn.cluster import KMeans


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('C:/Users/LENOVO/Desktop/project/Training.csv')


# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values
# df = df.dropna()

# Scale the symptom columns
X = df.iloc[:, :132].values
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# Run KMeans with 3 clusters
kmeans = KMeans(n_clusters=34, random_state=42)
kmeans.fit(X)

# Get the predicted clusters
df['Cluster'] = kmeans.labels_

# Display the first 10 rows of the dataframe
print(df.head(10))

# Display the number of prognoses in each cluster
print(df['Cluster'].value_counts())

# Display the mean severity of symptoms in each cluster
# print(df.groupby('Cluster').mean())

# Plot the distribution of prognoses in each cluster
# plt.figure(figsize=(10, 6))
# sns.countplot(x='Cluster', hue='prognosis', data=df)
# plt.title('Distribution of Prognoses in Each Cluster')
# plt.show()

n_clusters = len(kmeans.cluster_centers_)
labels = kmeans.labels_

# Add the cluster labels to the dataframe
df['Cluster'] = labels

# Display the number of diseases in each cluster
print("Number of diseases in each cluster:")
print(df['Cluster'].value_counts())
print()

# Display the unique diseases in each cluster
print("\nUnique diseases in each cluster:")
l1=[]
for i in range(n_clusters):
    cluster_df = df[df['Cluster'] == i]
    unique_diseases = cluster_df['prognosis'].unique()
    print(f"Cluster {i}:")
    print(unique_diseases[0])
    l=[]
    print(unique_diseases.tolist())
    l1.append(unique_diseases.tolist())
    
print(l1)
