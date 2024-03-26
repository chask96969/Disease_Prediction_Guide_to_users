from flask import Flask,request,jsonify
import pandas as pd
import numpy as np
# from random import shuffle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix,classification_report,precision_score,roc_curve
import seaborn as sns
from sklearn.utils import shuffle
# from pandas_profiling import ProfileReport
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier 
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import statistics
from sklearn.cluster import KMeans

app=Flask(__name__)

symptom_index = {
    "itching": 0,
    "skin rash": 1,
    "nodal skin eruptions": 2,
    "continuous sneezing": 3,
    "shivering": 4,
    "chills": 5,
    "joint pain": 6,
    "stomach pain": 7,
    "acidity": 8,
    "ulcers on tongue": 9,
    "muscle wasting": 10,
    "vomiting": 11,
    "burning micturition": 12,
    "spotting urination": 13,
    "fatigue": 14,
    "weight gain": 15,
    "anxiety": 16,
    "cold hands and feets": 17,
    "mood swings": 18,
    "weight loss": 19,
    "restlessness": 20,
    "lethargy": 21,
    "patches in throat": 22,
    "irregular sugar level": 23,
    "cough": 24,
    "high fever": 25,
    "sunken eyes": 26,
    "breathlessness": 27,
    "sweating": 28,
    "dehydration": 29,
    "indigestion": 30,
    "headache": 31,
    "yellowish skin": 32,
    "dark urine": 33,
    "nausea": 34,
    "loss of appetite": 35,
    "pain behind the eyes": 36,
    "back pain": 37,
    "constipation": 38,
    "abdominal pain": 39,
    "diarrhoea": 40,
    "mild fever": 41,
    "yellow urine": 42,
    "yellowing of eyes": 43,
    "acute liver failure": 44,
    "fluid overload": 45,
    "swelling of stomach": 46,
    "swelled lymph nodes": 47,
    "malaise": 48,
    "blurred and distorted vision": 49,
    "phlegm": 50,
    "throat irritation": 51,
    "redness of eyes": 52,
    "sinus pressure": 53,
    "runny nose": 54,
    "congestion": 55,
    "chest pain": 56,
    "weakness in limbs": 57,
    "fast heart rate": 58,
    "pain during bowel movements": 59,
    "pain in anal region": 60,
    "bloody stool": 61,
    "irritation in anus": 62,
    "neck pain": 63,
    "dizziness": 64,
    "cramps": 65,
    "bruising": 66,
    "obesity": 67,
    "swollen legs": 68,
    "swollen blood vessels": 69,
    "puffy face and eyes": 70,
    "enlarged thyroid": 71,
    "brittle nails": 72,
    "swollen extremeties": 73,
    "excessive hunger": 74,
    "extra marital contacts": 75,
    "drying and tingling lips": 76,
    "slurred speech": 77,
    "knee pain": 78,
    "hip joint pain": 79,
    "muscle weakness": 80,
    "stiff neck": 81,
    "swelling joints": 82,
    "movement stiffness": 83,
    "spinning movements": 84,
    "loss of balance": 85,
    "unsteadiness": 86,
    "weakness of one body side": 87,
    "loss of smell": 88,
    "bladder discomfort": 89,
    "foul smell ofurine": 90,
    "continuous feel of urine": 91,
    "passage of gases": 92,
    "internal itching": 93,
    "toxic look (typhos)": 94,
    "depression": 95,
    "irritability": 96,
    "muscle pain": 97,
    "altered sensorium": 98,
    "red spots over body": 99,
    "belly pain": 100,
    "abnormal menstruation": 101,
    "dischromic patches": 102,
    "watering from eyes": 103,
    "increased appetite": 104,
    "polyuria": 105,
    "family history": 106,
    "mucoid sputum": 107,
    "rusty sputum": 108,
    "lack of concentration": 109,
    "visual disturbances": 110,
    "receiving blood transfusion": 111,
    "receiving unsterile injections": 112,
    "coma": 113,
    "stomach bleeding": 114,
    "distention of abdomen": 115,
    "history of alcohol consumption": 116,
    "blood in sputum": 117,
    "prominent veins on calf": 118,
    "palpitations": 119,
    "painful walking": 120,
    "pus filled pimples": 121,
    "blackheads": 122,
    "scurring": 123,
    "skin peeling": 124,
    "silver like dusting": 125,
    "small dents in nails": 126,
    "inflammatory nails": 127,
    "blister": 128,
    "red sore around nose": 129,
    "yellow crust ooze": 130,
    "prognosis": 131
}

@app.route('/api/route',methods=['POST'])
def predict():
    # df = pd.read_csv('C:/Users/LENOVO/Desktop/project/dataset.csv')
    # # df = shuffle(df,random_state=42)
    # df.head()
    # # return {'disease':'itching'}
    symps=request.json['symp']
    print('symptom type',symps)
    print(symptom_index )

    df = pd.read_csv('dataset.csv')
    df = shuffle(df,random_state=42)
    df.head()

    for col in df.columns:
        df[col] = df[col].str.replace('_',' ')
    df.head()

    null_checker = df.apply(lambda x: sum(x.isnull())).to_frame(name='count')
    print(null_checker)

    plt.figure(figsize=(10,5))
    plt.plot(null_checker.index, null_checker['count'])
    plt.xticks(null_checker.index, null_checker.index, rotation=45,
    horizontalalignment='right')
    plt.title('Before removing Null values')
    plt.xlabel('column names')
    plt.margins(0.1)
    plt.show()

    cols = df.columns
    data = df[cols].values.flatten()

    s = pd.Series(data)
    s = s.str.strip()
    s = s.values.reshape(df.shape)

    df = pd.DataFrame(s, columns=df.columns)
    df.head()

    df = df.fillna(0)
    df.head()

    df1 = pd.read_csv('Symptom-severity.csv')
    df1['Symptom'] = df1['Symptom'].str.replace('_',' ')
    # df1.head()
    print(df1)

    vals = df.values
    symptoms = df1['Symptom'].unique()

    for i in range(len(symptoms)):
        vals[vals == symptoms[i]] = df1[df1['Symptom'] == symptoms[i]]['weight'].values[0]
        
    d = pd.DataFrame(vals, columns=cols)
    d.head()

    d = d.replace('dischromic  patches', 0)
    d = d.replace('spotting  urination',0)
    df = d.replace('foul smell of urine',0)
    df.head(10)

    null_checker = df.apply(lambda x: sum(x.isnull())).to_frame(name='count')
    print(null_checker)

    plt.figure(figsize=(10,5))
    plt.plot(null_checker.index, null_checker['count'])
    plt.xticks(null_checker.index, null_checker.index, rotation=45,
    horizontalalignment='right')
    plt.title('After removing Null values')
    plt.xlabel('column names')
    plt.margins(0.01)
    plt.show()

    print("Number of symptoms used to identify the disease ",len(df1['Symptom'].unique()))
    print("Number of diseases that can be identified ",len(df['Disease'].unique()))

    data = df.iloc[:,1:].values
    labels = df['Disease'].values

    x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size = 0.8,random_state=42)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    tree =DecisionTreeClassifier(criterion='gini',random_state=42,max_depth=13)
    tree.fit(x_train, y_train)
    preds=tree.predict(x_test)
    conf_mat = confusion_matrix(y_test, preds)
    print(conf_mat)
    df_cm = pd.DataFrame(conf_mat, index=df['Disease'].unique(), columns=df['Disease'].unique())
    print('F1-score% =', f1_score(y_test, preds, average='macro')*100, '|', 'Accuracy% =', accuracy_score(y_test, preds)*100)
    sns.heatmap(df_cm)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_mat, display_labels = [False, True])
# plt.figure(figsize=(12, 8))
# plot_tree(tree, filled=True, feature_names=x_train.columns, class_names=df['Disease'].unique(), rounded=True)
# plt.show()
# cm_display.plot()
# plt.show()
# print(len(conf_mat))
    kfold = KFold(n_splits=10,shuffle=True,random_state=42)
    DS_train =cross_val_score(tree, x_train, y_train, cv=kfold, scoring='accuracy')
    pd.DataFrame(DS_train,columns=['Scores'])
    print("Mean Accuracy: %.3f%%, Standard Deviation: (%.2f%%)" % (DS_train.mean()*100.0, DS_train.std()*100.0))

    kfold = KFold(n_splits=10,shuffle=True,random_state=42)
    DS_train =cross_val_score(tree, x_test, y_test, cv=kfold, scoring='accuracy')
    pd.DataFrame(DS_train,columns=['Scores'])
    print("Mean Accuracy: %.3f%%, Standard Deviation: (%.2f%%)" % (DS_train.mean()*100.0, DS_train.std()*100.0))

    rfc=RandomForestClassifier(random_state=42)

    rnd_forest = RandomForestClassifier(random_state=42, max_features='sqrt', n_estimators= 500, max_depth=13)
    rnd_forest.fit(x_train,y_train)
    preds=rnd_forest.predict(x_test)
    print(x_test[0])
    print(preds[0])
    conf_mat = confusion_matrix(y_test, preds)
    df_cm = pd.DataFrame(conf_mat, index=df['Disease'].unique(), columns=df['Disease'].unique())
    print('F1-score% =', f1_score(y_test, preds, average='macro')*100, '|', 'Accuracy% =', accuracy_score(y_test, preds)*100)
    sns.heatmap(df_cm)

    kfold = KFold(n_splits=10,shuffle=True,random_state=42)
    rnd_forest_train =cross_val_score(rnd_forest, x_train, y_train, cv=kfold, scoring='accuracy')
    pd.DataFrame(rnd_forest_train,columns=['Scores'])
    print("Mean Accuracy: %.3f%%, Standard Deviation: (%.2f%%)" % (rnd_forest_train.mean()*100.0, rnd_forest_train.std()*100.0))

    kfold = KFold(n_splits=10,shuffle=True,random_state=42)
    rnd_forest_test =cross_val_score(rnd_forest, x_test, y_test, cv=kfold, scoring='accuracy')
    pd.DataFrame(rnd_forest_test,columns=['Scores'])
    print("Mean Accuracy: %.3f%%, Standard Deviation: (%.2f%%)" % (rnd_forest_test.mean()*100.0, rnd_forest_test.std()*100.0))
    

    svm_model = SVC(kernel='linear', C=1.0)
    svm_model.fit(x_train, y_train)

    # Make predictions on the test set
    svm_preds = svm_model.predict(x_test)

    print("Features of the first test instance:", x_test[0])
    print("Predicted label for the first test instance:", svm_preds[0])
    # Calculate and print F1-score and accuracy for SVM
    svm_conf_mat = confusion_matrix(y_test, svm_preds)
    df_svm_cm = pd.DataFrame(svm_conf_mat, index=df['Disease'].unique(), columns=df['Disease'].unique())
    print('Support Vector Machines (SVM):')
    print('F1-score% =', f1_score(y_test, svm_preds, average='macro') * 100, '|', 'Accuracy% =', accuracy_score(y_test, svm_preds) * 100)
    print(x_train, y_train)

    # Visualize confusion matrix
    sns.heatmap(df_svm_cm)
    plt.show()

    knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors
    knn.fit(x_train, y_train)

    # Make predictions on the test set
    knn_preds = knn.predict(x_test)

    # Print features of the first test instance and its predicted label
    print("Features of the first test instance:", x_test[0])
    print("Predicted label for the first test instance:", knn_preds[0])

    # Calculate and print F1-score and accuracy for KNN
    knn_conf_mat = confusion_matrix(y_test, knn_preds)
    df_knn_cm = pd.DataFrame(knn_conf_mat, index=df['Disease'].unique(), columns=df['Disease'].unique())
    print('k-Nearest Neighbors (KNN):')
    print('F1-score% =', f1_score(y_test, knn_preds, average='macro') * 100, '|', 'Accuracy% =', accuracy_score(y_test, knn_preds) * 100)

    # Visualize confusion matrix
    sns.heatmap(df_knn_cm)
    plt.show()

    discrp = pd.read_csv("C:/Users/LENOVO/Desktop/project/symptom_Description.csv")

    ektra7at = pd.read_csv("symptom_precaution.csv")

    def predd(x,S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13,S14,S15,S16,S17):
        psymptoms = [S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13,S14,S15,S16,S17]
        #print(psymptoms)
        sev=0
        ct=0
        print(sev)
        a = np.array(df1["Symptom"])
        b = np.array(df1["weight"])
        for j in range(len(psymptoms)):
            for k in range(len(a)):
                if psymptoms[j]==a[k]:
                    psymptoms[j]=b[k]
                    sev+=b[k]
                    ct+=1


        psy = [psymptoms]
        pred2 = x.predict(psy)
        print('\n\n')
        disp= discrp[discrp['Disease']==pred2[0]]
        disp = disp.values[0][1]
        recomnd = ektra7at[ektra7at['Disease']==pred2[0]]
        c=np.where(ektra7at['Disease']==pred2[0])[0][0]
        precuation_list=[]
        for i in range(1,len(ektra7at.iloc[c])):
            precuation_list.append(ektra7at.iloc[c,i])
        print("The Disease Name: ",pred2[0])
        print("The Disease Discription: ",disp)
        t=disp
        print("Recommended Things to do at home: ")
        tt=''
        # for i in precuation_list:
        #     t+=*i
        severity=sev/ct
        sev=0
        ct=0
        return {'disease':pred2[0],'disease_dres':disp,'precaution':precuation_list,'severity':severity} 
    

    



    print('below Kmeans ')
    # Select the columns with symptoms
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

    print('above Kmeans ')






    
    sympList=df1["Symptom"].to_list()
    # sympList=df1["symps"].to_list()
    print(type(sympList),sympList)
    # s=predd(rnd_forest,sympList[0],sympList[1],sympList[2],0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    
    params = []  # Initialize all parameters to zero
    num_symps = len(symps)

    for i in range(num_symps):
        params.append(symps[i])

    lp=len(params)

    sorted_params = sorted(params, key=lambda x: symptom_index[x])

    params=sorted_params

    for i in range(lp,17):
        params.append(0)
    
    print('params ',params)

    # dis=[s1.disease,s2.disease,s3.disease,s4.disease]
    # print(statistics.mode(dis))
        
    # svm_preds = svm_model.predict(x_test)
    # rf_preds = rnd_forest.predict(x_test)
    # tree_preds = tree.predict(x_test)
    # knn_preds = knn.predict(x_test)

    # # Combine predictions using the voting method (mode)
    # combined_preds = []
    # for i in range(len(x_test)):
    #     preds = [svm_preds[i], rf_preds[i], tree_preds[i], knn_preds[i]]
    #     mode_pred = max(set(preds), key=preds.count)  # Calculate mode
    #     combined_preds.append(mode_pred)

    # # Calculate and print confusion matrix for the combined predictions
    # combined_conf_mat = confusion_matrix(y_test, combined_preds)
    # df_combined_cm = pd.DataFrame(combined_conf_mat, index=df['Disease'].unique(), columns=df['Disease'].unique())
    # print('Combined Voting Method:')
    # print(df_combined_cm)

    # Calculate and print F1-score and accuracy for the combined predictions
    # print('F1-score% =', f1_score(y_test, combined_preds, average='macro') * 100)
    # print('Accuracy% =', accuracy_score(y_test, combined_preds) * 100)

    s3=predd(svm_model,*params)
    s1=predd(rnd_forest, *params)
    s4=predd(tree,*params)
    s2=predd(knn,*params)

    dis=[s1['disease'],s2['disease'],s3['disease'],s4['disease']]
    calc=statistics.mode(dis)

    print('calc ',calc)

    disp= discrp[discrp['Disease']==calc]
    print('disp ',disp)
    disp = disp.values[0][1]
    recomnd = ektra7at[ektra7at['Disease']==calc]
    c=np.where(ektra7at['Disease']==calc)[0][0]
    precuation_list=[]
    for i in range(1,len(ektra7at.iloc[c])):
        precuation_list.append(ektra7at.iloc[c,i])

    s={'disease':calc,'disease_dres':disp,'precaution':precuation_list} 
    print('ret val ',s)

    # s=predd(rnd_forest,symps[0],symps[1],symps[2],0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    # print('s2 ',s2)
    for i in l1:
        if(s['disease'] in i):
            s['possible_dis']=i
    return s

if __name__ == '__main__':  
   app.run()