import csv

# Initialize an empty dictionary to store diseases and their associated unique symptoms
disease_symptoms_dict = {}

# Read the dataset from the CSV file
with open('C:/Users/LENOVO/Desktop/project/client/api/dataset.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        disease = row[0]
        symptoms = set(row[1:])  # Extract symptoms from the row and convert to a set to get unique values
        if disease not in disease_symptoms_dict:
            disease_symptoms_dict[disease] = symptoms
        else:
            disease_symptoms_dict[disease].update(symptoms)  # Update the symptoms for the disease

# Print the dictionary
for disease, symptoms in disease_symptoms_dict.items():
    # print(f"Disease: {disease}, Symptoms:",list(symptoms)[1:])
    t=list(symptoms)[1:]
    for i in range(len(t)):
        # t[i].replace(' ','')
        tt=t[i]
        tt.replace('_',' ')
        t[i]=tt
    print(f"Disease: {disease}, Symptoms:",t)

    print('\n')
