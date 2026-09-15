import bayespy as bp
import numpy as np
import csv
from colorama import init, Fore

init()

# --- Enumerations ---
ageEnum = {'SuperSeniorCitizen': 0, 'SeniorCitizen': 1, 'MiddleAged': 2, 'Youth': 3, 'Teen': 4}
genderEnum = {'Male': 0, 'Female': 1}
familyHistoryEnum = {'Yes': 0, 'No': 1}
dietEnum = {'High': 0, 'Medium': 1, 'Low': 2}
lifeStyleEnum = {'Athlete': 0, 'Active': 1, 'Moderate': 2, 'Sedetary': 3}
cholesterolEnum = {'High': 0, 'BorderLine': 1, 'Normal': 2}
heartDiseaseEnum = {'Yes': 0, 'No': 1}

# --- Load dataset ---
with open('heart_disease_data.csv') as csvfile:
    lines = csv.reader(csvfile)
    dataset = list(lines)

data = []
for x in dataset:
    data.append([
        ageEnum[x[0]],
        genderEnum[x[1]],
        familyHistoryEnum[x[2]],
        dietEnum[x[3]],
        lifeStyleEnum[x[4]],
        cholesterolEnum[x[5]],
        heartDiseaseEnum[x[6]]
    ])

data = np.array(data)
N = len(data)

# --- Define Bayesian nodes ---
p_age = bp.nodes.Dirichlet(1.0 * np.ones(5))
age = bp.nodes.Categorical(p_age, plates=(N,))
age.observe(data[:, 0])

p_gender = bp.nodes.Dirichlet(1.0 * np.ones(2))
gender = bp.nodes.Categorical(p_gender, plates=(N,))
gender.observe(data[:, 1])

p_familyhistory = bp.nodes.Dirichlet(1.0 * np.ones(2))
familyhistory = bp.nodes.Categorical(p_familyhistory, plates=(N,))
familyhistory.observe(data[:, 2])

p_diet = bp.nodes.Dirichlet(1.0 * np.ones(3))
diet = bp.nodes.Categorical(p_diet, plates=(N,))
diet.observe(data[:, 3])

p_lifestyle = bp.nodes.Dirichlet(1.0 * np.ones(4))
lifestyle = bp.nodes.Categorical(p_lifestyle, plates=(N,))
lifestyle.observe(data[:, 4])

p_cholesterol = bp.nodes.Dirichlet(1.0 * np.ones(3))
cholesterol = bp.nodes.Categorical(p_cholesterol, plates=(N,))
cholesterol.observe(data[:, 5])

# Heart disease node depends on all features
p_heartdisease = bp.nodes.Dirichlet(np.ones(2), plates=(5, 2, 2, 3, 4, 3))
heartdisease = bp.nodes.MultiMixture(
    [age, gender, familyhistory, diet, lifestyle, cholesterol],
    bp.nodes.Categorical,
    p_heartdisease
)
heartdisease.observe(data[:, 6])

# Update posterior
p_heartdisease.update()

# --- Interactive prediction loop ---
m = 0
while m == 0:
    print("\nEnter patient details using the codes shown below:\n")
    print("Age:", ageEnum)
    print("Gender:", genderEnum)
    print("Family History:", familyHistoryEnum)
    print("Diet:", dietEnum)
    print("Lifestyle:", lifeStyleEnum)
    print("Cholesterol:", cholesterolEnum)

    age_in = int(input("Enter Age code: "))
    gender_in = int(input("Enter Gender code: "))
    fam_in = int(input("Enter FamilyHistory code: "))
    diet_in = int(input("Enter Diet code: "))
    life_in = int(input("Enter Lifestyle code: "))
    chol_in = int(input("Enter Cholesterol code: "))

    res = bp.nodes.MultiMixture(
        [age_in, gender_in, fam_in, diet_in, life_in, chol_in],
        bp.nodes.Categorical,
        p_heartdisease
    ).get_moments()[0][heartDiseaseEnum['Yes']]

    print(Fore.GREEN + "Probability(HeartDisease) = " + str(res) + Fore.RESET)

    m = int(input("Enter 0 to Continue, 1 to Exit: "))
