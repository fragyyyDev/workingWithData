import requests
import zipfile
import os
import pandas as pd
import lxml
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tkinter import ttk, messagebox


#pip install matplotlib requests pandas lxml

# Promenny na API / stahnuti
KAGGLE_USERNAME = 'tadejalovec'
KAGGLE_KEY = 'c8cb958a944e4c161551ae9bc7bce507'
url = 'https://www.kaggle.com/api/v1/datasets/download/iammustafatz/diabetes-prediction-dataset'
auth = (KAGGLE_USERNAME, KAGGLE_KEY)

# Stahujeme dataset
response = requests.get(url, auth=auth, stream=True)
if response.status_code == 200:
    with open('diabetes-prediction-dataset.zip', 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print("Stahnuls to ✅")
else:
    print(f"Nestahlo se to . Kod erroru: {response.status_code} ❌")
    exit()

# Ze zipu extrahujeme
with zipfile.ZipFile('diabetes-prediction-dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('diabetes-prediction-dataset')

os.remove("diabetes-prediction-dataset.zip") # Odstranime starej zip
print("Odstranili jsme starej zip ✅")
print("Podarilo se vyndat dataset ze zipu ✅")

# Tady si delame dataframe pomoci pandas z toho csv
dataframe = pd.read_csv('diabetes-prediction-dataset/diabetes_prediction_dataset.csv')
# Previst dataframe na JSON
if(os.path.exists('diabetes-prediction-dataset/diabetes_prediction_dataset.json') or os.path.exists('diabetes-prediction-dataset/diabetes_prediction_dataset.xml')):
    print('Netvorim nove pretvorene XML a JSON jelikoz jiz existuji')
else:
    dataframe.to_json("diabetes-prediction-dataset.json")
    print("Uspesne prevedeno na .json ✅")
    dataframe.to_xml("diabetes-prediction-dataset.xml", attr_cols=['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes'])
    print("Uspesne prevedeno na .xml ✅")

# Dame ty prevedeny datasety do souboru kde je to csv pokud tam nejsou
if(os.path.exists('diabetes-prediction-dataset/diabetes_prediction_dataset.json') or os.path.exists('diabetes-prediction-dataset/diabetes_prediction_dataset.xml')):
    print('Neposunul jsem files to slozky jelikoz ve slozce jiz jsou')
else:
    os.rename("diabetes-prediction-dataset.json", "diabetes-prediction-dataset/diabetes_prediction_dataset.json")
    os.rename("diabetes-prediction-dataset.xml", "diabetes-prediction-dataset/diabetes_prediction_dataset.xml")
    print("Uspesne presunuty ruzne druhy datasetu do souboru ✅")

# Data na charts
dataframe_diabetesPositive = dataframe.loc[dataframe['diabetes'] == 1]
dataframe_diabetesPositiveBMIsOverweight = dataframe_diabetesPositive.loc[dataframe_diabetesPositive['bmi'] >= 25]
dataframe_diabetesPositiveBMIsNormal = dataframe_diabetesPositive.loc[(dataframe_diabetesPositive['bmi'] > 18.5) & (dataframe_diabetesPositive['bmi'] < 25)]
dataframe_diabetesPositiveBMIsUnderweight = dataframe_diabetesPositive.loc[dataframe_diabetesPositive['bmi'] < 18.5]

dataframe_heartDiseasePositive = dataframe.loc[dataframe['heart_disease'] == 1]
dataframe_heartDiseasePositiveBMIsOverweight = dataframe_heartDiseasePositive[dataframe_heartDiseasePositive['bmi'] >= 25]
dataframe_heartDiseasePositiveBMIsNormal = dataframe_heartDiseasePositive[(dataframe_heartDiseasePositive['bmi'] > 18.5) & (dataframe_heartDiseasePositive['bmi'] < 25)]
dataframe_heartDiseasePositiveBMIsUnderweight = dataframe_heartDiseasePositive[dataframe_heartDiseasePositive['bmi'] < 18.5]

dataframe_hypertension = dataframe.loc[dataframe['hypertension'] == 1]
dataframe_hypertensionPositiveBMIsOverweight = dataframe_hypertension[dataframe_hypertension['bmi'] >= 25]
dataframe_hypertensionPositiveBMIsNormal = dataframe_hypertension[(dataframe_hypertension['bmi'] > 18.5) & (dataframe_hypertension['bmi'] < 25)]
dataframe_hypertensionPositiveBMIsUnderweight = dataframe_hypertension[dataframe_hypertension['bmi'] < 18.5]

dataframe_smokers = dataframe.loc[
    (dataframe['smoking_history'] == 'current') | 
    (dataframe['smoking_history'] == 'former') | 
    (dataframe['smoking_history'] == 'not current')
]

dataframe_nonsmokers = dataframe.loc[dataframe['smoking_history'] == 'never']

dataframe_unhealthySmokers = dataframe_smokers[
    (dataframe_smokers['heart_disease'] == 1) | 
    (dataframe_smokers['diabetes'] == 1) | 
    (dataframe_smokers['hypertension'] == 1)
]

dataframe_unhealthyNoNSmokers = dataframe_nonsmokers[
    (dataframe_nonsmokers['heart_disease'] == 1) | 
    (dataframe_nonsmokers['diabetes'] == 1) | 
    (dataframe_nonsmokers['hypertension'] == 1)
]

average_bmi_diabetes = dataframe.loc[dataframe['diabetes'] == 1, 'bmi'].mean()
average_bmi_heart_disease = dataframe.loc[dataframe['heart_disease'] == 1, 'bmi'].mean()
average_bmi_hypertension = dataframe.loc[dataframe['hypertension'] == 1, 'bmi'].mean()


dataframe_diabetic = dataframe.loc[dataframe['diabetes'] == 1]
dataframe_non_diabetic = dataframe.loc[dataframe['diabetes'] == 0]


dataframe_with_conditions = dataframe.loc[
    (dataframe['diabetes'] == 1) | (dataframe['heart_disease'] == 1) | (dataframe['hypertension'] == 1)
]

#trenink modelu

X = dataframe[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
y = dataframe[['diabetes', 'hypertension', 'heart_disease']]
#testovaci prostredi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Pouzijeme model na logistickou regresi a vybereme ze model je na vice outputu
base_model = LogisticRegression(max_iter=200)
model = MultiOutputClassifier(base_model)
model.fit(X_train, y_train)

# Predpovidani na testovacim prostredi a presnost
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Multi-label model accuracy: {accuracy * 100:.2f}%")

#funkce co to pocita
def predict_conditions(age, bmi, HbA1c_level, blood_glucose_level):
    input_data = pd.DataFrame([[age, bmi, HbA1c_level, blood_glucose_level]], 
                              columns=['age', 'bmi', 'HbA1c_level', 'blood_glucose_level'])
    probabilities = model.predict_proba(input_data)
    
    # probability
    diabetes_prob = float(probabilities[0][0][1]) * 100
    hypertension_prob = float(probabilities[1][0][1]) * 100
    heart_disease_prob = float(probabilities[2][0][1]) * 100
    
    diabetes_prob = round(diabetes_prob, 1)
    hypertension_prob = round(hypertension_prob, 1)
    heart_disease_prob = round(heart_disease_prob, 1)
    
    # formatovani aby to bylo v procentech
    diabetes_prob = f"{diabetes_prob}%"
    hypertension_prob = f"{hypertension_prob}%"
    heart_disease_prob = f"{heart_disease_prob}%"
    
    return diabetes_prob, hypertension_prob, heart_disease_prob

def show_hba1c_distribution():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(dataframe_diabetic['HbA1c_level'], bins=10, alpha=0.6, color='orange', label="Diabetic")
    ax.hist(dataframe_non_diabetic['HbA1c_level'], bins=10, alpha=0.6, color='green', label="Non-Diabetic")
    ax.set_title("HbA1c Level Distribution (Diabetic vs Non-Diabetic)")
    ax.set_xlabel("HbA1c Level (%)")
    ax.set_ylabel("Frequency")
    ax.legend(title="Health Status", loc="upper right", frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig

def show_avg_bmi():
    conditions = ['Diabetes', 'Heart Disease', 'Hypertension']
    avg_bmi_values = [average_bmi_diabetes, average_bmi_heart_disease, average_bmi_hypertension]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(conditions, avg_bmi_values, color=['yellow', 'red', 'blue'])
    
    ax.set_title("Average BMI by Health Condition")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Average BMI")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    return fig


def show_age_distribution():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(dataframe_with_conditions['age'], bins=10, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_title("Age Distribution of People with Health Conditions")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Frequency")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig


#toggle diabetes
def show_diabetes_chart():
    labels_diabetes = ['Overweight', 'Normal Weight', 'Underweight']
    sizes_diabetes = [
        len(dataframe_diabetesPositiveBMIsOverweight),
        len(dataframe_diabetesPositiveBMIsNormal),
        len(dataframe_diabetesPositiveBMIsUnderweight)
    ]
    colors_diabetes = ['yellow', 'black', 'orange']
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, _, autotexts = ax.pie(sizes_diabetes, labels=labels_diabetes, colors=colors_diabetes, autopct='%1.1f%%', startangle=140)
    ax.set_title("BMI in diabetes", fontsize=14, weight="bold")
    ax.legend(wedges, labels_diabetes, title="BMI Category", loc="best", fontsize=10)
    plt.setp(autotexts, size=10, weight="bold")
    return fig

def show_heart_disease_chart():
    labels_heart_disease = ['Overweight', 'Normal Weight', 'Underweight']
    sizes_heart_disease = [
        len(dataframe_heartDiseasePositiveBMIsOverweight),
        len(dataframe_heartDiseasePositiveBMIsNormal),
        len(dataframe_heartDiseasePositiveBMIsUnderweight)
    ]
    colors_heart_disease = ['#ff6666', '#ff9999', '#ffcccc']
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, _, autotexts = ax.pie(sizes_heart_disease, labels=labels_heart_disease, colors=colors_heart_disease, autopct='%1.1f%%', startangle=140)
    ax.set_title("BMI in heart diseases", fontsize=14, weight="bold")
    ax.legend(wedges, labels_heart_disease, title="BMI Category", loc="best", fontsize=10)
    plt.setp(autotexts, size=10, weight="bold")
    return fig

def show_smokers():
    labels_smokers = ['Smokers', 'Non-Smokers']
    sizes_smokers = [len(dataframe_unhealthySmokers), len(dataframe_unhealthyNoNSmokers)]
    colors_smokers = ['#3399ff', '#66ccff']
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, _, autotexts = ax.pie(sizes_smokers, labels=labels_smokers, colors=colors_smokers, autopct='%1.1f%%', startangle=140)
    ax.set_title("Smoking vs Nonsmoking", fontsize=14, weight="bold")
    ax.legend(wedges, labels_smokers, title="Smoking Status", loc="best", fontsize=10)
    plt.setp(autotexts, size=10, weight="bold")
    return fig


def show_hypertension_chart():
    labels_hypertension = ['Overweight', 'Normal Weight', 'Underweight']
    sizes_hypertension = [
        len(dataframe_hypertensionPositiveBMIsOverweight),
        len(dataframe_hypertensionPositiveBMIsNormal),
        len(dataframe_hypertensionPositiveBMIsUnderweight)
    ]
    colors_hypertension = ['#66b3ff', '#99ccff', '#cce6ff']
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, _, autotexts = ax.pie(sizes_hypertension, labels=labels_hypertension, colors=colors_hypertension, autopct='%1.1f%%', startangle=140)
    ax.set_title("BMI in hypertension", fontsize=14, weight="bold")
    ax.legend(wedges, labels_hypertension, title="BMI Category", loc="best", fontsize=10)
    plt.setp(autotexts, size=10, weight="bold")
    return fig

#na to aby se to refreshlo ten chart
def update_chart(event):
    selected_option = chart_selection.get()
    for widget in chart_frame.winfo_children():
        widget.destroy()

    if selected_option == "Diabetes":
        fig = show_diabetes_chart()
    elif selected_option == "Heart Disease":
        fig = show_heart_disease_chart()
    elif selected_option == "Hypertension":
        fig = show_hypertension_chart()
    elif selected_option == "Smoking History":
        fig = show_smokers()
    elif selected_option == "Age Distribution":
        fig = show_age_distribution()
    elif selected_option == "Average BMI by Condition":
        fig = show_avg_bmi()
    elif selected_option == "HbA1c Distribution":
        fig = show_hba1c_distribution()
    elif selected_option == "Health Analysis":
        display_health_analysis_inputs() 
        return
    else:
        return

    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

def display_health_analysis_inputs():
    for widget in chart_frame.winfo_children():
        widget.destroy()

    label_age = tk.Label(chart_frame, text="Enter Age:")
    label_age.pack()
    entry_age = tk.Entry(chart_frame)
    entry_age.pack()

    label_bmi = tk.Label(chart_frame, text="Enter BMI:")
    label_bmi.pack()
    entry_bmi = tk.Entry(chart_frame)
    entry_bmi.pack()

    label_hba1c = tk.Label(chart_frame, text="Enter HbA1c Level:")
    label_hba1c.pack()
    entry_hba1c = tk.Entry(chart_frame)
    entry_hba1c.pack()

    label_blood_glucose = tk.Label(chart_frame, text="Enter Blood Glucose Level:")
    label_blood_glucose.pack()
    entry_blood_glucose = tk.Entry(chart_frame)
    entry_blood_glucose.pack()

    #button na odeslani
    submit_button = tk.Button(chart_frame, text="Predict Conditions", 
                              command=lambda: handle_predict_conditions(entry_age, entry_bmi, entry_hba1c, entry_blood_glucose))
    submit_button.pack(pady=20)

#veme values z inputu  streli do funkce a napise nam info back
def handle_predict_conditions(entry_age, entry_bmi, entry_hba1c, entry_blood_glucose):
    try:
        
        age = float(entry_age.get())
        bmi = float(entry_bmi.get())
        hba1c = float(entry_hba1c.get())
        blood_glucose = float(entry_blood_glucose.get())

        diabetes_prob, hypertension_prob, heart_disease_prob = predict_conditions(age, bmi, hba1c, blood_glucose)

        messagebox.showinfo("Prediction Results Model accuracy is about 87.1%", 
                            
                            f"Diabetes: {diabetes_prob}\nHypertension: {hypertension_prob}\nHeart Disease: {heart_disease_prob}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")



# Tkinter setup
app = tk.Tk()
app.title("Health Data Analysis")
app.geometry("700x500")

# Dropdown 
chart_selection = ttk.Combobox(app, values=[
    "Health Analysis",
    "Diabetes", "Heart Disease", "Hypertension", "Smoking History", 
    "Age Distribution", "Average BMI by Condition", "HbA1c Distribution"
])
chart_selection.set("Health Analysis")
chart_selection.bind("<<ComboboxSelected>>", update_chart)
chart_selection.pack(pady=10)
chart_frame = tk.Frame(app)
chart_frame.pack(fill=tk.BOTH, expand=True)
display_health_analysis_inputs()

app.mainloop()