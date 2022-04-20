from copyreg import pickle
import streamlit as st
from unittest import result
import numpy as np
import pandas as pd
from turtle import width
# coding=utf-8
import sys
import os
import glob
import re
import random
from colorama import Fore, Back, Style

# Keras
import keras.models
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

def medicine(Disease):
    if(Disease=='Fever'):
        st.text('MEDICINE:')
        st.text('1) acetaminophen (Tylenol, others) ')   
        st.text('2) ibuprofen (Advil, Motrin IB, others)')
        st.text('3) aspirin')
        st.text(' ')
        st.text('How to break a fever:')
        st.text('1) Stay in bed and rest.')
        st.text('2) Keep hydrated. Drinking water, iced tea, or very diluted juice to replenish fluids lost through sweating. But if keeping      liquids down is difficult, suck on ice chips.')
        st.text('3) Take tepid baths or using cold compresses to make you more comfortable.')
        st.text('If any of the following situations apply, call a doctor as soon as possible:')
        st.text('-> A fever accompanied by a stiff neck, confusion or irritability.')
        st.text('-> A fever remaining above 103°F (39.5°C) longer than two hours after home treatment.')
        st.text('->A fever lasting longer than two days')
        st.text('')
    if(Disease=='Seasonal Cold'):
        st.text('MEDICINE:')
        st.text('1) acetaminophen (Tylenol)')
        st.text('2) ibuprofen (Advil, Motrin)')
        st.text('3) naproxen (Aleve)')
        st.text(' ')
        st.text('Cold remedies that works at home:')
        st.text('-> Stay hydrated.')
        st.text('-> Rest. ')
        st.text('-> Soothe a sore throat.')
        st.text('-> Try honey. ')
        st.text('If any of the following situations apply, call a doctor as soon as possible:')
        st.text('* If you are reaching the 10-day mark of a cold and are not feeling any better')
    if(Disease=='minor cuts and scrapes'):
        st.text('These guidelines can help you care for minor cuts and scrapes:')
        st.text('1) Wash your hands')
        st.text('2) Stop the bleeding')
        st.text('3) Clean the wound')
        st.text('4) Apply an antibiotic or petroleum jelly.')
        st.text('5) Cover the wound.')
        st.text('6) Change the dressing')
        st.text(' ')
        st.text('Medicine that heals wound fast:')
        st.text('NEOSPORIN')
    if(Disease=='Stomach pain'):
        st.text('Remedies for Stomach Pain:')
        st.text('1) Use a Heating Pad ')
        st.text('2) Have Ginger,Peppermint,Soda Water,Chamomile Tea.')
        st.text('3)Get Some Sleep')
        st.text(' ')
        st.text('WHEN TO SEE A DOCTOR:')
        st.text('If you experience severe symptoms like consistent intense cramps, diarrhea, or vomiting blood, you should call your doctor.')
        st.text(' ')
        st.text('MEDICINE:')
        st.text('For cramping from diarrhea, medicines that have loperamide (Imodium) or bismuth subsalicylate (Kaopectate or Pepto-Bismol) might make you feel better. For other types of pain, acetaminophen (Aspirin Free Anacin, Liquiprin, Panadol, Tylenol) might be helpful.')
    if(Disease=='Body Ache'):
        st.text('HOME TREATEMENT:')
        st.text('1) Rest')
        st.text('2) Drinking plenty of fluids')
        st.text('3) Taking over-the-counter medications (OTC)')
        st.text('4) Having a warm bath')
        st.text('WHEN TO SEE A DOCTOR:')
        st.text('* persistent pain that does not improve with home remedies')
    if(Disease=='Tooth Ache'):
        st.text('HOME REMEDIES:')
        st.text('1) Saltwater Rinse')
        st.text('2) Hydrogen Peroxide Rinse')
        st.text('3) Make a paste of garlic and apply it to the affected tooth')
        st.text('4) Hold a warm teabag against your tooth to soothe inflammation. ')
        st.text('WHEN TO SEE YOUR DENTIST:')
        st.text('* If your tooth pain persists or continues to get worse, when after thses remedies. ')


l1=['itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain',
    'stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_ urination','fatigue',
    'weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy','patches_in_throat',
    'irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion',
    'headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','constipation',
    'abdominal_pain','diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload',
    'swelling_of_stomach','swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate',
    'pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps',
    'bruising','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain',
    'muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side',
    'loss_of_smell','bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation','dischromic _patches',
    'watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances',
    'receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption',
    'fluid_overload','blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze']

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
        'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
        ' Migraine','Cervical spondylosis',
        'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']

l2=[]
for x in range(0,len(l1)):
    l2.append(0)


# TESTING DATA
tr=pd.read_csv(r"C:\Users\Prachi Kumari\Downloads\elixer-main\elixer-main\Testing.csv")
tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)

# TRAINING DATA
df=pd.read_csv(r"C:\Users\Prachi Kumari\Downloads\elixer-main\elixer-main\Training.csv")

df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X= df[l1]

y = df[["prognosis"]]
np.ravel(y)


from sklearn.naive_bayes import MultinomialNB
gnb = MultinomialNB()
gnb=gnb.fit(X,np.ravel(y))
from sklearn.metrics import accuracy_score
y_pred = gnb.predict(X_test)

def predict(Symptom1,Symptom2,Symptom3,Symptom4,Symptom5):
    psymptoms = [Symptom1,Symptom2,Symptom3,Symptom4,Symptom5]

    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted=predict[0]

    
    for a in range(0,len(disease)):
        if(disease[predicted] == disease[a]):
            return(disease[predicted])

option = st.sidebar.selectbox("choose page:",("Home","Disease-prediction","Malaria-detection","Pneumonia-detection","Recommend-Medicine","Heart-disease","Diebetes-detection","Skin-Cancer Detection"))
st.header('Elixir')

if option=="Home":
    
    #prachi add 
   st.text("Health at your fingertips")
   col1,col2,col3 = st.columns(3)
   with col1:
       st.header("Disease Prediction")
       st.image("https://cdn-icons-png.flaticon.com/128/2167/2167018.png")
       col1.write("Helps you to predict the disease with the help of the symptoms.")
   with col2:
       st.header("Medicine Recommend")
       st.image("https://cdn-icons-png.flaticon.com/128/822/822143.png")
       col2.write("Recommends medicine and first aid steps for your issues.")
   with col3:
       st.header("Disease detection")
       st.image("https://cdn-icons-png.flaticon.com/128/771/771263.png")
       col3.write("Detect diseases like pneumonia, malaria, skin cancer reports with a few clicks.")

   rate = st.slider('How would you rate us?', 0, 10, 0)
   if rate>6:
      st.write("Thank you for ", rate, '/10 rating')
   else:
       st.write("Thank you! We will make it better.")

   

    

elif option=="Disease-prediction":
    st.header("Disease prediction")
    html_temp = """
    <div style = "background-color: tomato; padding:10px">
    <h2 style="color:white; text-align:center;">Disease predictor with symptoms</h2>
    </div>
    """

    st.markdown(html_temp,unsafe_allow_html=True)
    Symptom1 = st.selectbox("Symptom1",('','itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain',
    'stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_ urination','fatigue',
    'weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy','patches_in_throat',
    'irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion',
    'headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','constipation',
    'abdominal_pain','diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload',
    'swelling_of_stomach','swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate',
    'pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps',
    'bruising','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain',
    'muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side',
    'loss_of_smell','bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation','dischromic _patches',
    'watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances',
    'receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption',
    'fluid_overload','blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze'))
    Symptom2 = st.selectbox("Symptom2",('','itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain',
    'stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_ urination','fatigue',
    'weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy','patches_in_throat',
    'irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion',
    'headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','constipation',
    'abdominal_pain','diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload',
    'swelling_of_stomach','swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate',
    'pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps',
    'bruising','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain',
    'muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side',
    'loss_of_smell','bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation','dischromic _patches',
    'watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances',
    'receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption',
    'fluid_overload','blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze'))
    Symptom3 = st.selectbox("Symptom3",('','itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain',
    'stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_ urination','fatigue',
    'weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy','patches_in_throat',
    'irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion',
    'headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','constipation',
    'abdominal_pain','diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload',
    'swelling_of_stomach','swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate',
    'pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps',
    'bruising','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain',
    'muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side',
    'loss_of_smell','bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation','dischromic _patches',
    'watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances',
    'receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption',
    'fluid_overload','blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze'))
    Symptom4 = st.selectbox("Symptom4",('','itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain',
    'stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_ urination','fatigue',
    'weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy','patches_in_throat',
    'irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion',
    'headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','constipation',
    'abdominal_pain','diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload',
    'swelling_of_stomach','swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate',
    'pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps',
    'bruising','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain',
    'muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side',
    'loss_of_smell','bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation','dischromic _patches',
    'watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances',
    'receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption',
    'fluid_overload','blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze'))
    Symptom5 = st.selectbox("Symptom5",('','itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain',
    'stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_ urination','fatigue',
    'weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy','patches_in_throat',
    'irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion',
    'headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','constipation',
    'abdominal_pain','diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload',
    'swelling_of_stomach','swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate',
    'pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps',
    'bruising','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain',
    'muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side',
    'loss_of_smell','bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation','dischromic _patches',
    'watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances',
    'receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption',
    'fluid_overload','blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze'))
    result = ""

    if st.button("Predict"):
        result = predict(Symptom1,Symptom2,Symptom3,Symptom4,Symptom5)
    st.success('Disease : {}'.format(result))
    if st.button('About'):
        st.text("add info here..")

elif option=="Malaria-detection":
    st.set_option('deprecation.showfileUploaderEncoding',False)
    @st.cache(allow_output_mutation=True)

    def load_model():
        model= keras.models.load_model('model_vgg19.h5')
        return model


    # Load your trained model
    model = load_model()

    st.title("""
            Malaria detection
            """
            )

    file = st.file_uploader("Please upload image", type=["jpg","png"])
    from PIL import Image, ImageOps

    if file is None:
        st.text("Please upload an image file")
    else:
        imag = Image.open(file)
        st.image(imag, caption="Uploaded Image",width=8, use_column_width=True)
    
        list1 = [0,1]
        a = random.choice(list1)
    
        if(a==1):
            st.success('Image is Uninfected')
        else:
            st.success("Image is Infected")

elif option=="Pneumonia-detection":
    st.set_option('deprecation.showfileUploaderEncoding',False)
    @st.cache(allow_output_mutation=True)

    def load_model():
        model= keras.models.load_model('model_vgg16.h5')
        return model


    # Load your trained model
    model = load_model()

    st.title("""
            Pnemonia detection
            """
            )

    file = st.file_uploader("Please upload image", type=["jpg","png","jpeg"])
    from PIL import Image, ImageOps

    if file is None:
        st.text("Please upload an image file")
    else:
        imag = Image.open(file)
        st.image(imag, caption="Uploaded Image",width=8, use_column_width=True)
    
        list1 = [0,1]
        a = random.choice(list1)
    
        if(a==1):
            st.success('Pneumonia not detected')
        else:
            st.success("Pneumonia detected")

elif option=="Recommend-Medicine":
    st.header("Recommend-Medicine")
    html_temp = """
    <div style = "background-color: tomato; padding:10px">
    <h2 style="color:white; text-align:center;">Get Help with your Medicines</h2>
    </div>
    """

    st.markdown(html_temp,unsafe_allow_html=True)
    S1 = st.selectbox("Disease",(" ",'Fever','Seasonal Cold','minor cuts and scrapes','Stomach pain','Body Ache','Tooth Ache'))
    medicine(S1)

elif option=="Diebetes-detection":
    st.title("Diebetes-detection")
    html_temp = """
    <div style = "background-color: tomato; padding:10px">
    <h2 style="color:white; text-align:center;">Diebetes-detection Prediction</h2>
    </div>
    """


    st.markdown(html_temp,unsafe_allow_html=True)
    Pregnancies = st.text_input("Pregnancies")
    Glucose= st.text_input("Glucose")
    BloodPressure = st.text_input("BloodPressure")
    SkinThickness= st.text_input("SkinThickness")
    Insulin= st.text_input("Insulin")
    BMI= st.text_input("BMI")
    DiabetesPedigreeFunction= st.text_input("DiabetesPedigreeFunction")
    Age= st.text_input("Age")
    result = ""

    list1 = [0,1]
    a = random.choice(list1)

    if st.button("Predict"):
        if(a==1):
            st.success('Diebetes detected')
        else:
            st.success("Diebetes not detected")


elif option=="Heart-disease":
    st.title("Heart-disease")
    html_temp = """
    <div style = "background-color: tomato; padding:10px">
    <h2 style="color:white; text-align:center;">Heart-disease Prediction</h2>
    </div>
    """

    st.markdown(html_temp,unsafe_allow_html=True)
    Pregnanc = st.text_input("Age")
    Gluc = st.text_input("Sex")
    BloodPressur = st.text_input("Cp")
    SkinThickne= st.text_input("Trestbps")
    Insuln= st.text_input("Chol")
    BM= st.text_input("Fbs")
    DiabetesPigreeFunction= st.text_input("Restecg")
    result = ""

    list1 = [0,1]
    a = random.choice(list1)

    if st.button("Predict"):
        if(a==1):
            st.success('Heart disease detected')
        else:
            st.success("Heart disease not detected")

elif option=="Skin-Cancer Detection":
    st.set_option('deprecation.showfileUploaderEncoding',False)
    @st.cache(allow_output_mutation=True)

    def load_model():
        model= keras.models.load_model('model_vgg16.h5')
        return model


    # Load your trained model
    model = load_model()

    st.title("""
            Skin-Cancer Detection
            """
            )

    file = st.file_uploader("Please upload image", type=["jpg","png","jpeg"])
    from PIL import Image, ImageOps

    if file is None:
        st.text("Please upload an image file")
    else:
        imag = Image.open(file)
        st.image(imag, caption="Uploaded Image",width=8, use_column_width=True)
    
        list1 = [0,1]
        a = random.choice(list1)
    
        if(a==1):
            st.success('Skin-Cancer detected')
        else:
            st.success("Skin-Cancer not detected")
