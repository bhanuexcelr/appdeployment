# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:01:17 2020

"""

import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression
import pickle 

st.title('Model Deployment: Logistic Regression')

st.sidebar.header('User Input Parameters')

def user_input_features():
    CLMSEX = st.sidebar.selectbox('Gender',('1','0'))
    CLMINSUR = st.sidebar.selectbox('Insurance',('1','0'))
    SEATBELT = st.sidebar.selectbox('SeatBelt',('1','0'))
    CLMAGE = st.sidebar.number_input("Insert the Age",step=1)
    LOSS = st.sidebar.number_input("Insert Loss",step=1)
    data = {'CLMSEX':CLMSEX,
            'CLMINSUR':CLMINSUR,
            'SEATBELT':SEATBELT,
            'CLMAGE':CLMAGE,
            'LOSS':LOSS}
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)



#loaded_model=pickle.load(open('finalized_model.pkl','rb'))

#build the model
claimants = pd.read_csv("claimants.csv")
claimants.drop(["CASENUM"],inplace=True,axis = 1)
claimants = claimants.dropna()
X = claimants.iloc[:,1:]
Y = claimants.iloc[:,0]

classifier=LogisticRegression()
classifier.fit(X,Y)

prediction = classifier.predict(df)
prediction_proba = classifier.predict_proba(df)

st.write("probabilities")
st.write(prediction_proba[0])

st.subheader('Predicted Result')

if prediction==0:
    st.write("Claimant is likely to hire an attorney")
else:
    st.write("Claimant is likely to not hire an attorney")

