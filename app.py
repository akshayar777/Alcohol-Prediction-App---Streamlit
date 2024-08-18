import pandas as pd
import numpy as np
import pickle
import streamlit as st

model_file=open("modeloutput.pkl","rb")
model = pickle.load(model_file)

scaler_file = open("scaler.pkl", "rb")
scaler = pickle.load(scaler_file)

st.title('Alcohol Prediction App')
st.subheader("Application to predit the Total Litres of Pure Alcohol")

with st.expander('Data'):
    st.write('Raw data')
    df = pd.read_csv('beer-servings.csv')
    df1 = df[['beer_servings', 'spirit_servings', 'wine_servings', 'total_litres_of_pure_alcohol', 'continent']]
    df1

st.text("(Enter your details in the side bar)")
st.subheader("Your details are : ")


# Create sliders for beer, spirit & wine serving input datas
st.sidebar.header("Input Features")
beer_input = st.sidebar.number_input("Beer Serving",0,376,76)
spirit_input = st.sidebar.number_input("Spirit Serving",0,438,60)
wine_input = st.sidebar.number_input("Wine Serving",0,370,50)


# Create the selectbox for Continent
conti_input = st.sidebar.selectbox('Choose the continent:', 
                                   ['Select a continent', 'Africa', 'Asia', 'Europe', 'Oceania', 'South America', 'North America'])

# Create DataFrame for the input datas
data = {'Beer Serving': beer_input, 
        'Spirit Serving': spirit_input, 
        'Wine Serving': wine_input, 
        'Continent': conti_input}
input_data = pd.DataFrame(data, index=[0])
input_data


if conti_input!=None and conti_input!="Select a continent" and conti_input=="Africa":
    conti_input=0
elif conti_input!=None and conti_input!="Select a continent" and conti_input=="Asia":
    conti_input=1
elif conti_input!=None and conti_input!="Select a continent" and conti_input=="Europe":
    conti_input=2
elif conti_input!=None and conti_input!="Select a continent" and conti_input=="Oceania":
    conti_input=3
elif conti_input!=None and conti_input!="Select a continent" and conti_input=="South America":
    conti_input=4
elif conti_input!=None and conti_input!="Select a continent" and conti_input=="North America":
    conti_input=5



if st.button("PREDICT") and conti_input!=None and beer_input!=None and spirit_input!=None and wine_input!=None:
    import numpy as np
    input_data=np.array([[conti_input,int(beer_input),int(spirit_input),int(wine_input)]])
    input_df = pd.DataFrame(input_data, columns=['continent', 'beer_servings', 'spirit_servings', 'wine_servings'])


    # Scaling
    input_df[['beer_servings', 'spirit_servings', 'wine_servings']] = scaler.transform(input_df[['beer_servings', 'spirit_servings', 'wine_servings']])

    # Extract scaled features as a numpy array
    input_data_scaled = input_df.values

    prediction = model.predict(input_data_scaled)
    prediction = prediction.tolist()
    st.write(f"Predicted total litres of pure alcohol : {prediction[0]}")








