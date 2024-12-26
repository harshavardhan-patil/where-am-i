import folium
import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import base64
from loguru import logger
from src.modeling.predict import predict_nn
from geopy.geocoders import Nominatim
import json

#converting static image and setting as website background
def set_jpg_as_page_bg(bg):
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/jpg;base64,{base64.b64encode(open(bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
set_jpg_as_page_bg('static/bg.jpg')

# initialize Nominatim API 
geolocator = Nominatim(user_agent="GetLoc")

def process_image(uploaded_file):
    gps = predict_nn(uploaded_file)
    addr = geolocator.reverse(gps).raw['address']
    str = ""
    for key in addr.keys():
        str+=f"{addr[key]} \n"
    st.subheader("You are at:")
    st.code(str)
    # center on Liberty Bell, add marker
    m = folium.Map(location=gps, zoom_start=16)
    folium.Marker(gps,).add_to(m)

    # call to render Folium map in Streamlit, but don't get any data back
    # from the map (so that it won't rerun the app when the user interacts)
    st_folium(m, width=725, returned_objects=[])

st.title("Where am I?")
st.subheader("Find a location anywhere in the USA with just an image")


uploaded_file = st.file_uploader("Supported file types - JPG, JPEG, PNG")
if uploaded_file is not None:
    process_image(uploaded_file)



enable = st.checkbox("Enable camera")
picture = st.camera_input("Take a photo", disabled=not enable)

if picture:
    st.image(picture)

