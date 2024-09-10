import streamlit as st
from PIL import Image
import base64
from io import BytesIO
# Définir une URL d'image en ligne pour tester
img_url = "https://img.freepik.com/photos-gratuite/vue-dessus-medecine-fond-bleu_23-2149341569.jpg?t=st=1725964503~exp=1725968103~hmac=f1236079f03d3b3276f4e07615fc8d75d2f6565369a3795513c3c3e1e8518382&w=1060"
# Code CSS avec l'URL de l'image
page_bg_img = f'''
<style>
    .stApp {{
        background: url({img_url});
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}
</style>
'''

# Appliquer le CSS à l'application Streamlit
st.markdown(page_bg_img, unsafe_allow_html=True)

# Exemple de contenu de votre application
st.title("Welcome to MediSense Platform")
st.write("Access the best medical information and answers quickly and easily")
