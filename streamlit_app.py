import streamlit as st
from PIL import Image
import base64
from io import BytesIO

# Message de bienvenue
st.write('Welcome to MediSense Platform')

# Charger l'image
img_path = "https://github.com/inesbatigue/LLMProject/blob/main/top.jpg"  # Assurez-vous que le chemin est correct
image = Image.open(img_path)

# Convertir l'image en base64 pour l'utiliser dans du CSS
buffered = BytesIO()
image.save(buffered, format="JPEG")
img_str = base64.b64encode(buffered.getvalue()).decode()

# Code CSS pour définir l'image en arrière-plan avec base64
page_bg_img = f'''
<style>
.stApp {{
  background-image: url("data:image/jpeg;base64,{img_str}");
  background-size: cover;
  background-repeat: no-repeat;
  background-attachment: fixed;
}}
</style>
'''

# Appliquer le CSS à l'application Streamlit
st.markdown(page_bg_img, unsafe_allow_html=True)

# Exemple de contenu de votre application
st.title("Mon Application Streamlit")
st.write("Contenu de votre application avec une image en arrière-plan.")
