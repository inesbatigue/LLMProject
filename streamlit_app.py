import streamlit as st
st.write('Welcome to MediSense Platform')

# Code CSS pour définir l'image en arrière-plan
page_bg_img = '''
<style>
.stApp {
  background-image: url("C:\Users\benat\Documents\RAG\top.jpg");
  background-size: cover;
  background-repeat: no-repeat;
  background-attachment: fixed;
}
</style>
'''

# Appliquer le CSS à l'application Streamlit
st.markdown(page_bg_img, unsafe_allow_html=True)

# Exemple de contenu de votre application
st.title("Mon Application Streamlit")
st.write("Contenu de votre application avec une image en arrière-plan.")
