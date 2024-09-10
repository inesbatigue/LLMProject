import streamlit as st
st.write('Welcome to MediSense Platform')
# Login page function
def login_page():
    set_png_as_page_bg('top.jpg')  # Set background image
    #display_logo('logo.jpeg')  # Display logo
    st.logo('logo.jpeg', icon_image='logo.jpeg')
    st.title("Create Account")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

            st.session_state.logged_in = True
            st.experimental_user  # Reload the page to move to the model page


# Utility to add background image and logo
def add_background_and_logo(bg_image, logo_image):
    # Background Image
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({bg_image});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
