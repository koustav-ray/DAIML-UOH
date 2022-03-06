import streamlit as st
from classify_quote import predict_proba

# Set the page title and icon
st.set_page_config(page_title="Homesite Quote Conversion",page_icon="🏠",layout="wide")

# Set the title of the page
st.title("Home Insurance Quote Conversion")

# Set the text of the page
st.text("""This page is used to check whether a quotation would lead to successful quote conversion or not with expected percentage for Homesite Insurance Co.
Please enter the Quotation ID in the text box below to know the percentage(%) chance at which this Quote would be succesfully converted to purchase.""")	

# Set the header of the page
st.header("Enter Quotation ID")

# Create a form with name "Quote Conversion"
with st.form("Quote Conversion"):
    query = st.text_input("Enter Quote Number here") # Create a text input with name "Enter Quote Number here"
    if st.form_submit_button("Submit"): # if the user clicks the submit button
        is_success = predict_proba(query) # Call the predict_proba function and store the result in is_success
        if is_success == -1:
            st.write("Quotation ID not available")
        else:
            st.write("The chance of successful conversion of this quote is {}".format(is_success)) 

