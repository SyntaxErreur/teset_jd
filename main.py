import os
import re
import json
from openai import OpenAI
import streamlit as st
from typing import Dict, List, Any, Union
from dotenv import load_dotenv
load_dotenv()


import streamlit as st

def main():
    st.title("🎯 Job Description Parameter Extractor")
    
    # Add a sidebar with two options: CV and Questions
    st.sidebar.header("Navigation")
    option = st.sidebar.radio("Choose an option", ("CV", "Questions"))
    
    if option == "CV":
        st.header("Build Your CV")
        st.write("First, go to the CV section to build your CV. Paste the job description there and extract the necessary parameters.")
        st.text_area("Paste Job Description")
        # Add any CV-building functionality here
        
    elif option == "Questions":
        st.header("Generated Questions")
        st.write("After you've built your CV, go to the Questions section to see the updated questions.")
        st.write("Here you can iterate on the questions and refine them for your needs.")
        # Add the functionality for questions here

if __name__ == "__main__":
    main()
