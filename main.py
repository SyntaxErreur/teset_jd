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
    st.title("🎯 Job Description Enhancer & Questions Generator")
    
    # Add guidance text
    st.write("""
    **Welcome to the Job Description Enhancer & Questions Generator!**

    Here's how to use this tool:
    
    1. First, go to the **CV** section using the sidebar on the left. In the CV section, you will paste your job description.
    2. The system will enhance the job description and generate relevant parameters to improve it.
    3. At the **end of the CV** section, you will see the personalized questions that are generated based on the job description.
    4. Once you've refined your job description and reviewed the generated questions, move to the **Questions** section in the sidebar. 
    5. There, you can iterate and refine the questions further for your interview preparation.

    Make sure to follow this flow:
    - Start with **CV** to enhance the job description and review the generated questions.
    - Then, move to **Questions** to iterate and refine them for your interview prep.
    """)

if __name__ == "__main__":
    main()
