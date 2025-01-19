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
    
    # Add guidance text
    st.write("""
    **Welcome to the Job Description Parameter Extractor!**

    Here's how to use this tool:
    
    1. First, go to the **CV** section using the sidebar on the left. In the CV section, you will paste your job description.
    2. The system will process the job description and generate relevant parameters for your CV.
    3. At the **end of the CV** section, you will see the questions that are generated based on the job description.
    4. Once you've built your CV and reviewed the generated questions at the end, move to the **Questions** section in the sidebar. 
    5. There, you will see the updated questions, and you can iterate and refine them further.

    Make sure to follow this flow:
    - Start with **CV** to build the CV and review the generated questions.
    - Then, move to **Questions** to iterate on the questions and refine them.
    """)

if __name__ == "__main__":
    main()
