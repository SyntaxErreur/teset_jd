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
    3. Once you've built your CV, move to the **Questions** section in the sidebar. There, you will see the updated questions based on your CV.
    4. You can now iterate on those questions to build them along with your CV. 

    Make sure to follow this flow:
    - Start with **CV** to build the CV.
    - Then, go to **Questions** to review and refine the questions generated.
    """)
    
if __name__ == "__main__":
    main()
