import os
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

def run_inspector(file_path, user_query):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
    df = pd.read_csv(file_path)
    
    # Create a concise summary for the LLM
    df_info = f"""
    Columns: {list(df.columns)}
    Data Types: {df.dtypes.to_dict()}
    Missing Values: {df.isnull().sum().to_dict()}
    Head: {df.head(2).to_json()}
    """
    
    system_prompt = SystemMessage(content="""
    You are the 'Inspector Agent'. Your job is to describe the data's health and schema.
    Report: 1. Column names/types. 2. Any missing data issues. 3. Which columns are relevant to the user's query.
    Keep it professional and factual.
    """)
    
    prompt = f"User Question: {user_query}\n\nData Info:\n{df_info}"
    response = llm.invoke([system_prompt, HumanMessage(content=prompt)])
    return response.content