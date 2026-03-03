import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import execute_python # Import our shared tool

load_dotenv()

def run_statistician(inspector_report, file_path, user_query):
    # Bind the tool to the model
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
    llm_with_tools = llm.bind_tools([execute_python])
    
    prompt = f"""
    Inspector Report: {inspector_report}
    User Goal: {user_query}
    
    Use the 'execute_python' tool to run descriptive stats or correlations.
    Return the final numbers to the next agent.
    """
    
    # Simple one-turn tool call for this relay
    response = llm_with_tools.invoke(prompt)
    
    # Logic to handle the tool call
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        code_used = tool_call['args']['code']
        result = execute_python(code_used, file_path)
        return f"Stats Result: {result}", code_used

    return response.content, ""