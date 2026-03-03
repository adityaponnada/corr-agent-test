import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import execute_python

load_dotenv()

def run_plotter(stats_report, file_path, user_query):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
    llm_with_tools = llm.bind_tools([execute_python])
    
    prompt = f"""
    Stats to Plot: {stats_report}
    User Goal: {user_query}
    
    Use 'execute_python' to create a plot and save it to 'temp_plot.png'.
    """
    
    response = llm_with_tools.invoke(prompt)
    
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        code_used = tool_call['args']['code']
        execute_python(code_used, file_path)
        return "Plot generated and saved to temp_plot.png.", code_used

    return response.content, ""