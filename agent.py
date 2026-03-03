import os
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
from typing import Annotated
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
import google.generativeai as genai
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import sys
load_dotenv(override=True)

MY_KEY = os.environ.get("GOOGLE_API_KEY")
print(MY_KEY)

if MY_KEY:
    print("Key loaded successfully!")
else:
    raise ValueError("GOOGLE_API_KEY not found. Add it to your .env file.")

genai.configure(api_key=MY_KEY)

SYSTEM_PROMPT = """
You are a Senior Data Scientist Agent. Your goal is to help the user perform EDA on a CSV file.

### OPERATING RULES:
1. THOUGHT: Before every action, write a "Thought:" explaining why you are taking that step.
2. ACTION: You must call a tool to interact with the data. 
3. OBSERVATION: After a tool runs, analyze the output.
4. If you encounter an error, explain it and try a different approach.

### INITIAL MANDATE:
Your very first action MUST be to use the 'inspect_data' tool to understand the columns and data types.
ALWAYS write your 'THOUGHT' process in plain text before calling any tool. 
NEVER call a tool without writing your 'THOUGHT' process first.
"""

def inspect_data(file_path: str):
    """
    Loads the first 5 rows and the schema of a CSV file.
    Use this first to understand the structure of the data.
    """
    try:
        df = pd.read_csv(file_path)
        return {
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "head": df.head().to_dict()
        }
    except Exception as e:
        return f"Error loading file: {e}"

def calculate_correlation(file_path: str, col1: str, col2: str):
    """
    Calculates the Pearson correlation coefficient between two numeric columns.
    Input: file_path (str), col1 (str), col2 (str).
    """
    try:
        df = pd.read_csv(file_path)
        # Specialist check: Are they numeric?
        if not pd.api.types.is_numeric_dtype(df[col1]) or not pd.api.types.is_numeric_dtype(df[col2]):
            return f"Error: Both {col1} and {col2} must be numeric to calculate correlation."
        
        correlation = df[col1].corr(df[col2])
        return f"The Pearson correlation between {col1} and {col2} is {correlation:.2f}"
    except Exception as e:
        return f"Error calculating correlation: {e}"


def execute_python(code: str, file_path: str):
    """
    Executes Python code for data analysis. 
    The dataframe is available as 'df'. 
    Example: print(df.describe()) or sns.histplot(df['column'])
    """
    # 1. Headless Matplotlib to save M2 RAM
    plt.switch_backend('Agg')
    
    # 2. Setup the capture of output (stdout)
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    
    try:
        # Load the dataframe so it's ready for the agent's code
        df = pd.read_csv(file_path)
        
        # Prepare the environment with common libraries
        local_vars = {
            "pd": pd, "plt": plt, "sns": sns, "df": df
        }
        
        # 3. Execute the code
        exec(code, {}, local_vars)
        
        # Restore stdout and get the results
        sys.stdout = old_stdout
        result = redirected_output.getvalue()
        
        return result if result else "Code executed successfully (no printed output)."
        
    except Exception as e:
        sys.stdout = old_stdout
        return f"❌ Python Error: {e}"


# 1. Initialize the Brain (M2-friendly API)
# Make sure you have: export GOOGLE_API_KEY='your-key-here'
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=MY_KEY)

# 2. Bind the Tool to the LLM
llm_with_tools = llm.bind_tools([inspect_data, calculate_correlation, execute_python])
chat_history = [SystemMessage(content=SYSTEM_PROMPT)]

def run_agent(user_query, file_path):
    chat_history.append(HumanMessage(content=f"{user_query} (Data: {file_path})"))
    # Initialize the State (The Memory)
    # messages = [
    #     ("system", SYSTEM_PROMPT),
    #     ("user", f"{user_query}. The file is at: {file_path}")
    # ]

    print(f"🚀 Starting EDA Task: {user_query}")

    # The Agentic Loop
    for i in range(5):  # Safety limit of 5 turns
        response = llm_with_tools.invoke(chat_history)
        chat_history.append(response)
        has_tool_calls = bool(response.tool_calls)
        content = ""

        if response.content:
            # Check if it's a list (Gemini 2.x/3.x style)
            if isinstance(response.content, list):
                content = "".join([b['text'] for b in response.content if 'text' in b])
            else:
                content = response.content
            
            # if thought.strip():
            #     print(f"\n🤔 Agent Thought: {thought}")
        if has_tool_calls and content.strip():
            print(f"\n🤔 Agent Thought: {content}")

        if has_tool_calls:
            for tool_call in response.tool_calls:
                print(f"🛠️  Executing: {tool_call['name']}...")
                # ... (your tool execution logic) ...
        else:
            # 4. If no tools, this is the 'Final Answer'
            if content.strip():
                print(f"\n✅ Agent: {content}")
            break 

        # # Check if the LLM wants to call a tool
        # if not response.tool_calls:
        #     # print(f"✅ Final Answer: {response.content}")
        #     if isinstance(response.content, list):
        #         # LangChain sometimes returns a list of content blocks
        #         clean_text = "".join([block['text'] for block in response.content if 'text' in block])
        #     else:
        #         clean_text = response.content
        #     print(f"\n✅ Agent: {clean_text}")
        #     break

        for tool_call in response.tool_calls:
            print(f"🛠️  Action: Running {tool_call['name']}...")
            
            # Execute the tool (using our Specialist tools from earlier)
            if tool_call['name'] == "inspect_data":
                result = inspect_data(**tool_call['args'])
            elif tool_call['name'] == "calculate_correlation":
                result = calculate_correlation(**tool_call['args'])
            elif tool_call['name'] == "execute_python":
                # Pass the code AND the file_path
                result = execute_python(
                    code=tool_call['args']['code'], 
                    file_path=file_path # Pass the CSV path here
        )

            # Logic to handle "Big Data" Observations
            if len(str(result)) > 10000: # Arbitrary character limit for 8GB RAM safety
                 print(f"⚠️  Observation is too large ({len(str(result))} chars).")
                 if isinstance(result, dict) and "head" in result:
                    result["head"] = "TRUNCATED FOR BREVITY"
                    result["note"] = "The sample rows were removed to save memory, but the schema is above."
                #  user_guidance = input("The data is massive. Which columns/patterns should I focus on? ")
            
            # Feed the result back into the State
            chat_history.append(ToolMessage(
                tool_call_id=tool_call['id'],
                content=str(result)))

if __name__ == "__main__":
    path = input("📂 Enter CSV path: ")
    
    print("👋 I'm your Data Assistant. Type 'exit' to quit.")
    while True:
        user_input = input("\n👤 You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        run_agent(user_input, path)