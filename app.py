import streamlit as st
import pandas as pd
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from agent import run_agent, execute_python, calculate_correlation, inspect_data, SYSTEM_PROMPT
from langchain_core.messages import SystemMessage

# 1. Setup & Configuration
load_dotenv()
st.set_page_config(page_title="Research RA: EDA Agent", layout="wide")

# Initialize the LLM (Using your verified Tier 1 Key logic)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# 2. State Management (The "Memory")
# 1. PERSISTENCE: This is the secret sauce
if "chat_history" not in st.session_state:
    # Initialize with your System Prompt
    st.session_state.chat_history = [SystemMessage(content=SYSTEM_PROMPT)]

if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = []

# 3. Sidebar: File Upload
with st.sidebar:
    st.title("📂 Data Portal")
    uploaded_file = st.file_uploader("Upload your CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview", df.head())
        # Save locally so the agent's python_executor can find it
        df.to_csv("active_data.csv", index=False)

# 4. The Chat Interface
st.title("🤖 Research RA")
st.caption("Autonomous EDA Agent powered by Gemini & Python Executor")

# Display chat history from session state
for message in st.session_state.ui_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. The Input Loop
if prompt := st.chat_input("Ask me to analyze your data..."):
    # Add user message to UI
    st.session_state.ui_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Trigger the Agent Logic
    with st.chat_message("assistant"):
        with st.status("🧠 Thinking...", expanded=True) as status:
            # Here you would call your 'run_agent' function
            # For now, let's simulate the 'Thought' and 'Action'
            # st.write("Checking data structure...")
            final_answer, updated_history = run_agent(prompt, "active_data.csv", st.session_state.chat_history)
            st.session_state.chat_history = updated_history

            if os.path.exists("temp_plot.png"):
                st.image("temp_plot.png", caption="Generated Visualization")
                # Move it to a session folder or delete it to avoid ghosting
                os.rename("temp_plot.png", f"plot_{len(st.session_state.ui_messages)}.png")
            status.update(label="✅ Analysis Complete", state="complete")
            
            # --- AGENT EXECUTION LOGIC GOES HERE ---
            # You would use the 'execute_python' tool we built
            # result = run_agent(prompt, "active_data.csv", st.session_state.chat_history)
            # ---------------------------------------
            
            final_response = st.session_state.chat_history[-1].content
            st.markdown(final_response)
            st.session_state.ui_messages.append({"role": "assistant", "content": final_response})
        
        # st.markdown(response)
        # st.session_state.messages.append({"role": "assistant", "content": response})