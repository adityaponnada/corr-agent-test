import streamlit as st
import os
import time
from inspector_agent import run_inspector
from statistician_agent import run_statistician
from plotter_agent import run_plotter

st.title("🧪 Multi-Agent Research Lab")

# Setup Sidebar & File Upload
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    with open("data.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())

# initialize UI chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "image" in msg:
            st.image(msg["image"])

if prompt := st.chat_input("Start analysis..."):
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Assistant Response (The Relay)
    with st.chat_message("assistant"):
        # STEP 1: INSPECTOR
        with st.status("🕵️ Inspector is checking data...", expanded=False):
            inspect_report = run_inspector("data.csv", prompt)
            st.write(inspect_report)
        
        # STEP 2: STATISTICIAN
        with st.status("🔢 Statistician is calculating...", expanded=False):
            stats_report, raw_code = run_statistician(inspect_report, "data.csv", prompt)
            st.write(stats_report)
            with st.expander("📂 View Statistician's Code"):
                st.code(raw_code, language="python")
            
        # STEP 3: PLOTTER
        with st.status("🎨 Plotter is visualizing...", expanded=True):
            # The plotter agent calls execute_python and saves 'temp_plot.png'
            plot_status, raw_plot_code = run_plotter(stats_report, "data.csv", prompt)
            st.write(plot_status)
            with st.expander("📂 View Plotter's Code"):
                st.code(raw_plot_code, language="python")
        
        # --- THE RENDERING MAGIC ---
        if os.path.exists("temp_plot.png"):
            # Display the plot in the chat bubble
            st.image("temp_plot.png", caption="Analysis Visualization")
            
            # Save it to session state so it persists on rerun
            # (We rename it so the next turn doesn't overwrite the UI history)
            unique_plot_name = f"plot_{int(time.time())}.png"
            os.rename("temp_plot.png", unique_plot_name)
            
            # Final output for history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": stats_report, 
                "image": unique_plot_name
            })
        else:
            st.markdown(stats_report)
            st.session_state.messages.append({"role": "assistant", "content": stats_report})