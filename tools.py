import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

def execute_python(code: str, file_path: str):
    """Executes Python code for data analysis and returns the output string."""
    plt.switch_backend('Agg') # Essential for headless M2 execution
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    
    try:
        df = pd.read_csv(file_path)
        # Shared environment for all agents
        local_vars = {"pd": pd, "plt": plt, "sns": sns, "df": df}
        exec(code, {}, local_vars)
        
        # If the code generated a plot, save it
        if plt.get_fignums():
            plt.savefig("temp_plot.png")
            plt.close()
            
        sys.stdout = old_stdout
        return redirected_output.getvalue() or "Execution successful (no text output)."
    except Exception as e:
        sys.stdout = old_stdout
        return f"❌ Python Error: {e}"