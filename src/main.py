import os
import sys 

# Add the tools directory to the system path
tools_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(tools_dir)

# Import the Streamlit interface
from tools.streamlit_interface import StreamlitInterface

# Run the Streamlit interface
if __name__ == "__main__":
    interface = StreamlitInterface()
    interface.run()  # Corrected comment syntax
