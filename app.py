import streamlit as st
import math
from typing import List, Dict, Tuple

from utils import Calculate

def show_canvas():
    st.title("CNNCanvas")
    st.markdown("---")

    # Input Image Specification
    st.subheader("Input Shape")
    c1, c2, c3 = st.columns(3)
    with c1:
        input_height = st.number_input("Input Height", 32, 512, 224)
    with c2:
        input_width = st.number_input("Input Width", 32, 512, 224)
    with c3:
        input_channels = st.number_input("Input Channels", 1, 3, 3)

    st.markdown("---")

if __name__ == "__main__":
    show_canvas()