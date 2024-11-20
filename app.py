import streamlit as st
import pandas as pd
import math
from typing import List, Dict, Tuple

from utils import Calculate

st.set_page_config(
    page_title="CNNCanvas",
    page_icon=":bar_chart:",
    initial_sidebar_state="expanded",
)

def show_canvas():
    st.title("CNNCanvas")
    st.markdown("---")

     # Step 1: Take Input layer configurations
    st.subheader("Input Shape")
    c1, c2, c3 = st.columns(3)
    with c1:
        input_height = st.number_input("Input Height", 32, 512, 224)
    with c2:
        input_width = st.number_input("Input Width", 32, 512, 224)
    with c3:
        input_channels = st.number_input("Input Channels", 1, 3, 3)

    st.markdown("---")

    # Layer configuration
    st.subheader("Layer Configuration")
    layers = []
    
    num_layers = st.number_input("Number of layers", 1, 10, 3)
    
    total_params = 0
    current_shape = (input_height, input_width)
    current_channels = input_channels
    
    for i in range(num_layers):
        st.markdown(f"#### Layer {i+1}")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            layer_type = st.selectbox(
                f"Layer {i+1} Type",
                ["Convolution", "MaxPool", "AvgPool"],
                key=f"layer_type_{i}"
            )
        with col2:
            kernel_size = st.slider(
                f"Kernel Size",
                2, 7, 3,
                key=f"kernel_{i}"
            )
            if layer_type == "Convolution":
                out_channels = st.number_input(
                    "Output Channels",
                    1, 512, 64,
                    key=f"channels_{i}"
                )
        with col3:
            stride = st.slider(
                f"Stride",
                1, 3, 1,
                key=f"stride_{i}"
            )
            if layer_type == "Convolution":
                padding = st.slider(
                    f"Padding",
                    0, 3, 1,
                    key=f"padding_{i}"
                )

        # Calculate output shape and parameters
        if layer_type == "Convolution":
            current_shape = Calculate.calculate_conv_output_shape(
                current_shape, kernel_size, stride, padding
            )
            params = Calculate.calculate_parameters(
                current_channels, out_channels, kernel_size
            )
            current_channels = out_channels
            total_params += params
        else:
            current_shape = Calculate.calculate_pool_output_shape(
                current_shape, kernel_size, stride
            )
            params = 0

        # Store layer information
        layer_info = {
            "Layer": f"Layer {i+1}",
            "Type": layer_type,
            "Output Shape": f"{current_shape}",
            "Parameters": params,
            "Kernel": kernel_size,
            "Stride": stride
        }
        if layer_type == "Convolution":
            layer_info["Padding"] = padding
        
        layers.append(layer_info)
        
        # Display layer information
        st.info(f"""
        Output Shape: {current_shape}
        Parameters: {params:,}
        """)

    # Calculate total model size
    total_model_size = Calculate.calculate_model_size(total_params)
    
    # Summary
    st.header("Architecture Summary")
    df = pd.DataFrame(layers)
    st.dataframe(df)
    
    st.success(f"""
    Total Parameters: {total_params:,}
    Final Output Shape: {current_shape}
    Total Model Size: {total_model_size:.2f} MB
    """)

if __name__ == "__main__":
    show_canvas()