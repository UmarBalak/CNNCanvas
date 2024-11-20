import streamlit as st
import pandas as pd
import math
from typing import List, Dict, Tuple

from utils import Calculate

st.set_page_config(
    page_title="CNNCanvas",
    page_icon=":bar_chart:",
    initial_sidebar_state="expanded",
    layout="wide"
)

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;} /* Hides the main menu */
    footer {visibility: hidden;} /* Hides the footer */
    header {visibility: hidden;} /* Hides the header */
    .css-1d391kg {visibility: hidden;} /* Hides the status indicator */
    .css-1v3fvcr {visibility: hidden;} /* Hides the Streamlit watermark */
    .css-1v0mbdj {visibility: hidden;} /* Hides the overall container */
    # </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# Initialize session state for layers if not exists
if 'layers_config' not in st.session_state:
    st.session_state.layers_config = [{}]

def add_layer(index):
    st.session_state.layers_config.insert(index + 1, {})
    
def delete_layer(index):
    if len(st.session_state.layers_config) > 1:
        st.session_state.layers_config.pop(index)

# Custom CSS for small circular buttons
st.markdown("""
    <style>
        .stButton button {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            font-size: 5px;
            padding: 0;
            margin-top: 5px;
        }
    </style>
""", unsafe_allow_html=True)


def show_canvas():
    st.title("CNNCanvas")
    st.subheader("Dynamic Convolutional Neural Network Designer")
    st.markdown("---")

    # Sidebar for app navigation
    st.sidebar.title("Configure Your Model")

    # Input Dimensions Section
    st.sidebar.header("Input Dimensions")
    input_height = st.sidebar.number_input("Input Height (H)", 1, 1024, 224)
    input_width = st.sidebar.number_input("Input Width (W)", 1, 1024, 224)
    input_channels = st.sidebar.number_input("Input Channels (C)", 1, 512, 3)

    # Initialize state to store layers
    if "layers" not in st.session_state:
        st.session_state.layers = []

    # Initialize total parameters count
    total_params = 0
    current_shape = (input_height, input_width)
    current_channels = input_channels

    # Dynamic layer management
    for i in range(len(st.session_state.layers_config)):
        # Create three columns: expander, spacer, and buttons
        col_exp, spacer, col_buttons1, col_buttons2 = st.columns([20, 1, 2, 2])
        
        # Main expander in the first column
        with col_exp:
            layer_expander = st.expander(f"Layer {i + 1} Configuration", expanded=(i==0))
            
            with layer_expander:
                e5, col1, e6, col2, e7, col3, e8 = st.columns([1, 4, 1, 4, 1, 4, 1])
                
                with col1:
                    layer_type = st.selectbox(
                        f"Type",
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
                    "Layer": f"Layer {i + 1}",
                    "Type": layer_type,
                    "Output Shape": f"{current_shape}",
                    "Parameters": params,
                    "Kernel": kernel_size,
                    "Stride": stride
                }
                if layer_type == "Convolution":
                    layer_info["Padding"] = padding
                
                st.session_state.layers.append(layer_info)
                
                # Display layer information dynamically
                st.info(f"""
                **Output Shape:** {current_shape}
                **Parameters:** {params:,}
                """)

        # Buttons in the third column, aligned with expander header
        with col_buttons1:
            st.button("Add", key=f"add_{i}", on_click=add_layer, args=(i,), help="Add new layer")
        with col_buttons2:
            if len(st.session_state.layers_config) > 1:
                st.button("Del", key=f"delete_{i}", on_click=delete_layer, args=(i,), help="Delete this layer")


    # Display Total Parameters and Model Size
    st.markdown("---")
    model_size = Calculate.calculate_model_size(total_params)
    st.write(f"**Total Parameters:** {total_params:,}")
    st.write(f"**Estimated Model Size:** {model_size:.2f} MB")

if __name__ == "__main__":
    show_canvas()