import streamlit as st
import pandas as pd
import math
from typing import List, Dict, Tuple
import copy

from utils import Calculate, ACTIVATION_FUNCTIONS

st.set_page_config(
    page_title="CNNCanvas",
    page_icon=":bar_chart:",
    initial_sidebar_state="expanded",
    layout="wide"
)

# Initialize session state for layers if not exists
if 'layers_config' not in st.session_state:
    st.session_state.layers_config = [{}]

# Ensure at least one layer exists
if not st.session_state.layers_config:
    st.session_state.layers_config = [{}]

# def add_layer(index):
#     st.session_state.layers_config.insert(index + 1, {})
    
# def delete_layer(index):
#     if len(st.session_state.layers_config) > 1:
#         st.session_state.layers_config.pop(index)

def add_layer(index: int):
    """
    Add a new layer configuration with intelligent copying.
    
    Args:
        index (int): The index after which to insert the new layer
    """
    
    try:
        # If there are existing layers and the index is valid, 
        # create a deep copy of the layer at the given index
        if 0 <= index < len(st.session_state.layers_config):
            # Deep copy the layer configuration
            new_layer_config = copy.deepcopy(st.session_state.layers_config[index])
            
            # Clear some potentially unique identifiers to avoid conflicts
            new_layer_config.pop('Name', None)
        else:
            # If index is out of bounds, create an empty layer
            new_layer_config = {}
        
        # Insert the new layer configuration
        st.session_state.layers_config.insert(index + 1, new_layer_config)
        
        # Optional: Reset layers to force recalculation
        if 'layers' in st.session_state:
            del st.session_state.layers
        
        # Optional: Provide user feedback
        st.toast(f"Added new layer after Layer {index + 1}", icon="✨")
    
    except Exception as e:
        st.error(f"Error adding layer: {e}")
        st.exception(e)

def delete_layer(index: int):
    """
    Delete a layer configuration with safety checks.
    
    Args:
        index (int): The index of the layer to delete
    """
    try:
        # Prevent deletion if only one layer remains
        if len(st.session_state.layers_config) > 1:
            # Remove the layer at the specified index
            deleted_layer = st.session_state.layers_config.pop(index)
            
            # Optional: Reset layers to force recalculation
            if 'layers' in st.session_state:
                del st.session_state.layers
            
            # Provide user feedback
            st.toast(f"Deleted Layer {index + 1}", icon="🗑️")
        else:
            # Prevent deletion of the last layer
            st.warning("Cannot delete the last layer. At least one layer must exist.")
    
    except Exception as e:
        st.error(f"Error deleting layer: {e}")
        st.exception(e)

# Custom CSS for enhanced layout
st.markdown("""
    <style>
        .stButton button {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            padding: 0;
            margin-top: 50px;
        }
        .layer-section {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .layer-header {
            padding: 5px;
            margin-bottom: 10px;
            border-bottom: 1px solid #e0e0e0;
        }
    </style>
""", unsafe_allow_html=True)

def show_activation_params(activation, key_prefix):
    """Helper function to show activation function parameters"""
    params = {}
    if activation in ACTIVATION_FUNCTIONS:
        for param_name, param_config in ACTIVATION_FUNCTIONS[activation]["params"].items():
            params[param_name] = st.number_input(
                f"{param_name}",
                min_value=param_config["min"],
                max_value=param_config["max"],
                value=param_config["default"],
                step=param_config["step"],
                key=f"{key_prefix}_act_{param_name}"
            )
    return params

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
        # Create columns for layout
        col_exp, spacer1, col_buttons1, spacer2, col_buttons2 = st.columns([40, 1, 2, 1, 2])
        
        # Main expander column
        with col_exp:
            # Layer naming and configuration
            layer_name = st.text_input(
                "Layer Name",
                value=f"layer_{i+1}",
                key=f"layer_name_{i}",
                help="Enter a unique name for this layer"
            )
            
            layer_expander = st.expander(f"{layer_name} Configuration", expanded=(i==0))
            
            with layer_expander:
                # Basic layer configuration
                e5, col1, e6, col2, e7, col3, e8 = st.columns([1, 8, 1, 8, 1, 8, 1])
                
                with col1:
                    layer_type = st.selectbox(
                        f"Layer Type",
                        ["Convolution", "MaxPool", "AvgPool"],
                        key=f"layer_type_{i}"
                    )
                
                with col2:
                    kernel_size = st.slider(
                        f"Kernel Size",
                        2, 7, 3,
                        key=f"kernel_{i}"
                    )
                
                with col3:
                    stride = st.slider(
                        f"Stride",
                        1, 3, 1,
                        key=f"stride_{i}"
                    )

                e9, col4, e10, col5, e11, col6, e12 = st.columns([1, 8, 1, 8, 1, 8, 1])
                with col4:
                    # Add activation function selection right below layer type
                    if layer_type == "Convolution":
                        activation_type = st.selectbox(
                            "Activation",
                            list(ACTIVATION_FUNCTIONS.keys()),
                            key=f"activation_{i}",
                            help="Select activation function for this layer"
                        )
                        activation_params = show_activation_params(activation_type, f"layer_{i}")
                
                with col5:
                    if layer_type == "Convolution":
                        out_channels = st.number_input(
                            "Output Channels",
                            1, 512, 64,
                            key=f"channels_{i}"
                        )
                
                with col6:
                    if layer_type == "Convolution":
                        padding = st.slider(
                            f"Padding",
                            0, 3, 1,
                            key=f"padding_{i}"
                        )

                st.markdown("---")
                st.markdown("#### Advance Configuration")

                # Convolution-specific options
                if layer_type == "Convolution":
                    # Bias option
                    use_bias = st.checkbox(
                        "Use Bias",
                        value=True,
                        key=f"bias_{i}",
                        help="Enable or disable bias terms in convolution"
                    )

                    # Regularization section
                    st.markdown("##### Regularization")
                    s1, reg_col1, s2, reg_col2, s3 = st.columns([1, 13, 3, 13, 1])
                    
                    with reg_col1:
                        use_batch_norm = st.checkbox(
                            "Use Batch Normalization",
                            value=True,
                            key=f"batch_norm_{i}"
                        )
                        if use_batch_norm:
                            momentum = st.slider(
                                "BatchNorm Momentum",
                                min_value=0.0,
                                max_value=1.0,
                                value=0.99,
                                step=0.01,
                                key=f"bn_momentum_{i}"
                            )
                    
                    with reg_col2:
                        use_dropout = st.checkbox(
                            "Use Dropout",
                            value=False,
                            key=f"dropout_{i}"
                        )
                        if use_dropout:
                            dropout_rate = st.slider(
                                "Dropout Rate",
                                min_value=0.0,
                                max_value=0.9,
                                value=0.3,
                                step=0.1,
                                key=f"dropout_rate_{i}"
                            )
                            dropout_placement = st.radio(
                                "Dropout Placement",
                                ["Before Conv", "After Conv", "After BatchNorm"],
                                index=2,
                                key=f"dropout_placement_{i}"
                            )

                # Calculate output shape and parameters
                if layer_type == "Convolution":
                    current_shape = Calculate.calculate_conv_output_shape(
                        current_shape, kernel_size, stride, padding
                    )
                    # Calculate parameters including bias if enabled
                    conv_params = Calculate.calculate_parameters(
                        current_channels, out_channels, kernel_size
                    )
                    if use_bias:
                        conv_params += out_channels  # Add bias parameters
                        
                    # Add batch norm parameters if enabled
                    if use_batch_norm:
                        batch_norm_params = 2 * out_channels  # gamma and beta
                        params = conv_params + batch_norm_params
                    else:
                        params = conv_params
                    
                    current_channels = out_channels
                    total_params += params
                else:
                    current_shape = Calculate.calculate_pool_output_shape(
                        current_shape, kernel_size, stride
                    )
                    params = 0

                # Store layer information
                layer_info = {
                    "Name": layer_name,
                    "Type": layer_type,
                    "Output Shape": f"{current_shape}",
                    "Parameters": params,
                    "Kernel": kernel_size,
                    "Stride": stride
                }
                if layer_type == "Convolution":
                    layer_info.update({
                        "Padding": padding,
                        "Bias": use_bias,
                        "Activation": {
                            "Type": activation_type,
                            "Params": activation_params
                        },
                        "BatchNorm": {
                            "Enabled": use_batch_norm,
                            "Momentum": momentum if use_batch_norm else None
                        },
                        "Dropout": {
                            "Enabled": use_dropout,
                            "Rate": dropout_rate if use_dropout else None,
                            "Placement": dropout_placement if use_dropout else None
                        }
                    })
                
                st.session_state.layers.append(layer_info)
                
                st.markdown("---")
                st.markdown("#### Summary")

                # Display layer information dynamically
                s4, details1, s5, details2, s6 = st.columns([1, 13, 3, 13, 1])

                # Column 1: General Information
                with details1:
                    activation_details_line = (f"- **Activation Details:** `{', '.join(f'{k}={v}' for k, v in activation_params.items())}`" 
                            if activation_params else "")
                    st.markdown(f"""
                        - **Output Shape:** `{current_shape}`
                        - **Parameters:** `{params:,}`
                        - **Activation:** `{activation_type}`
                        {activation_details_line}
                    """)

                # Column 2: Additional Details
                with details2:
                    batch_norm_details = (f"- **BatchNorm:** `momentum={momentum}`" if use_batch_norm else "")
                    dropout_details = (f"- **Dropout:** `{dropout_rate}` ({dropout_placement})" if use_dropout else "")
                    st.markdown(f"""
                    - **Bias:** {'Enabled' if use_bias else 'Disabled'}
                    {batch_norm_details}
                    {dropout_details}
                    """)

        # Layer manipulation buttons
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