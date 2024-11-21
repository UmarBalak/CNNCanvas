import math

class Calculate:
    @staticmethod
    def calculate_conv_output_shape(input_shape: tuple, kernel_size: int, stride: int, padding: int) -> tuple:
        """Calculate output shape after convolution"""
        h_in, w_in = input_shape
        h_out = math.ceil(((h_in + (2 * padding) - kernel_size) / stride) + 1)
        w_out = math.ceil(((w_in + (2 * padding) - kernel_size) / stride) + 1)
        return (h_out, w_out)

    @staticmethod
    def calculate_pool_output_shape(input_shape: tuple, kernel_size: int, stride: int) -> tuple:
        """Calculate output shape after pooling"""
        h_in, w_in = input_shape
        h_out = math.ceil((h_in - kernel_size) / stride + 1)
        w_out = math.ceil((w_in - kernel_size) / stride + 1)
        return (h_out, w_out)

    @staticmethod
    def calculate_parameters(in_channels: int, out_channels: int, kernel_size: int, bias: bool = True) -> int:
        """Calculate number of parameters in a conv layer"""
        params = in_channels * out_channels * pow(kernel_size, 2)
        if bias:
            params += out_channels
        return params

    @staticmethod
    def calculate_model_size(total_params: int) -> float:
        """Calculate model size in MB based on total parameters."""
        return (total_params * 4) / pow(1024, 2)

# Define activation functions and their parameters
ACTIVATION_FUNCTIONS = {
    "ReLU": {"params": {}},
    "LeakyReLU": {
        "params": {
            "negative_slope": {
                "type": "float",
                "min": 0.0,
                "max": 0.5,
                "default": 0.01,
                "step": 0.01
            }
        }
    },
    "PReLU": {"params": {}},
    "ELU": {
        "params": {
            "alpha": {
                "type": "float",
                "min": 0.1,
                "max": 2.0,
                "default": 1.0,
                "step": 0.1
            }
        }
    },
    "SELU": {"params": {}},
    "GELU": {"params": {}},
    "Sigmoid": {"params": {}},
    "Tanh": {"params": {}},
    "None": {"params": {}}
}
