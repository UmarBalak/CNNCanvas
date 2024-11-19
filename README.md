# CNNCanvas  

**CNNCanvas** is an intuitive tool for building, analyzing, and experimenting with Convolutional Neural Networks (CNNs). It helps users explore how changes in layer parameters affect the network's output shapes and model size. Designed with modularity and interactivity in mind, CNNCanvas is ideal for learning and experimenting with CNN architectures.  

## 🌟 Current Features  

- **Layer-by-Layer Model Creation**:  
  Add layers manually to define custom CNN architectures.  

- **Parameter Exploration**:  
  Experiment with layer parameters like:  
  - Number of filters.  
  - Kernel size.  
  - Strides and padding.  
  - Pooling operations.  

- **Output Shape Analysis**:  
  View the output shape of each layer dynamically.  

- **Model Size Estimation**:  
  Explore how parameters impact the total size of the model.  

- **Interactive Learning**:  
  Understand the effects of architectural decisions step-by-step.  

## ✨ Planned Features  

- **Visualizations**:  
  Visualize feature maps and filter activations for added clarity.  

- **Prebuilt Architectures**:  
  Analyze common CNN models like VGG, ResNet, and Inception.  

- **Export Options**:  
  Export models in TensorFlow/Pytorch formats.  

- **Comparative Analysis**:  
  Compare multiple models on size, speed, and performance metrics.  

## 🚀 Technologies Used  

- **Frameworks/Libraries**:  
  - Python  
  - TensorFlow/Keras 
  - PyTorch 
  - Streamlit  

## 📂 Project Structure  

```plaintext  
CNNCanvas/  
├── layers/              # Core layer definitions  
├── utils/               # Utility functions for analysis  
├── app.py               # Main Streamlit app  
├── requirements.txt     # Dependencies  
└── README.md            # Project documentation  
