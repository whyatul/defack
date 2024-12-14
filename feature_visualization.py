import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import streamlit as st
from train_xception import DeepfakeXception
from train_efficientnet import DeepfakeEfficientNet
from train_swin import DeepfakeSwin
from train_cross_attention import DeepfakeCrossAttention
from train_two_stream import DeepfakeTwoStream
from train_cnn_transformer import DeepfakeCNNTransformer

class FeatureExtractor:
    def __init__(self, model, model_type):
        self.model = model
        self.model_type = model_type
        self.features = {}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        try:
            if self.model_type == "xception":
                # Register more hooks for Xception
                layers = ['conv1', 'conv2', 'conv3', 'conv4', 'block1', 'block2', 'block3', 
                         'block4', 'block5', 'block6', 'block7', 'block8', 'block9', 'block10', 
                         'block11', 'block12']
                for layer_name in layers:
                    if hasattr(self.model.backbone, layer_name):
                        layer = getattr(self.model.backbone, layer_name)
                        if layer_name.startswith('block'):
                            self.hooks.append(layer.rep.register_forward_hook(
                                lambda m, i, o, name=layer_name: self._hook_fn(f'{name}_rep', o)
                            ))
                        else:
                            self.hooks.append(layer.register_forward_hook(
                                lambda m, i, o, name=layer_name: self._hook_fn(name, o)
                            ))
                
            elif self.model_type == "efficientnet":
                # Register hooks for more EfficientNet blocks
                for i, block in enumerate(self.model.backbone._blocks):
                    if i % 3 == 0:  # Sample every third block to avoid too many visualizations
                        self.hooks.append(block.register_forward_hook(
                            lambda m, i, o, idx=i: self._hook_fn(f'block_{idx}', o)
                        ))
                
            elif self.model_type == "swin":
                # Register hooks for all Swin layers
                for i, layer in enumerate(self.model.backbone.layers):
                    self.hooks.append(layer.register_forward_hook(
                        lambda m, i, o, idx=i: self._hook_fn(f'swin_layer_{idx}', o)
                    ))
                    if hasattr(layer, 'blocks'):
                        for j, block in enumerate(layer.blocks):
                            self.hooks.append(block.register_forward_hook(
                                lambda m, i, o, idx1=i, idx2=j: self._hook_fn(f'swin_block_{idx1}_{idx2}', o)
                            ))
                
            elif self.model_type == "cross_attention":
                # Register hooks for more attention layers
                self.hooks.append(self.model.backbone.register_forward_hook(
                    lambda m, i, o: self._hook_fn('backbone_features', o)
                ))
                self.hooks.append(self.model.cross_attention1.register_forward_hook(
                    lambda m, i, o: self._hook_fn('cross_attn1', o)
                ))
                self.hooks.append(self.model.cross_attention2.register_forward_hook(
                    lambda m, i, o: self._hook_fn('cross_attn2', o)
                ))
                
            elif self.model_type == "two_stream":
                # Register hooks for both streams
                self.hooks.append(self.model.rgb_stream.register_forward_hook(
                    lambda m, i, o: self._hook_fn('rgb_stream', o)
                ))
                self.hooks.append(self.model.freq_stream.register_forward_hook(
                    lambda m, i, o: self._hook_fn('freq_stream', o)
                ))
                if hasattr(self.model, 'fusion'):
                    self.hooks.append(self.model.fusion.register_forward_hook(
                        lambda m, i, o: self._hook_fn('fusion', o)
                    ))
                
            elif self.model_type == "cnn_transformer":
                # Register hooks for CNN and transformer layers
                self.hooks.append(self.model.backbone.register_forward_hook(
                    lambda m, i, o: self._hook_fn('cnn_features', o)
                ))
                for i, layer in enumerate(self.model.transformer_layers):
                    self.hooks.append(layer.register_forward_hook(
                        lambda m, i, o, idx=i: self._hook_fn(f'transformer_layer_{idx}', o)
                    ))
                    self.hooks.append(layer.attn.register_forward_hook(
                        lambda m, i, o, idx=i: self._hook_fn(f'attention_layer_{idx}', o)
                    ))
        except Exception as e:
            logger.error(f"Error registering hooks for {self.model_type}: {str(e)}")
            raise
    
    def _hook_fn(self, layer_name, output):
        self.features[layer_name] = output.detach()
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_features(self):
        return self.features

class FeatureVisualizer:
    @staticmethod
    def normalize_feature_map(feature_map):
        if len(feature_map.shape) == 4:
            feature_map = feature_map.squeeze(0)
        if len(feature_map.shape) == 3:
            feature_map = feature_map.mean(0)
        
        feature_map = feature_map.cpu().numpy()
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        return feature_map
    
    @staticmethod
    def create_heatmap(feature_map, size=(224, 224)):
        # Resize feature map
        feature_map_resized = cv2.resize(feature_map, size)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * feature_map_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return heatmap
    
    @staticmethod
    def overlay_heatmap(image, heatmap, alpha=0.4):
        # Convert PIL Image to numpy array if necessary
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Resize image to match heatmap
        image = cv2.resize(image, (heatmap.shape[1], heatmap.shape[0]))
        
        # Overlay
        overlayed = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        return overlayed
    
    @staticmethod
    def visualize_attention(attention_map, size=(224, 224)):
        # Process attention map
        if torch.is_tensor(attention_map):
            attention_map = attention_map.cpu().numpy()
        
        if len(attention_map.shape) > 2:
            attention_map = attention_map.mean(axis=tuple(range(len(attention_map.shape)-2)))
        
        # Normalize and resize
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        attention_map = cv2.resize(attention_map, size)
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return heatmap

def get_feature_maps(model, model_type, image_tensor):
    """Extract and visualize feature maps for a given model and image"""
    # Initialize feature extractor
    extractor = FeatureExtractor(model, model_type)
    visualizer = FeatureVisualizer()
    
    # Forward pass
    with torch.no_grad():
        _ = model(image_tensor)
    
    # Get features
    features = extractor.get_features()
    
    # Process features based on model type
    visualizations = {}
    
    if model_type == "xception":
        for layer_name, feature in features.items():
            feature_map = visualizer.normalize_feature_map(feature)
            heatmap = visualizer.create_heatmap(feature_map)
            visualizations[f"Xception {layer_name}"] = heatmap
            
    elif model_type == "efficientnet":
        for layer_name, feature in features.items():
            feature_map = visualizer.normalize_feature_map(feature)
            heatmap = visualizer.create_heatmap(feature_map)
            visualizations[f"EfficientNet {layer_name}"] = heatmap
            
    elif model_type == "swin":
        for layer_name, feature in features.items():
            if "layer" in layer_name:
                feature_map = visualizer.normalize_feature_map(feature)
                heatmap = visualizer.create_heatmap(feature_map)
                visualizations[f"Swin {layer_name}"] = heatmap
                
    elif model_type == "cross_attention":
        for layer_name, feature in features.items():
            attn_map = visualizer.visualize_attention(feature)
            visualizations[f"Cross Attention {layer_name}"] = attn_map
            
    elif model_type == "two_stream":
        for layer_name, feature in features.items():
            feature_map = visualizer.normalize_feature_map(feature)
            heatmap = visualizer.create_heatmap(feature_map)
            visualizations[f"Two Stream {layer_name}"] = heatmap
            
    elif model_type == "cnn_transformer":
        for layer_name, feature in features.items():
            if "cnn" in layer_name:
                feature_map = visualizer.normalize_feature_map(feature)
                heatmap = visualizer.create_heatmap(feature_map)
            else:
                heatmap = visualizer.visualize_attention(feature)
            visualizations[f"CNN-Transformer {layer_name}"] = heatmap
    
    # Remove hooks
    extractor.remove_hooks()
    
    return visualizations

def display_feature_maps(image, visualizations):
    """Display feature maps in an organized layout"""
    st.write("### Model Feature Visualizations")
    
    # Create tabs for different visualization types
    tab1, tab2 = st.tabs(["Grid View", "Detailed View"])
    
    with tab1:
        # Grid view with more columns
        num_cols = 4  # Increased from 2 to 4
        num_rows = (len(visualizations) + num_cols - 1) // num_cols
        
        # Display original image in its own row
        st.image(image, caption="Original Image", use_container_width=True)
        
        # Display feature maps in grid
        for i in range(0, len(visualizations), num_cols):
            cols = st.columns(num_cols)
            for j in range(num_cols):
                idx = i + j
                if idx < len(visualizations):
                    title, heatmap = list(visualizations.items())[idx]
                    with cols[j]:
                        st.image(heatmap, caption=title, use_container_width=True)
    
    with tab2:
        # Detailed view with larger images and explanations
        st.image(image, caption="Original Image", use_container_width=True)
        
        for title, heatmap in visualizations.items():
            st.write(f"#### {title}")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(heatmap, use_container_width=True)
            
            with col2:
                # Add explanations based on layer type
                if "conv" in title.lower():
                    st.write("Convolutional layer feature map showing detected patterns and edges.")
                elif "attention" in title.lower():
                    st.write("Attention map highlighting regions the model focuses on for decision making.")
                elif "block" in title.lower():
                    st.write("Intermediate block features showing hierarchical pattern recognition.")
                elif "stream" in title.lower():
                    st.write("Stream-specific features from the two-stream architecture.")
                elif "transformer" in title.lower():
                    st.write("Transformer layer representations showing global context understanding.")
                
                # Add overlay option
                if st.button(f"Show Overlay for {title}"):
                    overlay = FeatureVisualizer.overlay_heatmap(image, heatmap)
                    st.image(overlay, caption="Overlay with original image", use_container_width=True)