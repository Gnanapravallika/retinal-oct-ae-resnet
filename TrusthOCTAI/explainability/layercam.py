"""
TrusthOCTAI LayerCAM Saliency Map Generator
Generates pixel-precise Class Activation Maps for Layer 3 (x3) and Layer 4 (x4) features.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2

class LayerCAM:
    """
    LayerCAM generator for visual explainability of retinal lesions.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()
        
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        
        weights = np.maximum(gradients, 0)
        cam = np.sum(weights * activations, axis=0)
        cam = np.maximum(cam, 0)
        
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
            
        cam = cv2.resize(cam, (224, 224))
        return cam
