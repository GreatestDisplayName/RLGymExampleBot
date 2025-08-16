import os
import pickle
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from logger import logger # Import logger

class NeuralNetwork(nn.Module):
    """
    A simple feedforward neural network for the agent's policy.
    """
    def __init__(self, input_size: int, hidden_size: int = 256, output_size: int = 8,
                 dropout_rate: float = 0.05, use_layer_norm: bool = False):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
        if self.use_layer_norm:
            self.ln1 = nn.LayerNorm(hidden_size)
            self.ln2 = nn.LayerNorm(hidden_size // 2)
        else:
            self.bn1 = nn.BatchNorm1d(hidden_size)
            self.bn2 = nn.BatchNorm1d(hidden_size // 2)
            
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor (observation).
            
        Returns:
            torch.Tensor: Output tensor (actions).
        """
        if self.use_layer_norm:
            x = F.leaky_relu(self.ln1(self.fc1(x)), 0.1)
            x = self.dropout(x)
            x = F.leaky_relu(self.ln2(self.fc2(x)), 0.1)
        else:
            x = F.leaky_relu(self.bn1(self.fc1(x)), 0.1)
            x = self.dropout(x)
            x = F.leaky_relu(self.bn2(self.fc2(x)), 0.1)
            
        x = self.fc3(x)  # Output raw logits
        return x


class Agent:
    """
    RLGym agent that uses a neural network to predict actions.
    """
    def __init__(self, input_size: int = 107, hidden_size: int = 128,
                 output_size: int = 8, model_path: Optional[str] = None,
                 dropout_rate: float = 0.05, use_layer_norm: bool = False):
        """
        Initializes the agent.
        
        Args:
            input_size (int): The size of the observation space.
            hidden_size (int): The number of neurons in the hidden layers.
            output_size (int): The size of the action space.
            model_path (Optional[str]): Path to a pre-trained model to load.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.actor = NeuralNetwork(
            self.input_size,
            self.hidden_size,
            self.output_size,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm
        ).to(self.device)
        
        # Load pre-trained model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.info("No pre-trained model found, using random weights")
    
    def load_model(self, model_path: str):
        """
        Loads a trained model from the specified path.
        Supports PyTorch .pth and legacy pickle formats.
        
        Args:
            model_path (str): The path to the model file.
        """
        try:
            if model_path.endswith('.pth'):
                # PyTorch model
                self.actor.load_state_dict(torch.load(model_path, map_location=self.device))
            else:
                # Pickle model (legacy format)
                with open(model_path, 'rb') as file:
                    model_data = pickle.load(file)
                    if isinstance(model_data, dict) and 'state_dict' in model_data:
                        self.actor.load_state_dict(model_data['state_dict'])
                    else:
                        self.actor.load_state_dict(model_data)
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}", exc_info=True)
            raise # Re-raise the exception after logging
    
    def save_model(self, model_path: str):
        """
        Saves the trained model to the specified path.
        
        Args:
            model_path (str): The path to save the model file.
        """
        try:
            torch.save(self.actor.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model to {model_path}: {e}", exc_info=True)
            raise # Re-raise the exception after logging
    
    def act(self, state: np.ndarray) -> np.ndarray:
        """
        Gets an action from the neural network based on the current state.
        
        Args:
            state (np.ndarray): The current observation from the environment.
            
        Returns:
            np.ndarray: The predicted action.
        """
        # Convert state to tensor
        # Ensure state is 2D (batch_size, input_size)
        if state.ndim == 1:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        else:
            state_tensor = torch.FloatTensor(state).to(self.device)
        
        # Get action from network
        with torch.no_grad():
            action = self.actor(state_tensor)
            action = torch.tanh(action) # Apply tanh to get actions in [-1, 1] range
        
        # Convert to numpy array and return
        return action.cpu().numpy().squeeze() # Use squeeze to handle both single and batch observations
    
    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """
        Gets action probabilities from the neural network (useful for training).
        
        Args:
            state (np.ndarray): The current observation from the environment.
            
        Returns:
            np.ndarray: The action probabilities.
        """
        if state.ndim == 1:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        else:
            state_tensor = torch.FloatTensor(state).to(self.device)
            
        with torch.no_grad():
            action_logits = self.actor(state_tensor)
            # Convert to probabilities using softmax
            probs = F.softmax(action_logits, dim=-1)
        return probs.cpu().numpy().squeeze()
