from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO, SAC, TD3

from agent import NeuralNetwork
from logger import logger


def convert_sb3_to_pytorch(sb3_model_path: str, output_path: str, agent_type: str = "PPO") -> NeuralNetwork:
    """
    Convert a Stable-Baselines3 model to PyTorch format.
    
    Args:
        sb3_model_path (str): Path to the Stable-Baselines3 model file (without .zip extension).
        output_path (str): Path to save the converted PyTorch model (.pth extension).
        agent_type (str): Type of agent (e.g., "PPO", "SAC", "TD3").
        
    Returns:
        NeuralNetwork: The converted PyTorch neural network.
        
    Raises:
        ValueError: If an unsupported agent type is provided.
    """
    
    logger.info(f"Converting {agent_type} model from {sb3_model_path}")
    
    # Load the Stable-Baselines3 model
    if agent_type == "PPO":
        model = PPO.load(sb3_model_path)
    elif agent_type == "SAC":
        model = SAC.load(sb3_model_path)
    elif agent_type == "TD3":
        model = TD3.load(sb3_model_path)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Get the policy network
    policy = model.policy
    
    # Create a new PyTorch network with the same architecture
    input_size = policy.mlp_extractor.policy_net[0].in_features
    hidden_size = policy.mlp_extractor.policy_net[0].out_features
    
    logger.info(f"Input size: {input_size}")
    logger.info(f"Hidden size: {hidden_size}")
    
    # Create the neural network
    pytorch_net = NeuralNetwork(input_size, hidden_size, 8)
    
    # Copy weights from the policy network
    # Extract features
    pytorch_net.fc1.weight.data = policy.mlp_extractor.policy_net[0].weight.data.clone()
    pytorch_net.fc1.bias.data = policy.mlp_extractor.policy_net[0].bias.data.clone()
    
    pytorch_net.fc2.weight.data = policy.mlp_extractor.policy_net[2].weight.data.clone()
    pytorch_net.fc2.bias.data = policy.mlp_extractor.policy_net[2].bias.data.clone()
    
    # Action head
    pytorch_net.fc3.weight.data = policy.action_net.weight.data.clone()
    pytorch_net.fc3.bias.data = policy.action_net.bias.data.clone()
    
    # Save the PyTorch model
    torch.save(pytorch_net.state_dict(), output_path)
    logger.info(f"Model converted and saved to {output_path}")
    
    return pytorch_net


def test_converted_model(model_path: str, test_input: np.ndarray, hidden_size: int = 256) -> torch.Tensor:
    """
    Test the converted model with a sample input.
    
    Args:
        model_path (str): Path to the converted PyTorch model.
        test_input (np.ndarray): A sample input observation.
        hidden_size (int): The hidden size used in the NeuralNetwork.
        
    Returns:
        torch.Tensor: The output of the model for the test input.
    """
    
    logger.info("Testing converted model...")
    
    # Load the model
    model = NeuralNetwork(test_input.shape[0], hidden_size, 8)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Test with sample input
    with torch.no_grad():
        test_tensor = torch.FloatTensor(test_input).unsqueeze(0)
        output = model(test_tensor)
        
    logger.info(f"Input shape: {test_input.shape}")
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Sample output: {output[0].numpy()}")
    
    return output


if __name__ == "__main__":
    # Example usage
    sb3_model_path = "models/PPO/PPO_final"  # Path to your trained model
    output_path = "models/PPO/PPO_converted.pth"  # Output PyTorch model
    
    # Check if the model exists
    if Path(sb3_model_path + ".zip").exists():
        try:
            # Convert the model
            converted_model = convert_sb3_to_pytorch(sb3_model_path, output_path, "PPO")
            
            # Test with a sample input (you'll need to adjust the size based on your obs)
            sample_input = np.random.randn(107)  # Adjust size based on your observation space
            # Pass the hidden_size from the converted model
            test_converted_model(output_path, sample_input, converted_model.fc1.out_features)
            
            logger.info("Model conversion completed successfully!")
            
        except Exception as e:
            logger.error(f"Error converting model: {e}")
            logger.exception(e)
    else:
        logger.info(f"Model not found at {sb3_model_path}")
        logger.info("Please train a model first using train.py")
