# Complete RLGym Workflow System

## Overview

The Complete Workflow System provides an automated pipeline for the entire RLGym training process: **Load Map → Train → Export Model → Load Model → Play**. This system orchestrates all the individual components into a seamless, end-to-end workflow.

## 🚀 Quick Start

### Using the Batch File (Windows)

**Important Note on Running Batch Files:**
The `.bat` files are designed to be run from the project's root directory (`RLGymExampleBot/`). If you run them from a subdirectory (e.g., `RLGymExampleBot/scripts/`), you might encounter "file not found" errors for Python scripts. Ensure your command prompt or terminal is in the correct directory, or adjust the paths within the `.bat` files accordingly (as was done for `SCRIPT_DIR=..\src`).

These batch files act as wrappers, forwarding any arguments you provide directly to the underlying Python script (`src/complete_workflow.py`).

```batch
# Run with default settings
complete_workflow.bat

# Run with custom agent type (arguments are passed to the Python script)
complete_workflow.bat --algorithm SAC

# Run with custom timesteps
complete_workflow.bat --timesteps 50000

# Run with full automation (note: these are switches to *skip* steps)
complete_workflow.bat --no-convert --no-play
```

### Using Python Directly

You can also run the `complete_workflow.py` script directly from the command line, similar to the batch file, passing arguments as needed.

```bash
# Run with default settings
python src/complete_workflow.py

# Run with custom algorithm
python src/complete_workflow.py --algorithm SAC

# Run with custom timesteps
python src/complete_workflow.py --timesteps 50000
```

Alternatively, you can use the `CompleteWorkflow` class within your Python scripts:

```python
from src.complete_workflow import CompleteWorkflow

# Create workflow instance
workflow = CompleteWorkflow()

# Run complete pipeline
config = {
    "algorithm": "PPO",
    "timesteps": 100000,
    "auto_convert": True,
    "auto_play": True
}
success = workflow.run_complete_workflow(config)
```

## 📋 Workflow Steps

### 1. Load Map
- **Purpose**: Initialize the training environment with specified difficulty
- **Function**: `workflow.load_map(difficulty="medium", max_steps=1000)`
- **Options**: 
  - `difficulty`: "easy", "medium", "hard"
  - `max_steps`: Maximum steps per episode

### 2. Train Agent
- **Purpose**: Train the RL agent using the specified algorithm
- **Function**: `workflow.train_agent(algorithm="PPO", timesteps=100000)`
- **Supported Algorithms**: PPO, SAC, TD3, A2C, DQN
- **Features**:
  - Automatic checkpointing
  - TensorBoard logging
  - Progress monitoring
  - Resume from previous training

### 3. Export Model
- **Purpose**: Convert Stable-Baselines3 model to PyTorch format
- **Function**: `workflow.export_model(sb3_model_path, algorithm="PPO")`
- **Output**: Pure PyTorch model (`.pth` format)
- **Compatibility**: Ready for deployment in RLBot

### 4. Load Model
- **Purpose**: Load the converted model into the agent
- **Function**: `workflow.load_model(model_path)`
- **Validation**: Tests model with sample input
- **Integration**: Prepares agent for gameplay

### 5. Play with Model
- **Purpose**: Test the trained agent in the environment
- **Function**: `workflow.play_with_model(agent, n_episodes=3)`
- **Features**:
  - Multiple episode testing
  - Performance metrics
  - Optional rendering

## 🎯 Configuration Options

### Training Configuration
```python
config = {
    "algorithm": "PPO",           # Algorithm: PPO, SAC, TD3, A2C, DQN
    "timesteps": 100000,           # Training duration
    "save_freq": 10000,            # Checkpoint frequency
    "resume_from": None,           # Resume from checkpoint
    "test_after": True,            # Test agent after training
    "auto_convert": True,          # Auto-convert to PyTorch (corresponds to --no-convert in CLI)
    "auto_play": True,             # Auto-test after conversion (corresponds to --no-play in CLI)
    "test_episodes": 3,            # Number of test episodes
    "render_test": False           # Render during testing
}
```

### Environment Configuration
```python
env_config = {
    "difficulty": "medium",        # Training difficulty
    "max_steps": 1000,            # Max steps per episode
    "tick_skip": 8,               # Game ticks between actions
    "spawn_opponents": False      # Include opponents
}
```

## 🔧 Advanced Usage

### Custom Workflow Execution
```python
workflow = CompleteWorkflow()

# Step-by-step execution
workflow.load_map(difficulty="hard")
workflow.train_agent(algorithm="SAC", timesteps=200000)
workflow.export_model("models/SAC/SAC_1.zip", "SAC")
workflow.load_model("models/SAC/SAC_converted.pth")
workflow.play_with_model(workflow.agent, n_episodes=5)
```

### Individual Component Testing
```python
# Test specific components
workflow._test_converted_model("models/PPO/PPO_converted.pth", test_input)

# Update bot configuration
workflow.update_bot_config("models/PPO/PPO_converted.pth")
```

### Batch Processing
```python
# Train multiple agents
algorithms = ["PPO", "SAC", "TD3"]
for algorithm_type in algorithms:
    config = {"algorithm": algorithm_type, "timesteps": 50000}
    success = workflow.run_complete_workflow(config)
    if success:
        print(f"✅ {algorithm_type} training completed successfully")
```

## 📁 File Structure

```
RLGymExampleBot/
├── src/
│   ├── complete_workflow.py      # Main workflow orchestrator
│   ├── demo_workflow.py          # Usage examples and demos
│   ├── train.py                  # Training functions
│   ├── convert_model.py          # Model conversion
│   ├── agent.py                  # Agent implementation
│   └── bot.py                    # RLBot integration
├── scripts/
│   └── complete_workflow.bat     # Windows batch file
├── models/                       # Trained models
│   ├── PPO/
│   ├── SAC/
│   └── TD3/
└── logs/                         # Training logs
```

## 🎮 Demo Functions

The `demo_workflow.py` file provides several pre-configured examples:

### Quick Training Demo
```python
from src.demo_workflow import demo_quick_training
success = demo_quick_training()  # 10k timesteps, PPO
```

### SAC Training Demo
```python
from src.demo_workflow import demo_sac_training
success = demo_sac_training()    # 50k timesteps, SAC
```

### Custom Workflow Demo
```python
from src.demo_workflow import demo_custom_workflow
success = demo_custom_workflow() # Custom configuration
```

### Component Testing Demo
```python
from src.demo_workflow import demo_component_testing
success = demo_component_testing() # Test individual steps
```

## 🔍 Monitoring and Debugging

### Training Progress
- **TensorBoard**: View training metrics in real-time
- **Console Output**: Progress bars and status updates
- **Log Files**: Detailed training logs in `logs/` directory

### Model Validation
- **Conversion Testing**: Automatic validation of converted models
- **Input/Output Testing**: Verify model behavior with sample data
- **Performance Metrics**: Episode rewards and completion rates

### Error Handling
- **Graceful Failures**: Individual step failures don't stop the pipeline
- **Detailed Logging**: Comprehensive error messages and stack traces
- **Recovery Options**: Resume from checkpoints or restart failed steps

## 🚨 Troubleshooting

### Common Issues

#### Training Fails to Start
```bash
# Check dependencies
pip install -r requirements.txt

# Verify environment setup
python -c "import gymnasium; import stable_baselines3"
```

#### Model Conversion Fails
```bash
# Ensure model file exists
ls models/PPO/PPO_1.zip

# Check agent type matches
python src/convert_model.py --help
```

#### Bot Integration Issues
```bash
# Verify model path in bot.py
grep "model_path" src/bot.py

# Check RLBot configuration
python run.py --help
```

### Performance Optimization

#### Training Speed
- Increase `tick_skip` for faster training
- Use vectorized environments
- Adjust batch sizes and learning rates

#### Memory Usage
- Reduce `max_steps` per episode
- Lower batch sizes
- Use gradient clipping

## 📚 API Reference

### CompleteWorkflow Class

#### Constructor
```python
CompleteWorkflow(project_root="..")
```

#### Methods

##### `load_map(difficulty="medium", max_steps=1000)`
Initialize training environment.

##### `train_agent(algorithm="PPO", timesteps=100000, save_freq=10000, resume_from=None, test_after=True)`
Train the RL agent.

##### `export_model(sb3_model_path, algorithm="PPO")`
Convert model to PyTorch format.

##### `load_model(model_path)`
Load converted model into agent.

##### `play_with_model(agent, n_episodes=3, render=False)`
Test the trained agent.

##### `update_bot_config(model_path)`
Update bot configuration with model path.

##### `run_complete_workflow(config=None)`
Execute complete pipeline.

## 🔗 Integration with Existing Systems

### Self-Play League
```python
# Use trained models in league
from src.self_play_league import SelfPlayLeague
league = SelfPlayLeague()
league.add_player("models/PPO/PPO_converted.pth", "PPO_Agent")
```

### RLBot Framework
```python
# Run trained bot
python run.py

# Or use GUI
python run_gui.py
```

### Hyperparameter Optimization
```python
# Integrate with existing optimization
from src.hyperparameter_optimization import optimize_hyperparameters
best_params = optimize_hyperparameters(workflow)
```

## 🎯 Best Practices

### Training Workflow
1. **Start Small**: Begin with 10k-50k timesteps for testing
2. **Validate Early**: Test models frequently during training
3. **Monitor Metrics**: Use TensorBoard to track progress
4. **Save Checkpoints**: Enable automatic checkpointing
5. **Test Thoroughly**: Validate converted models before deployment

### Model Management
1. **Version Control**: Keep track of model versions
2. **Backup Models**: Store important models in multiple locations
3. **Documentation**: Record training parameters and results
4. **Validation**: Test models with diverse scenarios

### Performance Tuning
1. **Environment Tuning**: Adjust difficulty and complexity
2. **Algorithm Selection**: Choose appropriate algorithm for task
3. **Hyperparameter Tuning**: Optimize learning rates and network sizes
4. **Hardware Utilization**: Use GPU acceleration when available

## 🚀 Future Enhancements

### Planned Features
- **Multi-Agent Training**: Train multiple agents simultaneously
- **Distributed Training**: Scale training across multiple machines
- **Advanced Metrics**: More sophisticated performance evaluation
- **AutoML Integration**: Automatic hyperparameter optimization
- **Cloud Deployment**: Training in cloud environments

### Extension Points
- **Custom Environments**: Add new game modes and scenarios
- **Algorithm Plugins**: Integrate new RL algorithms
- **Evaluation Frameworks**: Custom evaluation metrics
- **Deployment Options**: Additional deployment targets

## 📞 Support and Community

### Getting Help
- **Documentation**: Check this README and project docs
- **Issues**: Report bugs and feature requests
- **Discussions**: Join community discussions
- **Examples**: Review demo scripts and examples

### Contributing
- **Code**: Submit pull requests
- **Documentation**: Improve documentation
- **Testing**: Test on different platforms
- **Feedback**: Share experiences and suggestions

---

**Happy Training! 🚀⚽**

For more information, see the main [README.md](README.md) and project documentation.
