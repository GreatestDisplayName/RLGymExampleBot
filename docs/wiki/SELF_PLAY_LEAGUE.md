# 🚀 RLGym Self-Play League System

The Self-Play League is a powerful system that allows multiple AI agents to compete against each other and improve through self-play training. This system is inspired by successful approaches used in AlphaGo, AlphaZero, and other breakthrough AI systems.

## 🎯 What is Self-Play?

Self-play is a training technique where AI agents learn by playing against themselves or other AI agents. This creates a continuous improvement cycle:

1. **Agents compete** against each other in matches
2. **Performance is measured** using an ELO rating system
3. **Agents are trained** against the best opponents
4. **New strategies emerge** as agents adapt to each other
5. **Overall skill level increases** through this competitive environment

## 🏗️ System Architecture

### Core Components

- **`SelfPlayLeague`**: Main league management class
- **`LeaguePlayer`**: Represents each agent in the league
- **`SelfPlayCallback`**: Integrates with training to evaluate agents
- **`league_manager.py`**: Command-line interface for league management

### Key Features

- **ELO Rating System**: Fair ranking based on performance
- **Multiple Algorithm Types**: Support for PPO, SAC, and TD3 algorithms
- **Persistent Storage**: League data is saved and loaded automatically
- **Tournament System**: Run round-robin tournaments between all agents
- **Training Integration**: Seamlessly train agents within the league

## 🚀 Quick Start

### 1. Create Your First Agents

*Note: The `league_manager.py` script uses the reinforcement learning algorithm name (e.g., PPO, SAC, TD3) as an "agent type" argument when creating and managing agents within the league.*

```bash
# Create and train initial agents
python src/league_manager.py create-agent <AGENT_NAME> <ALGORITHM_NAME>
# Example:
python src/league_manager.py create-agent Alpha PPO
python src/league_manager.py create-agent Beta SAC
python src/league_manager.py create-agent Gamma TD3
```

### 2. View the League

```bash
# See current standings
python src/league_manager.py leaderboard

# List all players
python src/league_manager.py list-players
```

### 3. Play Matches

```bash
# Play a single match
python src/league_manager.py play-match Alpha Beta

# Play multiple games
python src/league_manager.py play-match Alpha Beta --games 5
```

### 4. Run a Tournament

```bash
# Run a tournament with 3 games per match
python src/league_manager.py tournament --games 3
```

### 5. Train Agents

```bash
# Train an existing agent
python src/league_manager.py train-agent Alpha PPO --timesteps 100000
```

## 🎮 Using the Windows Batch File

For Windows users, you can use the convenient batch file. Remember that `league_manager.bat` requires a command (e.g., `create-agent`, `play-match`, `leaderboard`). If no command is provided, it will display its help message.

```batch
# Show help (by running without arguments)
league_manager.bat

# Create agents
league_manager.bat create-agent Alpha PPO
league_manager.bat create-agent Beta SAC

# Play matches
league_manager.bat play-match Alpha Beta --games 3

# View leaderboard
league_manager.bat leaderboard
```

## 📊 Understanding the System

### ELO Rating System

The league uses an ELO rating system (similar to chess ratings) to quantify the relative skill levels of agents. A higher ELO rating indicates a stronger agent.

- **Base Rating**: 1000 (starting point for new agents)
- **K-Factor**: 32 (how much ratings change after each match; a higher K-factor means ratings change more rapidly)
- **Rating Changes**: Based on expected vs. actual performance (e.g., beating a much higher-rated opponent yields a larger rating increase than beating a lower-rated one).

### Match Results

Matches are determined by:
- **Total Reward**: Sum of rewards over all steps
- **Higher Reward Wins**: Agent with better performance wins
- **Draws**: When rewards are equal

### Player Statistics

Each player tracks:
- **Rating**: Current ELO rating
- **Games Played**: Total number of matches
- **Wins/Losses/Draws**: Match outcomes
- **Win Rate**: Percentage of games won

## 🔧 Advanced Usage

### Custom Training

```python
from src.self_play_league import SelfPlayLeague

# Create league
league = SelfPlayLeague()

# Train agent with custom parameters
league.train_agent("CustomAgent", "PPO", total_timesteps=200000)

# Play against specific opponents
league.play_match("CustomAgent", "Alpha", n_games=10)
```

### League Management

```python
# Add existing model
league.add_player("ExistingAgent", "PPO", model_path="path/to/model.pth")

# Get player info
player = league.players["Alpha"]
print(f"Rating: {player.rating}, Win Rate: {player.win_rate}")

# Save/load league data
league.save_league()
league.load_league()
```

### Custom Callbacks

```python
from src.self_play_league import SelfPlayCallback

# Create callback with custom evaluation frequency
callback = SelfPlayCallback(league, eval_freq=2000)

# Use in training
model.learn(total_timesteps=100000, callback=callback)
```

## 📁 File Structure

```
league/
├── models/           # Trained agent models
├── logs/            # Training logs
└── league_data.json # League state and statistics
```

## 🎯 Best Practices

### 1. **Start Small**
- Begin with 2-3 agents
- Use shorter training runs initially
- Focus on getting the system working

### 2. **Balanced Training**
- Train agents for similar amounts of time
- Use different algorithms for diversity
- Avoid overtraining single agents

### 3. **Regular Evaluation**
- Play matches frequently
- Monitor rating changes
- Identify which agents are improving

### 4. **Tournament Scheduling**
- Run tournaments after major training sessions
- Use multiple games per match for reliability
- Track long-term performance trends

## 🔍 Troubleshooting

### Common Issues

**Agent Not Learning**
- Check training timesteps
- Verify environment compatibility
- Monitor reward signals

**Rating Not Updating**
- Ensure matches are completing
- Check league data file permissions
- Verify agent names match

**Training Errors**
- Check model compatibility
- Verify observation/action spaces
- Monitor memory usage

### Debug Commands

```bash
# Check league status
python src/league_manager.py info Alpha

# Verify environment
python src/training_env.py

# Test agent loading
python -c "from src.agent import Agent; a = Agent(); print('Agent loaded successfully')"
```

## 🚀 Future Enhancements

### Planned Features

- **Real-time Visualization**: Live match viewing
- **Advanced Matchmaking**: Skill-based opponent selection
- **Meta-Learning**: Agents that learn to adapt strategies
- **Distributed Training**: Multi-machine training support
- **Performance Analytics**: Detailed skill breakdowns

### Integration Possibilities

- **RLBot Framework**: Real Rocket League gameplay
- **Complete Workflow System**: Train agents using the `complete_workflow.py` and then integrate them into the league for self-play.
- **Custom Environments**: Specialized training scenarios
- **External APIs**: Integration with other RL systems
- **Web Interface**: Browser-based league management

## 📚 Learning Resources

### Self-Play Papers
- **AlphaGo Zero**: Mastering the game of Go without human knowledge
- **AlphaZero**: A general reinforcement learning algorithm
- **MuZero**: Mastering Atari, Go, Chess and Shogi by planning

### RLGym Resources
- **Official Documentation**: [RLGym Documentation](https://rlgym.org/)
- **Community Examples**: [RLGym Examples](https://github.com/RLGym/RLGym)
- **Research Papers**: [RLGym Research](https://rlgym.org/research)

## 🤝 Contributing

The self-play league system is designed to be extensible. You can:

- **Add New Algorithms**: Implement additional RL algorithms
- **Improve Match Logic**: Enhance game simulation
- **Add Analytics**: Create better performance metrics
- **Optimize Training**: Improve training efficiency

## 📄 License

This self-play league system is part of the RLGym Example Bot project and follows the same licensing terms.

---

**Happy Training! 🚀⚽**

The self-play league will help your agents become stronger through competition and continuous improvement. Start with a few agents and watch them evolve into skilled players!
