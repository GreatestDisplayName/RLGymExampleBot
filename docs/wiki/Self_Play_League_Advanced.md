# Self-Play League: Advanced Usage

This page delves into advanced features and usage patterns for the RLGymExampleBot's self-play league system. For a basic introduction to the league, please refer to the main [Self-Play League documentation](../../SELF_PLAY_LEAGUE.md).

## 1. Understanding the League Structure

The self-play league is managed by the `SelfPlayLeague` class in `src/self_play_league.py`. It maintains:

*   **Players**: Instances of `LeaguePlayer` (defined in the same file), each representing an agent with its name, model path, agent type, and ELO rating.
*   **Match History**: A record of all played matches, including results and rewards.
*   **League Statistics**: Overall statistics about the league's activity.

League data is persistently stored in `league/league_data.json` within your project's root.

## 2. Managing League Players

### Adding Existing Models to the League

You can add a pre-trained model (e.g., one you trained outside the league or a converted `.pth` model) to the league.

```bash
# Add a player named "MyCustomBot" using a PPO model
python src/league_manager.py add-player MyCustomBot PPO --model-path models/PPO/my_custom_model_converted.pth
```

### Player Versioning

The league automatically tracks player versions. Each time an agent is trained within the league (using `train-agent`), its version number is incremented, and a new model file is saved (e.g., `Alpha_v2.pth`). This allows for tracking the evolution of agents.

## 3. Advanced Training within the League

The `train-agent` command allows you to train an existing league player. When training within the league, the agent can be evaluated against other league opponents.

```bash
# Train agent "Alpha" for an additional 100,000 timesteps
python src/league_manager.py train-agent Alpha PPO --timesteps 100000
```

## 4. Running Custom Tournaments

Beyond simple `play-match` commands, you can organize full tournaments between all active players in the league.

```bash
# Run a tournament where each pair plays 5 games
python src/league_manager.py tournament --games 5

# Run a named tournament
python src/league_manager.py tournament --games 3 --name "Summer_Championship"
```

Tournament results are saved in the `league/tournaments/` directory.

## 5. Interpreting League Statistics

The `leaderboard` and `stats` commands provide insights into the league's performance.

```bash
# View the current ELO leaderboard
python src/league_manager.py leaderboard

# Show overall league statistics
python src/league_manager.py stats

# Get detailed information about a specific player
python src/league_manager.py info MyCustomBot
```

## 6. League Data Management and Cleanup

The league data can grow over time. You can clean up old match history and backup files.

```bash
# Clean up old data, keeping only the last 500 matches and 5 backups
python src/league_manager.py cleanup --max-matches 500 --max-backups 5
```

## 7. Customizing League Logic

For advanced users, you can modify the `src/self_play_league.py` file to customize:

*   **ELO Rating System**: Adjust `base_rating`, `k_factor`, `min_games_for_rating`, or `rating_decay_factor`.
*   **Match Logic**: Modify `_play_single_game` or `_determine_winner` to change how games are played or winners are decided.
*   **Training Evaluation**: Customize `evaluate_current_agent` to define how agents are assessed during training within the league.

By leveraging these advanced features, you can build sophisticated self-play training regimes and gain deeper insights into your agents' learning progress.

[Home](Home.md)