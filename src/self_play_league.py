import json
import os
import shutil
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from agent import Agent, NeuralNetwork
from logger import logger
from training_env import make_training_env, SimpleRocketLeagueEnv
from utils import SUPPORTED_AGENT_TYPES, create_model


@dataclass
class LeaguePlayer:
    """Represents a player in the self-play league"""
    name: str
    model_path: str
    agent_type: str
    rating: float = 1000.0
    mmr: float = 1000.0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    last_updated: float = 0.0
    version: int = 1
    total_training_steps: int = 0
    best_rating: float = 1000.0
    win_streak: int = 0
    loss_streak: int = 0
    
    @property
    def win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played
    
    @property
    def total_games(self) -> int:
        return self.wins + self.losses + self.draws
    
    @property
    def rating_change(self) -> float:
        return self.rating - self.best_rating
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LeaguePlayer':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class MatchResult:
    """Represents the result of a match"""
    player1: str
    player2: str
    winner: str
    reason: str
    steps: List[Dict]
    total_reward1: float
    total_reward2: float
    timestamp: float
    match_id: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SelfPlayCallback(BaseCallback):
    """Enhanced callback for self-play training"""
    
    def __init__(self, league, eval_freq=1000, verbose=0):
        super().__init__(verbose)
        self.league = league
        self.eval_freq = eval_freq
        self.last_eval = 0
        self.eval_results = []
        
    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval >= self.eval_freq:
            try:
                result = self.league.evaluate_current_agent(self.training_env)
                if result:
                    self.eval_results.append(result)
                self.last_eval = self.num_timesteps
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")
        return True


class SelfPlayLeague:
    """Enhanced self-play league for training multiple agents"""
    
    def __init__(self, league_dir="league", base_rating=1000.0, k_factor=32, 
                 min_games_for_rating=5, rating_decay_factor=0.95):
        self.league_dir = Path(league_dir)
        self.base_rating = base_rating
        self.k_factor = k_factor
        self.min_games_for_rating = min_games_for_rating
        self.rating_decay_factor = rating_decay_factor
        
        self.players: Dict[str, LeaguePlayer] = {}
        self.match_history: List[MatchResult] = []
        self.current_training_agent = None
        self.league_stats = {
            "total_matches": 0,
            "total_games": 0,
            "league_created": time.time(),
            "last_tournament": None
        }
        
        # Create league directory structure
        self._create_directories()
        
        # Load existing league data
        self.load_league()
        
        logger.info(f"Self-play league initialized at {self.league_dir}")
    
    def _create_directories(self):
        """Create necessary directory structure"""
        (self.league_dir / "models").mkdir(parents=True, exist_ok=True)
        (self.league_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.league_dir / "backups").mkdir(parents=True, exist_ok=True)
        (self.league_dir / "tournaments").mkdir(parents=True, exist_ok=True)
    
    def add_player(self, name: str, agent_type: str = "PPO", 
                   model_path: Optional[str] = None) -> str:
        """Add a new player to the league with validation"""
        if not name or not name.strip():
            raise ValueError("Player name cannot be empty")
        
        if name in self.players:
            raise ValueError(f"Player '{name}' already exists in the league")
        
        if agent_type not in SUPPORTED_AGENT_TYPES:
            raise ValueError(f"Invalid agent type: {agent_type}. Supported types: {', '.join(SUPPORTED_AGENT_TYPES)}")
        
        # Create model path if not provided
        if model_path is None:
            model_path = str(self.league_dir / "models" / f"{name}_v1.pth")
        else:
            # Validate model path
            if not os.path.exists(model_path):
                logger.warning(f"Model path {model_path} does not exist. Player will be created without a model.")
        
        player = LeaguePlayer(
            name=name.strip(),
            model_path=model_path,
            agent_type=agent_type,
            rating=self.base_rating,
            last_updated=time.time()
        )
        
        self.players[name] = player
        self.save_league()
        logger.info(f"Player '{name}' added to league")
        return name
    
    def remove_player(self, name: str):
        """Remove a player from the league"""
        if name not in self.players:
            raise ValueError(f"Player '{name}' not found in league")
        del self.players[name]
        self.save_league()
        logger.info(f"Player '{name}' removed from league")

    def create_initial_agent(self, name: str, agent_type: str = "PPO",
                            training_steps: int = 10000) -> str:
        """Create and train an initial agent for the league"""
        logger.info(f"Creating initial agent: {name} ({agent_type})")

        try:
            # Ensure player exists and get initial model path
            # If player doesn't exist, add them. If they do, get their current model_path.
            if name not in self.players:
                model_path = self.add_player(name, agent_type)
            else:
                player = self.players[name]
                model_path = player.model_path # Use existing model path

            # Create training environment
            env = make_training_env()
            vec_env = DummyVecEnv([lambda: env])

            # Initialize the agent
            model = create_model(agent_type, vec_env, self.league_dir / "models")

            # Train the agent
            logger.info(f"Training {name} for {training_steps} steps...")
            model.learn(total_timesteps=training_steps)

            # Save the model to the player's current model path
            model.save(model_path)

            # Update player stats
            player = self.players[name] # Get updated player object
            player.total_training_steps += training_steps
            player.version += 1 # Increment version after training
            player.last_updated = time.time() # Update last updated time

            # Save league data after updating player stats
            self.save_league()

            logger.info(f"Initial agent '{name}' created and saved to {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Failed to create agent '{name}': {e}")
            raise
    
    def play_match(self, player1_name: str, player2_name: str, 
                   n_games: int = 1, max_steps_per_game: int = 1000) -> List[MatchResult]:
        """Play matches between two players with enhanced game logic"""
        if player1_name not in self.players or player2_name not in self.players:
            raise ValueError("Both players must be in the league")
        
        if player1_name == player2_name:
            raise ValueError("Cannot play a match against yourself")
        
        if n_games < 1:
            raise ValueError("Number of games must be at least 1")
        
        logger.info(f"Playing {n_games} match(es) between '{player1_name}' and '{player2_name}'")
        
        matches = []
        env = make_training_env()
        
        try:
            for game in range(n_games):
                # Load both agents
                agent1 = self.load_agent(player1_name)
                agent2 = self.load_agent(player2_name)
                
                # Play the game
                result = self._play_single_game(
                    env, agent1, agent2, player1_name, player2_name, 
                    max_steps_per_game, game + 1
                )
                matches.append(result)
                
                # Update ratings and stats
                self._update_ratings(result)
                self._update_player_stats(result)
                
                # Update league stats
                self.league_stats["total_games"] += 1
                
                logger.info(f"Game {game + 1}: {result.winner} wins ({result.reason})")
            
            # Save league data
            self.save_league()
            self.league_stats["total_matches"] += 1
            
            return matches
            
        except Exception as e:
            logger.error(f"Match failed: {e}", exc_info=False)
            raise
        finally:
            env.close()
    
    def _play_single_game(self, env, agent1, agent2, player1_name: str, 
                          player2_name: str, max_steps: int, game_num: int) -> MatchResult:
        """Play a single game between two agents with enhanced logic"""
        obs, info = env.reset()
        done = False
        truncated = False
        step = 0
        
        # Track game state
        steps_data = []
        total_reward1 = 0.0
        total_reward2 = 0.0
        
        # Game state tracking
        last_ball_touch = None
        ball_positions = []
        car_positions = []
        
        while not done and not truncated and step < max_steps:
            try:
                # Agent 1's turn
                action1 = agent1.act(obs)
                obs, reward1, done, truncated, info = env.step(action1)
                total_reward1 += float(reward1)
                
                # Agent 2's turn (if game not over)
                if not done and not truncated:
                    action2 = agent2.act(obs)
                    obs, reward2, done, truncated, info = env.step(action2)
                    total_reward2 += float(reward2)
                else:
                    action2 = np.zeros(env.action_space.shape[0])  # No action if game is over
                    reward2 = 0.0
                
                # Record step data
                step_data = {
                    "step": step,
                    "action1": action1.tolist() if hasattr(action1, 'tolist') else action1,
                    "action2": action2.tolist() if hasattr(action2, 'tolist') else action2,
                    "reward1": float(reward1),
                    "reward2": float(reward2),
                    "observation": obs.tolist() if hasattr(obs, 'tolist') else obs
                }
                steps_data.append(step_data)
                
                step += 1
                
            except Exception as e:
                logger.warning(f"Error during game step {step}: {e}")
                break
        
        # Determine winner with more sophisticated logic
        winner, reason = self._determine_winner(
            total_reward1, total_reward2, step, max_steps, 
            player1_name, player2_name
        )
        
        # Create match result
        match_id = f"{player1_name}_vs_{player2_name}_game_{game_num}_{int(time.time())}"
        result = MatchResult(
            player1=player1_name,
            player2=player2_name,
            winner=winner,
            reason=reason,
            steps=steps_data,
            total_reward1=total_reward1,
            total_reward2=total_reward2,
            timestamp=time.time(),
            match_id=match_id
        )
        
        # Add to match history
        self.match_history.append(result)
        
        return result
    
    def _determine_winner(self, reward1: float, reward2: float, steps: int, 
                          max_steps: int, player1: str, player2: str) -> Tuple[str, str]:
        """Determine winner with sophisticated logic"""
        # Check for timeout
        if steps >= max_steps:
            if reward1 > reward2:
                return player1, "timeout_higher_reward"
            elif reward2 > reward1:
                return player2, "timeout_higher_reward"
            else:
                return "draw", "timeout_equal_reward"
        
        # Check for significant reward difference
        reward_diff = abs(reward1 - reward2)
        if reward_diff > 10.0:  # Significant difference threshold
            if reward1 > reward2:
                return player1, "significant_reward_lead"
            else:
                return player2, "significant_reward_lead"
        
        # Check for moderate reward difference
        if reward_diff > 2.0:
            if reward1 > reward2:
                return player1, "moderate_reward_lead"
            else:
                return player2, "moderate_reward_lead"
        
        # Small difference - consider it a draw
        return "draw", "close_match"
    
    def _update_ratings(self, match_result: MatchResult):
        """Update player ratings using enhanced ELO system"""
        player1_name = match_result.player1
        player2_name = match_result.player2
        winner = match_result.winner
        
        player1 = self.players[player1_name]
        player2 = self.players[player2_name]
        
        # Dynamic K-factor based on player experience
        k1 = self._calculate_dynamic_k_factor(player1)
        k2 = self._calculate_dynamic_k_factor(player2)
        
        # Calculate expected scores
        expected1 = 1 / (1 + 10 ** ((player2.rating - player1.rating) / 400))
        expected2 = 1 - expected1
        
        # Calculate actual scores
        if winner == player1_name:
            actual1, actual2 = 1.0, 0.0
        elif winner == player2_name:
            actual1, actual2 = 0.0, 1.0
        else:  # Draw
            actual1, actual2 = 0.5, 0.5
        
        # Update ratings
        rating_change1 = k1 * (actual1 - expected1)
        rating_change2 = k2 * (actual2 - expected2)
        
        player1.rating += rating_change1
        player2.rating += rating_change2
        
        # Ensure ratings don't go below 100
        player1.rating = max(100, player1.rating)
        player2.rating = max(100, player2.rating)
        
        # Update best rating
        player1.best_rating = max(player1.best_rating, player1.rating)
        player2.best_rating = max(player2.best_rating, player2.rating)
        
        # Update last updated time
        player1.last_updated = time.time()
        player2.last_updated = time.time()
        
        logger.info(f"Rating update: {player1_name} {rating_change1:+.1f} -> {player1.rating:.1f}, "
                   f"{player2_name} {rating_change2:+.1f} -> {player2.rating:.1f}")
    
    def _calculate_dynamic_k_factor(self, player: LeaguePlayer) -> float:
        """Calculate dynamic K-factor based on player experience and rating"""
        base_k = self.k_factor
        
        # Reduce K-factor for experienced players
        if player.games_played > 30:
            base_k *= 0.5
        elif player.games_played > 20:
            base_k *= 0.7
        elif player.games_played > 10:
            base_k *= 0.85
        
        # Adjust K-factor based on rating
        if player.rating > 2000:
            base_k *= 0.8
        elif player.rating > 1500:
            base_k *= 0.9
        
        return max(16, base_k)  # Minimum K-factor of 16
    
    def _update_player_stats(self, match_result: MatchResult):
        """Update player statistics with enhanced tracking"""
        player1_name = match_result.player1
        player2_name = match_result.player2
        winner = match_result.winner
        
        player1 = self.players[player1_name]
        player2 = self.players[player2_name]
        
        # Update games played
        player1.games_played += 1
        player2.games_played += 1
        
        # Update wins/losses/draws and streaks
        if winner == player1_name:
            player1.wins += 1
            player2.losses += 1
            player1.win_streak += 1
            player1.loss_streak = 0
            player2.win_streak = 0
            player2.loss_streak += 1
        elif winner == player2_name:
            player1.losses += 1
            player2.wins += 1
            player1.win_streak = 0
            player1.loss_streak += 1
            player2.win_streak += 1
            player2.loss_streak = 0
        else:  # Draw
            player1.draws += 1
            player2.draws += 1
            player1.win_streak = 0
            player1.loss_streak = 0
            player2.win_streak = 0
            player2.loss_streak = 0
    
    def load_agent(self, player_name: str) -> Agent:
        """Load an agent from the league with error handling"""
        if player_name not in self.players:
            raise ValueError(f"Player '{player_name}' not found in league")
        
        player = self.players[player_name]
        
        try:
            agent = Agent(model_path=player.model_path)
            return agent
        except Exception as e:
            logger.error(f"Failed to load agent for player '{player_name}': {e}", exc_info=False)
            raise
    
    def train_agent(self, name: str, agent_type: str = "PPO", 
                   total_timesteps: int = 100000, 
                   opponents: Optional[List[str]] = None) -> str:
        """Train an agent against league opponents with enhanced training"""
        logger.info(f"Training agent '{name}' for {total_timesteps} timesteps")
        
        try:
            # Create training environment
            env = make_training_env()
            vec_env = DummyVecEnv([lambda: env])
            
            # Initialize the agent
            model = create_model(agent_type, vec_env, self.league_dir / "models")
            
            # Add self-play callback
            callback = SelfPlayCallback(self, eval_freq=5000)
            
            # Train the agent
            model.learn(total_timesteps=total_timesteps, callback=callback)
            
            # Save the model with versioning
            if name not in self.players:
                self.add_player(name, agent_type)
            
            player = self.players[name]
            player.version += 1
            player.total_training_steps += total_timesteps
            
            # Create new model path with version
            new_model_path = str(self.league_dir / "models" / f"{name}_v{player.version}.pth")
            model.save(new_model_path)
            
            # Update player's model path
            player.model_path = new_model_path
            
            logger.info(f"Agent '{name}' trained and saved to {new_model_path}")
            return new_model_path
            
        except Exception as e:
            logger.error(f"Failed to train agent '{name}': {e}")
            raise
        finally:
            env.close()
    
    def evaluate_current_agent(self, training_env) -> Optional[Dict]:
        """Evaluate the currently training agent against league opponents"""
        if not self.players:
            return None
        
        try:
            # Get the best opponent
            best_opponent = max(self.players.values(), key=lambda p: p.rating)
            
            logger.info(f"Evaluating current agent against {best_opponent.name} "
                       f"(rating: {best_opponent.rating:.1f})")
            
            # For now, just log that evaluation happened
            # In a full implementation, you'd extract the policy and play matches
            return {
                "opponent": best_opponent.name,
                "opponent_rating": best_opponent.rating,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return None
    
    def get_leaderboard(self) -> List[LeaguePlayer]:
        """Get the current league leaderboard with enhanced sorting"""
        sorted_players = sorted(
            self.players.values(), 
            key=lambda p: (
                p.rating,           # Primary: rating
                p.win_rate,         # Secondary: win rate
                p.games_played,     # Tertiary: games played
                p.wins,             # Quaternary: total wins
                p.name              # Final: alphabetical
            ), 
            reverse=True
        )
        return sorted_players
    
    def print_leaderboard(self):
        """Print the current league leaderboard with enhanced formatting"""
        leaderboard = self.get_leaderboard()
        
        if not leaderboard:
            print("\n📋 No players in the league yet.")
            return
        
        print("\n" + "="*100)
        print("🏆 LEAGUE LEADERBOARD 🏆")
        print("="*100)
        print(f"{'Rank':<4} {'Name':<15} {'Rating':<8} {'Change':<8} {'W/L/D':<10} "
              f"{'Win Rate':<10} {'Games':<6} {'Streak':<8} {'Type':<6}")
        print("-"*100)
        
        for i, player in enumerate(leaderboard, 1):
            wld = f"{player.wins}/{player.losses}/{player.draws}"
            win_rate = f"{player.win_rate:.3f}"
            rating_change = f"{player.rating_change:+.1f}"
            
            # Determine streak display
            if player.win_streak > 0:
                streak = f"W{player.win_streak}"
            elif player.loss_streak > 0:
                streak = f"L{player.loss_streak}"
            else:
                streak = "-"
            
            print(f"{i:<4} {player.name:<15} {player.rating:<8.1f} {rating_change:<8} "
                  f"{wld:<10} {win_rate:<10} {player.games_played:<6} {streak:<8} {player.agent_type:<6}")
        
        print("="*100)
        
        # Print league statistics
        print(f"\n📊 League Statistics:")
        print(f"   Total Players: {len(leaderboard)}")
        print(f"   Total Matches: {self.league_stats['total_matches']}")
        print(f"   Total Games: {self.league_stats['total_games']}")
        print(f"   League Age: {(time.time() - self.league_stats['league_created']) / 86400:.1f} days")
    
    def run_tournament(self, games_per_match: int = 3, 
                      tournament_name: Optional[str] = None) -> Dict:
        """Run a tournament between all players with enhanced bracket system"""
        if len(self.players) < 2:
            raise ValueError("Need at least 2 players for a tournament")
        
        if not tournament_name:
            tournament_name = f"Tournament_{int(time.time())}"
        
        logger.info(f"🏆 Starting tournament '{tournament_name}' with {len(self.players)} players")
        logger.info(f"   Games per match: {games_per_match}")
        
        tournament_data = {
            "name": tournament_name,
            "timestamp": time.time(),
            "players": list(self.players.keys()),
            "games_per_match": games_per_match,
            "matches": [],
            "results": {}
        }
        
        # Create round-robin tournament
        player_names = list(self.players.keys())
        total_matches = len(player_names) * (len(player_names) - 1) // 2 * games_per_match
        current_match = 0
        
        try:
            for i, player1 in enumerate(player_names):
                for player2 in player_names[i+1:]:
                    current_match += 1
                    logger.info(f"🎮 Match {current_match}/{total_matches}: {player1} vs {player2}")
                    
                    try:
                        matches = self.play_match(player1, player2, games_per_match)
                        
                        # Count results
                        wins1 = sum(1 for m in matches if m.winner == player1)
                        wins2 = sum(1 for m in matches if m.winner == player2)
                        draws = sum(1 for m in matches if m.winner == "draw")
                        
                        match_result = {
                            "player1": player1,
                            "player2": player2,
                            "wins1": wins1,
                            "wins2": wins2,
                            "draws": draws,
                            "winner": player1 if wins1 > wins2 else (player2 if wins2 > wins1 else "draw")
                        }
                        
                        tournament_data["matches"].append(match_result)
                        
                        # Update tournament results
                        for player in [player1, player2]:
                            if player not in tournament_data["results"]:
                                tournament_data["results"][player] = {"wins": 0, "losses": 0, "draws": 0}
                            
                            if match_result["winner"] == player:
                                tournament_data["results"][player]["wins"] += 1
                            elif match_result["winner"] == "draw":
                                tournament_data["results"][player]["draws"] += 1
                            else:
                                tournament_data["results"][player]["losses"] += 1
                        
                        logger.info(f"   Result: {player1} {wins1} - {wins2} {player2} ({draws} draws)")
                        
                    except Exception as e:
                        logger.error(f"   ❌ Error in match {player1} vs {player2}: {e}")
                        continue
            
            # Save tournament data
            tournament_file = self.league_dir / "tournaments" / f"{tournament_name}.json"
            with open(tournament_file, 'w', encoding='utf-8') as f:
                json.dump(tournament_data, f, indent=2)
            
            # Update league stats
            self.league_stats["last_tournament"] = tournament_name
            
            logger.info(f"🏆 Tournament '{tournament_name}' completed!")
            return tournament_data
            
        except Exception as e:
            logger.error(f"Tournament failed: {e}")
            raise
    
    def save_league(self):
        """Save league data to disk with backup"""
        try:
            # Create backup of existing data
            league_file = self.league_dir / "league_data.json"
            if league_file.exists():
                backup_file = self.league_dir / "backups" / f"league_backup_{int(time.time())}.json"
                shutil.copy2(league_file, backup_file)
            
            # Prepare league data
            league_data = {
                "players": {name: player.to_dict() for name, player in self.players.items()},
                "match_history": [match.to_dict() for match in self.match_history[-1000:]],  # Keep last 1000 matches
                "league_stats": self.league_stats,
                "base_rating": self.base_rating,
                "k_factor": self.k_factor,
                "min_games_for_rating": self.min_games_for_rating,
                "rating_decay_factor": self.rating_decay_factor,
                "last_saved": time.time()
            }
            
            # Save to file
            with open(league_file, 'w', encoding='utf-8') as f:
                json.dump(league_data, f, indent=2)
            
            logger.debug("League data saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save league data: {e}")
            raise
    
    def load_league(self):
        """Load league data from disk with error handling"""
        league_file = self.league_dir / "league_data.json"
        
        try:
            if league_file.exists():
                with open(league_file, 'r', encoding='utf-8') as f:
                    league_data = json.load(f)
                
                # Load players
                for name, player_data in league_data["players"].items():
                    self.players[name] = LeaguePlayer.from_dict(player_data)
                
                # Load match history
                self.match_history = [MatchResult(**match_data) for match_data in league_data.get("match_history", [])]
                
                # Load other data
                self.league_stats = league_data.get("league_stats", self.league_stats)
                self.base_rating = league_data.get("base_rating", 1000.0)
                self.k_factor = league_data.get("k_factor", 32)
                self.min_games_for_rating = league_data.get("min_games_for_rating", 5)
                self.rating_decay_factor = league_data.get("rating_decay_factor", 0.95)
                
                logger.info(f"Loaded league with {len(self.players)} players and {len(self.match_history)} matches")
            else:
                logger.info("No existing league found, starting fresh")
                
        except Exception as e:
            logger.error(f"Failed to load league data: {e}")
            logger.info("Starting with fresh league data")
    
    def get_player_stats(self, player_name: str) -> Optional[Dict]:
        """Get comprehensive statistics for a specific player"""
        if player_name not in self.players:
            return None
        
        player = self.players[player_name]
        
        # Calculate additional stats
        total_games = player.total_games
        win_rate = player.win_rate
        rating_change = player.rating_change
        
        # Recent performance (last 10 games)
        recent_matches = [m for m in self.match_history[-20:] 
                         if player_name in [m.player1, m.player2]]
        recent_wins = sum(1 for m in recent_matches[-10:] 
                         if m.winner == player_name)
        recent_games = min(10, len(recent_matches))
        recent_win_rate = recent_wins / recent_games if recent_games > 0 else 0.0
        
        return {
            "basic_info": player.to_dict(),
            "performance": {
                "total_games": total_games,
                "win_rate": win_rate,
                "recent_win_rate": recent_win_rate,
                "rating_change": rating_change,
                "best_rating": player.best_rating,
                "current_streak": max(player.win_streak, player.loss_streak)
            },
            "recent_matches": recent_matches[-10:]  # Last 10 matches
        }
    
    def cleanup_old_data(self, max_matches: int = 1000, max_backups: int = 10):
        """Clean up old match history and backups"""
        try:
            # Clean up old matches
            if len(self.match_history) > max_matches:
                self.match_history = self.match_history[-max_matches:]
                logger.info(f"Cleaned up match history, keeping {max_matches} most recent matches")
            
            # Clean up old backups
            backup_dir = self.league_dir / "backups"
            if backup_dir.exists():
                backup_files = sorted(backup_dir.glob("league_backup_*.json"))
                if len(backup_files) > max_backups:
                    files_to_remove = backup_files[:-max_backups]
                    for file in files_to_remove:
                        file.unlink()
                    logger.info(f"Cleaned up {len(files_to_remove)} old backup files")
            
            self.save_league()
            
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")


def main():
    """Main function to demonstrate the enhanced self-play league"""
    print("🚀 Starting Enhanced Self-Play League System...")
    
    try:
        # Create league
        league = SelfPlayLeague()
        
        # Create initial agents if league is empty
        if not league.players:
            print("Creating initial agents...")
            league.create_initial_agent("Alpha", "PPO", training_steps=15000)
            league.create_initial_agent("Beta", "SAC", training_steps=15000)
            league.create_initial_agent("Gamma", "TD3", training_steps=15000)
        
        # Print current leaderboard
        league.print_leaderboard()
        
        # Play some matches
        print("\n🎮 Playing matches...")
        if len(league.players) >= 2:
            player_names = list(league.players.keys())
            league.play_match(player_names[0], player_names[1], n_games=3)
        
        # Print updated leaderboard
        league.print_leaderboard()
        
        # Run a tournament
        print("\n🏆 Running tournament...")
        tournament_results = league.run_tournament(games_per_match=2)
        
        # Train a new agent
        print("\n🎯 Training new agent...")
        league.train_agent("Delta", "PPO", total_timesteps=75000)
        
        # Play more matches
        print("\n🎮 Playing more matches...")
        if len(league.players) >= 2:
            player_names = list(league.players.keys())
            league.play_match(player_names[0], player_names[1], n_games=2)
        
        # Final leaderboard
        league.print_leaderboard()
        
        # Cleanup
        league.cleanup_old_data()
        
        print("\n✅ Enhanced self-play league demonstration completed!")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()
