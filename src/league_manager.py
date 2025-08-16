#!/usr/bin/env python3
"""
Enhanced League Manager - Command-line interface for the Self-Play League
"""

import argparse
import time
from pathlib import Path
from self_play_league import SelfPlayLeague
from logger import logger
from utils import SUPPORTED_AGENT_TYPES


def print_banner():
    """Print the league manager banner"""
    print("="*100)
    print("🚀 RLGym Enhanced Self-Play League Manager 🚀")
    print("="*100)
    print()


def add_player_command(league, args):
    """Add a new player to the league"""
    try:
        model_path = league.add_player(args.name, args.agent_type, args.model_path)
        print(f"✅ Successfully added player '{args.name}' to the league")
        print(f"   Model path: {model_path}")
        print(f"   Agent type: {args.agent_type}")
        print(f"   Initial rating: {league.base_rating}")
    except ValueError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def create_agent_command(league, args):
    """Create and train an initial agent"""
    try:
        print(f"🎯 Creating initial agent '{args.name}'...")
        start_time = time.time()
        
        model_path = league.create_initial_agent(
            args.name, 
            args.agent_type, 
            training_steps=args.training_steps
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"✅ Successfully created agent '{args.name}'")
        print(f"   Model saved to: {model_path}")
        print(f"   Agent type: {args.agent_type}")
        print(f"   Training steps: {args.training_steps}")
        print(f"   Time taken: {elapsed_time:.1f} seconds")
        
    except Exception as e:
        print(f"❌ Error creating agent: {e}")
        logger.exception(e)


def train_agent_command(league, args):
    """Train an existing agent"""
    try:
        print(f"🎯 Training agent '{args.name}'...")
        start_time = time.time()
        
        model_path = league.train_agent(
            args.name, 
            args.agent_type, 
            args.timesteps
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"✅ Successfully trained agent '{args.name}'")
        print(f"   Model saved to: {model_path}")
        print(f"   Training steps: {args.timesteps}")
        print(f"   Time taken: {elapsed_time:.1f} seconds")
        
    except Exception as e:
        print(f"❌ Error training agent: {e}")
        logger.exception(e)


def play_match_command(league, args):
    """Play matches between two players"""
    try:
        print(f"🎮 Playing {args.games} match(es) between '{args.player1}' and '{args.player2}'...")
        start_time = time.time()
        
        matches = league.play_match(args.player1, args.player2, args.games, args.max_steps)
        
        elapsed_time = time.time() - start_time
        
        print(f"✅ Matches completed in {elapsed_time:.1f} seconds!")
        
        # Show match results
        for i, match in enumerate(matches, 1):
            winner = match.winner
            reason = match.reason
            total_reward1 = match.total_reward1
            total_reward2 = match.total_reward2
            
            print(f"   Game {i}: {winner} wins ({reason})")
            print(f"      {args.player1}: {total_reward1:.2f} total reward")
            print(f"      {args.player2}: {total_reward2:.2f} total reward")
        
        # Show updated ratings
        print("\n📊 Updated ratings:")
        for player_name in [args.player1, args.player2]:
            player = league.players[player_name]
            rating_change = player.rating_change
            change_symbol = "+" if rating_change >= 0 else ""
            print(f"   {player_name}: {player.rating:.1f} ({change_symbol}{rating_change:.1f}) "
                  f"[W/L/D: {player.wins}/{player.losses}/{player.draws}]")
            
    except ValueError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.exception(e)


def leaderboard_command(league):
    """Show the league leaderboard"""
    try:
        league.print_leaderboard()
    except Exception as e:
        print(f"❌ Error displaying leaderboard: {e}")


def list_players_command(league):
    """List all players in the league"""
    try:
        if not league.players:
            print("📋 No players in the league yet.")
            return
        
        print("📋 League Players:")
        print("-" * 80)
        print(f"{'Name':<15} {'Type':<6} {'Rating':<8} {'Change':<8} {'W/L/D':<10} "
              f"{'Win Rate':<10} {'Streak':<8} {'Version':<8}")
        print("-" * 80)
        
        for player in league.players.values():
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
            
            print(f"{player.name:<15} {player.agent_type:<6} {player.rating:<8.1f} "
                  f"{rating_change:<8} {wld:<10} {win_rate:<10} {streak:<8} v{player.version:<7}")
        
        print("-" * 80)
        print(f"Total players: {len(league.players)}")
        
    except Exception as e:
        print(f"❌ Error listing players: {e}")


def info_command(league, args):
    """Show detailed information about a player"""
    try:
        if args.name not in league.players:
            print(f"❌ Player '{args.name}' not found in the league")
            return
        
        player = league.players[args.name]
        stats = league.get_player_stats(args.name)
        
        print(f"📊 Player Information: {player.name}")
        print("=" * 60)
        print(f"Agent Type:         {player.agent_type}")
        print(f"Current Rating:     {player.rating:.1f}")
        print(f"Best Rating:        {player.best_rating:.1f}")
        print(f"Rating Change:      {player.rating_change:+.1f}")
        print(f"Games Played:       {player.games_played}")
        print(f"Wins:               {player.wins}")
        print(f"Losses:             {player.losses}")
        print(f"Draws:              {player.draws}")
        print(f"Win Rate:           {player.win_rate:.3f}")
        print(f"Current Streak:     {max(player.win_streak, player.loss_streak)} "
              f"({'Wins' if player.win_streak > 0 else 'Losses' if player.loss_streak > 0 else 'None'})")
        print(f"Model Version:      v{player.version}")
        print(f"Total Training:     {player.total_training_steps:,} steps")
        print(f"Model Path:         {player.model_path}")
        print(f"Last Updated:       {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(player.last_updated))}")
        
        if stats and "performance" in stats:
            perf = stats["performance"]
            print(f"\n📈 Performance Metrics:")
            print(f"   Recent Win Rate:  {perf['recent_win_rate']:.3f}")
            print(f"   Best Rating:      {perf['best_rating']:.1f}")
        
        if stats and "recent_matches" in stats and stats["recent_matches"]:
            print(f"\n🎮 Recent Matches (Last 10):")
            for match in stats["recent_matches"][-10:]:
                opponent = match.player2 if match.player1 == args.name else match.player1
                result = "W" if match.winner == args.name else ("L" if match.winner == opponent else "D")
                print(f"   vs {opponent}: {result} ({match.reason})")
        
    except Exception as e:
        print(f"❌ Error displaying player info: {e}")


def tournament_command(league, args):
    """Run a tournament between all players"""
    try:
        if len(league.players) < 2:
            print("❌ Need at least 2 players for a tournament")
            return
        
        print(f"🏆 Starting tournament with {len(league.players)} players...")
        print(f"   Games per match: {args.games}")
        print(f"   Round-robin format")
        print()
        
        start_time = time.time()
        tournament_results = league.run_tournament(args.games, args.name)
        elapsed_time = time.time() - start_time
        
        print(f"\n🏆 Tournament completed in {elapsed_time:.1f} seconds!")
        print("\n📊 Final Standings:")
        league.print_leaderboard()
        
        # Show tournament summary
        if tournament_results and "results" in tournament_results:
            print(f"\n🏆 Tournament Summary:")
            results = tournament_results["results"]
            sorted_results = sorted(
                list(results.items()),
                key=lambda x: (x[1]["wins"], x[1]["draws"]),
                reverse=True
            )
            
            for i, (name, data) in enumerate(sorted_results, 1):
                print(f"   {i}. {name}: {data['wins']}W {data['losses']}L {data['draws']}D")
        
    except Exception as e:
        print(f"❌ Tournament failed: {e}")
        logger.exception(e)


def stats_command(league):
    """Show league statistics"""
    try:
        print("📊 League Statistics")
        print("=" * 50)
        
        stats = league.league_stats
        print(f"Total Players:      {len(league.players)}")
        print(f"Total Matches:      {stats['total_matches']}")
        print(f"Total Games:        {stats['total_games']}")
        
        if stats['league_created']:
            league_age = (time.time() - stats['league_created']) / 86400
            print(f"League Age:         {league_age:.1f} days")
        
        if stats['last_tournament']:
            print(f"Last Tournament:   {stats['last_tournament']}")
        
        # Player statistics
        if league.players:
            print(f"\n📈 Player Statistics:")
            total_games = sum(p.games_played for p in league.players.values())
            total_wins = sum(p.wins for p in league.players.values())
            total_draws = sum(p.draws for p in league.players.values())
            avg_rating = sum(p.rating for p in league.players.values()) / len(league.players)
            
            print(f"   Total Games:      {total_games}")
            print(f"   Total Wins:       {total_wins}")
            print(f"   Total Draws:      {total_draws}")
            print(f"   Average Rating:   {avg_rating:.1f}")
            
            # Top performers
            top_rated = max(league.players.values(), key=lambda p: p.rating)
            most_wins = max(league.players.values(), key=lambda p: p.wins)
            best_winrate = max(league.players.values(), key=lambda p: p.win_rate)
            
            print(f"\n🏆 Top Performers:")
            print(f"   Highest Rating:   {top_rated.name} ({top_rated.rating:.1f})")
            print(f"   Most Wins:        {most_wins.name} ({most_wins.wins})")
            print(f"   Best Win Rate:    {best_winrate.name} ({best_winrate.win_rate:.3f})")
        
    except Exception as e:
        print(f"❌ Error displaying statistics: {e}")


def cleanup_command(league, args):
    """Clean up old league data"""
    try:
        print("🧹 Cleaning up old league data...")
        league.cleanup_old_data(args.max_matches, args.max_backups)
        print("✅ Cleanup completed successfully!")
        
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")


def main():
    """Main function for the enhanced league manager"""
    parser = argparse.ArgumentParser(
        description="RLGym Enhanced Self-Play League Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add a new player
  python league_manager.py add-player Alpha PPO
  
  # Create and train an initial agent
  python league_manager.py create-agent Beta SAC --training-steps 20000
  
  # Train an existing agent
  python league_manager.py train-agent Alpha PPO --timesteps 100000
  
  # Play matches
  python league_manager.py play-match Alpha Beta --games 5 --max-steps 1500
  
  # Run a tournament
  python league_manager.py tournament --games 3 --name "Spring_2024"
  
  # View leaderboard
  python league_manager.py leaderboard
  
  # Show league statistics
  python league_manager.py stats
  
  # Clean up old data
  python league_manager.py cleanup --max-matches 500 --max-backups 5
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Add player command
    add_parser = subparsers.add_parser("add-player", help="Add a new player to the league")
    add_parser.add_argument("name", help="Player name")
    add_parser.add_argument("agent_type", choices=SUPPORTED_AGENT_TYPES, help="Agent type")
    add_parser.add_argument("--model-path", help="Path to existing model (optional)")
    
    # Create agent command
    create_parser = subparsers.add_parser("create-agent", help="Create and train an initial agent")
    create_parser.add_argument("name", help="Agent name")
    create_parser.add_argument("agent_type", choices=SUPPORTED_AGENT_TYPES, help="Agent type")
    create_parser.add_argument("--training-steps", type=int, default=15000, 
                              help="Training steps for initial agent")
    
    # Train agent command
    train_parser = subparsers.add_parser("train-agent", help="Train an existing agent")
    train_parser.add_argument("name", help="Agent name")
    train_parser.add_argument("agent_type", choices=SUPPORTED_AGENT_TYPES, help="Agent type")
    train_parser.add_argument("--timesteps", type=int, default=100000, help="Training timesteps")
    
    # Play match command
    play_parser = subparsers.add_parser("play-match", help="Play matches between two players")
    play_parser.add_argument("player1", help="First player name")
    play_parser.add_argument("player2", help="Second player name")
    play_parser.add_argument("--games", type=int, default=1, help="Number of games to play")
    play_parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per game")
    
    # Leaderboard command
    subparsers.add_parser("leaderboard", help="Show the league leaderboard")
    
    # List players command
    subparsers.add_parser("list-players", help="List all players in the league")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show detailed information about a player")
    info_parser.add_argument("name", help="Player name")
    
    # Tournament command
    tournament_parser = subparsers.add_parser("tournament", help="Run a tournament between all players")
    tournament_parser.add_argument("--games", type=int, default=1, help="Games per match")
    tournament_parser.add_argument("--name", help="Tournament name (optional)")
    
    # Stats command
    subparsers.add_parser("stats", help="Show league statistics")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old league data")
    cleanup_parser.add_argument("--max-matches", type=int, default=1000, 
                               help="Maximum matches to keep in history")
    cleanup_parser.add_argument("--max-backups", type=int, default=10, 
                               help="Maximum backup files to keep")
    
    args = parser.parse_args()
    
    if not args.command:
        print_banner()
        parser.print_help()
        return
    
    try:
        # Initialize league
        league = SelfPlayLeague()
        
        # Execute command
        if args.command == "add-player":
            add_player_command(league, args)
        elif args.command == "create-agent":
            create_agent_command(league, args)
        elif args.command == "train-agent":
            train_agent_command(league, args)
        elif args.command == "play-match":
            play_match_command(league, args)
        elif args.command == "leaderboard":
            leaderboard_command(league)
        elif args.command == "list-players":
            list_players_command(league)
        elif args.command == "info":
            info_command(league, args)
        elif args.command == "tournament":
            tournament_command(league, args)
        elif args.command == "stats":
            stats_command(league)
        elif args.command == "cleanup":
            cleanup_command(league, args)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.exception(e)


if __name__ == "__main__":
    main()
