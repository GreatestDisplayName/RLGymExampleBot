#!/usr/bin/env python3
"""
Demo script showing how to use the CompleteWorkflow class programmatically
"""

import time
from complete_workflow import CompleteWorkflow


def demo_quick_training():
    """Demo: Quick training run"""
    print("🎯 DEMO 1: Quick Training Run")
    print("=" * 50)
    
    workflow = CompleteWorkflow()
    
    config = {
        "agent_type": "PPO",
        "timesteps": 10000,  # Quick 10k steps
        "save_freq": 2000,
        "test_episodes": 3,
        "render_test": False,
        "auto_convert": True,
        "auto_play": True
    }
    
    success = workflow.run_complete_workflow(config)
    
    if success:
        print("✅ Quick training demo completed successfully!")
    else:
        print("❌ Quick training demo failed!")
    
    return success


def demo_sac_training():
    """Demo: SAC algorithm training"""
    print("\n🎯 DEMO 2: SAC Algorithm Training")
    print("=" * 50)
    
    workflow = CompleteWorkflow()
    
    config = {
        "agent_type": "SAC",
        "timesteps": 50000,  # Medium training
        "save_freq": 10000,
        "test_episodes": 5,
        "render_test": False,
        "auto_convert": True,
        "auto_play": True
    }
    
    success = workflow.run_complete_workflow(config)
    
    if success:
        print("✅ SAC training demo completed successfully!")
    else:
        print("❌ SAC training demo failed!")
    
    return success


def demo_custom_workflow():
    """Demo: Custom workflow with manual control"""
    print("\n🎯 DEMO 3: Custom Workflow with Manual Control")
    print("=" * 50)
    
    workflow = CompleteWorkflow()
    
    # Step 1: Load environment
    print("Loading environment...")
    if not workflow.load_map(difficulty="hard", max_steps=1500):
        print("❌ Failed to load environment")
        return False
    
    # Step 2: Train agent
    print("Training agent...")
    model_path = workflow.train_agent(
        agent_type="PPO",
        total_timesteps=20000,
    )
    
    if not model_path:
        print("❌ Training failed")
        return False
    
    # Step 3: Export model
    print("Exporting model...")
    converted_path = workflow.export_model(model_path, "PPO")
    
    if not converted_path:
        print("❌ Model export failed")
        return False
    
    # Step 4: Load model
    print("Loading model...")
    agent = workflow.load_model(converted_path)
    
    if not agent:
        print("❌ Model loading failed")
        return False
    
    # Step 5: Play with model
    print("Playing with model...")
    play_results = workflow.play_with_model(agent, n_episodes=3, render=False)
    
    if play_results:
        print("✅ Custom workflow completed successfully!")
        print(f"   Play results: {play_results}")
        return True
    else:
        print("❌ Play testing failed")
        return False


def demo_workflow_components():
    """Demo: Individual workflow components"""
    print("\n🎯 DEMO 4: Individual Workflow Components")
    print("=" * 50)
    
    workflow = CompleteWorkflow()
    
    # Test environment loading
    print("1. Testing environment loading...")
    env_loaded = workflow.load_map(difficulty="easy", max_steps=500)
    print(f"   Environment loaded: {env_loaded}")
    
    # Test model conversion (if we have a trained model)
    print("\n2. Testing model conversion...")
    test_model_path = "../models/PPO/PPO_final"
    if workflow.models_dir.exists() and (workflow.models_dir / "PPO" / "PPO_final.zip").exists():
        converted = workflow.export_model(test_model_path, "PPO")
        print(f"   Model converted: {converted is not None}")
    else:
        print("   No trained model found, skipping conversion test")
    
    # Test bot configuration update
    print("\n3. Testing bot configuration update...")
    test_path = "../models/PPO/PPO_converted.pth"
    
    
    print("\n✅ Component testing completed!")


def main():
    """Main demo function"""
    print("🚀 RLGym Complete Workflow Demo")
    print("=" * 60)
    print("This demo shows different ways to use the CompleteWorkflow class")
    print("=" * 60)
    
    demos = [
        ("Quick Training", demo_quick_training),
        ("SAC Training", demo_sac_training),
        ("Custom Workflow", demo_custom_workflow),
        ("Component Testing", demo_workflow_components)
    ]
    
    print("\nAvailable demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"   {i}. {name}")
    
    print("\nRunning all demos...")
    print("=" * 60)
    
    results = []
    
    for name, demo_func in demos:
        try:
            print(f"\n🎬 Starting demo: {name}")
            start_time = time.time()
            
            success = demo_func()
            elapsed_time = time.time() - start_time
            
            results.append((name, success, elapsed_time))
            
            print(f"   Demo '{name}' completed in {elapsed_time:.1f}s")
            
            # Small delay between demos
            time.sleep(1)
            
        except Exception as e:
            print(f"   Demo '{name}' failed with error: {e}")
            results.append((name, False, 0))
    
    # Summary
    print("\n" + "=" * 60)
    print("🎬 DEMO SUMMARY")
    print("=" * 60)
    
    for name, success, elapsed_time in results:
        status = "✅ PASS" if success else "❌ FAIL"
        time_str = f"{elapsed_time:.1f}s" if elapsed_time > 0 else "N/A"
        print(f"   {name}: {status} ({time_str})")
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} demos passed")
    
    if passed == total:
        print("🎉 All demos completed successfully!")
    else:
        print("⚠️  Some demos failed. Check the output above for details.")
    
    print("\n🎯 Next steps:")
    print("   1. Use 'python complete_workflow.py --help' for command-line options")
    print("   2. Run 'complete_workflow.bat' on Windows")
    print("   3. Check the generated models and logs")


if __name__ == "__main__":
    main()
