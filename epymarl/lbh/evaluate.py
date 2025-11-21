import sys
import os
# Add epymarl directory to path for lbh import
sys.path.append("..")
# Add src directory to path for modules imports (needed by controllers)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch as th
import numpy as np
import imageio
import gymnasium as gym
from types import SimpleNamespace
import matplotlib.pyplot as plt

# Import your custom environment package to trigger registration
import lbh 
from src.controllers import REGISTRY as mac_REGISTRY

def load_mac(checkpoint_path, env_info, args):
    """
    Loads a trained MAC (Multi-Agent Controller) from a checkpoint.
    """
    # Create scheme dictionary matching the structure expected by BasicMAC
    # Based on run.py lines 108-126
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "actions_onehot": {"vshape": (env_info["n_actions"],), "group": "agents"},
    }
    # Add reward based on common_reward setting
    if getattr(args, "common_reward", True):
        scheme["reward"] = {"vshape": (1,)}
    else:
        scheme["reward"] = {"vshape": (args.n_agents,)}
    
    groups = {"agents": args.n_agents}
    
    # BasicMAC expects (scheme, groups, args)
    mac = mac_REGISTRY["basic_mac"](scheme, groups, args)
    
    # Load weights
    model_path = os.path.join(checkpoint_path, "agent.th")
    # Force map_location to current device to avoid CUDA errors on CPU machines
    state_dict = torch.load(model_path, map_location=args.device)
    # Load directly into the agent (BasicMAC doesn't have load_state_dict, but agent does)
    mac.agent.load_state_dict(state_dict)
    mac.agent.to(args.device)
    mac.agent.eval()  # Set agent to evaluation mode
    
    return mac

def run_episode(mac, env, args, seed=42):
    """
    Runs a single episode and returns frames + total reward.
    """
    frames = []
    
    # Gym.make returns (obs, info)
    obs_tuple, info = env.reset(seed=seed)
    
    # Initialize hidden states for RNN (batch_size=1)
    # mac.init_hidden() sets mac.hidden_states internally, doesn't return it
    mac.init_hidden(batch_size=1)
    # Extract and reshape hidden states for agent forward pass
    # mac.hidden_states is (batch_size, n_agents, hidden_dim)
    # Agent expects (batch_size * n_agents, hidden_dim)
    hidden_states = mac.hidden_states.reshape(1 * args.n_agents, -1)
    last_actions = torch.zeros(1, args.n_agents, args.n_actions).to(args.device)
    
    terminated = False
    truncated = False
    total_reward = 0
    step = 0
    
    # Get unwrapped environment for direct method access
    unwrapped = env.unwrapped
    
    # Capture Frame 0
    frames.append(unwrapped.render(mode="rgb_array"))
    
    while not (terminated or truncated):
        # Prepare inputs
        obs_tensor = torch.tensor(np.array(obs_tuple), dtype=torch.float32).unsqueeze(0).to(args.device)
        
        # Access custom methods directly via unwrapped
        avail_actions = unwrapped.get_avail_actions()
            
        avail_tensor = torch.tensor(np.array(avail_actions), dtype=torch.long).unsqueeze(0).to(args.device)

        # Construct Input Vector manually based on standard EPyMARL RNN input
        # Match BasicMAC._build_inputs logic exactly
        inputs = [obs_tensor]
        
        # Add Last Action only if obs_last_action is True
        if args.obs_last_action:
            inputs.append(last_actions)
                
        # Add Agent ID only if obs_agent_id is True
        if args.obs_agent_id:
            agent_ids = torch.eye(args.n_agents, device=args.device).unsqueeze(0)
            inputs.append(agent_ids)
            
        # Concatenate features
        inputs = torch.cat([x.reshape(1*args.n_agents, -1) for x in inputs], dim=1)
        
        # Agent Forward Pass
        agent_outs, hidden_states = mac.agent(inputs, hidden_states)
        
        # Reshape hidden_states back to (batch_size, n_agents, hidden_dim) for next iteration
        # Agent returns (batch_size * n_agents, hidden_dim), need to reshape
        hidden_states = hidden_states.reshape(1, args.n_agents, -1)
        
        # Apply Action Masking manually
        # Set logits of unavailable actions to a very large negative number
        agent_outs[avail_tensor.reshape(1*args.n_agents, -1) == 0] = -1e10
        
        # Greedy Action Selection (ArgMax)
        chosen_actions = agent_outs.argmax(dim=1).reshape(args.n_agents).cpu().numpy()
        
        # Update Last Actions for next step
        if args.obs_last_action:
            last_actions = torch.zeros(1, args.n_agents, args.n_actions).to(args.device)
            for i, act_idx in enumerate(chosen_actions):
                last_actions[0, i, act_idx] = 1.0
        
        # Reshape hidden_states back to flat for next agent call
        hidden_states = hidden_states.reshape(1 * args.n_agents, -1)

        # Step Environment
        obs_tuple, rewards, terminated, truncated, info = env.step(tuple(chosen_actions))
        total_reward += sum(rewards)
        
        frames.append(unwrapped.render(mode="rgb_array"))
        step += 1
    
    return frames, total_reward

def evaluate_methods(env_id, models_dict):
    print(f"Initializing Environment: {env_id}")
    
    # --- 1. Setup Environment via Gym.make ---
    env = gym.make(env_id)
    unwrapped_env = env.unwrapped
    
    # Get Env Info
    obs_shape = unwrapped_env.get_obs_size()
    n_agents = unwrapped_env.get_n_agents()
    
    # Calculate State Size matching GymmaWrapper logic
    # GymmaWrapper concatenates all observations: n_agents * obs_size
    state_shape = n_agents * obs_shape
    
    env_info = {
        "n_agents": n_agents,
        "n_actions": unwrapped_env.get_total_actions(),
        "state_shape": state_shape, 
        "obs_shape": obs_shape
    }
    
    print(f"Environment Info: Obs Shape={obs_shape}, State Shape={state_shape}, Actions={env_info['n_actions']}")
    
    # --- 2. Run Evaluation Loop ---
    results = {}
    
    for method_name, path in models_dict.items():
        print(f"Evaluating: {method_name}...")
        
        # Setup Args - matching MAPPO config
        args = SimpleNamespace()
        args.n_agents = env_info["n_agents"]
        args.n_actions = env_info["n_actions"]
        args.hidden_dim = 128  # MAPPO config uses hidden_dim (not rnn_hidden_dim)
        args.use_rnn = True  # MAPPO uses RNN
        args.agent = "rnn"
        args.agent_output_type = "pi_logits"  # MAPPO uses policy logits
        args.action_selector = "soft_policies"  # Required by BasicMAC
        args.obs_agent_id = True  # MAPPO config
        args.obs_last_action = False  # MAPPO config
        args.common_reward = True  # Default assumption
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            mac = load_mac(path, env_info, args)
            frames, reward = run_episode(mac, env, args, seed=42) 
            
            results[method_name] = {
                "reward": reward,
                "frames": frames
            }
            print(f"  > Score: {reward:.4f}")
            
            # Save individual GIF
            gif_name = f"{env_id}_{method_name}.gif"
            imageio.mimsave(gif_name, frames, fps=4, loop=0)
            print(f"  > Saved {gif_name}")
            
        except Exception as e:
            print(f"  > Failed to load/run {method_name}: {e}")
            import traceback
            traceback.print_exc()

    # --- 3. Optional: Create Side-by-Side GIF ---
    if len(results) >= 2:
        print("Generating Comparison GIF...")
        keys = list(results.keys())
        frames1 = results[keys[0]]["frames"]
        frames2 = results[keys[1]]["frames"]
        
        # Pad to same length
        max_len = max(len(frames1), len(frames2))
        combined_frames = []
        
        for i in range(max_len):
            img1 = frames1[i] if i < len(frames1) else frames1[-1]
            img2 = frames2[i] if i < len(frames2) else frames2[-1]
            
            # Concatenate horizontally
            combined = np.hstack((img1, img2))
            combined_frames.append(combined)
            
        imageio.mimsave(f"comparison_{env_id}_{keys[0]}_vs_{keys[1]}.gif", combined_frames, fps=4, loop=0)
        print(f"Saved comparison_{env_id}_{keys[0]}_vs_{keys[1]}.gif")

if __name__ == "__main__":
    # Example Usage
    
    # 1. Define your models
    trained_models = {
        # "QMIX": "results/models/qmix_seed1/2000000", 
        "MAPPO": "../results/models/mappo_seed792134287_Hacking-3p-Medium-v0_2025-11-20/12506340"
    }
    
    # 2. Choose the Env ID you want to test on
    # Options: "Hacking-2p-Easy-v0", "Hacking-3p-Medium-v0", "Hacking-4p-Hard-v0"
    target_env_id = "Hacking-3p-Medium-v0"
    
    evaluate_methods(target_env_id, trained_models)