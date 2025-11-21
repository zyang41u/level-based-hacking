import sys
sys.path.append("..")
import numpy as np
import imageio
from lbh.lbh_env import LBHEnv, ACTION_INTERACT, ACTION_NONE

def make_gif():
    print("Generating GIF...")
    
    # Setup Environment (Use your Medium Map logic)
    env = LBHEnv(
        grid_size=7,
        max_steps=50,
        min_hack_level=0, max_hack_level=20,
        min_combat_level=1, max_combat_level=5,
        min_security_level=1, max_security_level=20,
        min_threat_level=1, max_threat_level=10,
        reward_lambda=0.3,
        num_agents=3, num_enemies=2, num_data_centers=2,
        comm_allowed=True, force_combat_first=True,
        agent_configs=[{"combat": 4, "hack": 0}]*3,
        enemy_configs=[{"threat": 4}, {"threat": 1}],
        dc_configs=[{"security": 3}, {"security": 10}]
    )

    obs, info = env.reset(seed=42)
    frames = []
    
    # Capture Frame 0
    frames.append(env.render(mode="rgb_array"))
    
    # Run a few random steps
    for _ in range(20):
        actions = list(env.action_space.sample())

        actions[0] = ACTION_INTERACT
        
        obs, rewards, terminated, truncated, info = env.step(tuple(actions))
        frames.append(env.render(mode="rgb_array"))
        
        if terminated or truncated:
            break
            
    # Save GIF
    imageio.mimsave('lbh_visualization.gif', frames, fps=2, loop=0)
    print("Done! Saved to lbh_visualization.gif")

if __name__ == "__main__":
    make_gif()