from itertools import product
import math

from gymnasium import register


def is_valid_config(
    num_agents: int,
    num_enemies: int,
    num_data_centers: int,
    min_hack_level: float,
    max_hack_level: float,
    min_combat_level: float,
    max_combat_level: float,
    min_security_level: float,
    max_security_level: float,
    min_threat_level: float,
    max_threat_level: float,
    force_combat_first: bool,
) -> bool:
    """
    Validates environment configuration according to all rules in lbh_env.py.
    Returns True if valid, False otherwise.
    """
    # 1. Basic range validations
    if min_hack_level > max_hack_level:
        return False
    if min_combat_level > max_combat_level:
        return False
    if min_security_level > max_security_level:
        return False
    if min_threat_level > max_threat_level:
        return False
    if max_hack_level < max_security_level:
        return False
    
    # 2. Entity count validations
    if num_agents <= 0:
        return False
    if num_enemies < 0:
        return False
    if num_data_centers < 0:
        return False
    
    # 3. Combat validations
    # All enemies must be defeatable (worst case: all agents have min_combat_level)
    if num_agents > 0 and max_threat_level > num_agents * min_combat_level:
        return False
    
    # At least one enemy must be defeatable (if enemies exist)
    if num_agents > 0 and num_enemies > 0:
        single_agent_can_defeat = min_threat_level <= max_combat_level
        all_agents_can_defeat = min_threat_level <= num_agents * min_combat_level
        if not (single_agent_can_defeat or all_agents_can_defeat):
            return False
    
    # 4. Hacking validations
    # After defeating all enemies, agents must be able to hack all data centers
    if num_agents > 0 and num_enemies > 0 and num_data_centers > 0:
        worst_case_total_hacking = num_agents * min_hack_level + num_enemies * min_threat_level
        if worst_case_total_hacking < max_security_level:
            return False
    
    # 5. force_combat_first validation
    # If force_combat_first is True, min_hack_level must be < min_security_level
    # (so that initial hack levels can be capped below min DC security)
    if force_combat_first and num_data_centers > 0:
        if min_hack_level >= min_security_level:
            return False
    
    return True


# sizes = range(5, 20)
# players = range(2, 10)
# enemies = range(1, 10)
# data_centers = range(1, 5)
# comm_allowed = [True, False]
# force_combat_first = [True, False]

# # reward_lambdas = [0.1, 0.25, 0.35, 0.5]
# reward_lambdas = [0.1, 0.25, 0.45]

# for s, p, e, dc, comm, fcf, rl in product(
#     sizes, players, enemies, data_centers, comm_allowed, force_combat_first, reward_lambdas
# ):
#     # Fixed level ranges
#     min_hack = 0
#     max_hack = 70
#     min_combat = 5
#     max_combat = 20
#     min_security = 10
#     max_security = 60
    
#     # Calculate min_threat to ensure winnability after defeating all enemies
#     # Requirement: n_agents * min_hack_level + n_enemies * min_threat_level >= max_security_level
#     # Solving for min_threat: min_threat >= (max_security - n_agents * min_hack) / n_enemies
#     if e > 0:
#         required_min_threat = (max_security - p * min_hack) / e
#         min_threat = max(1, math.ceil(required_min_threat))
#     else:
#         min_threat = 1
    
#     # Calculate max_threat to ensure all enemies are defeatable
#     # Requirement: max_threat_level <= n_agents * min_combat_level
#     max_threat = p * min_combat
    
#     # Validate the configuration using the same rules as lbh_env.py
#     if not is_valid_config(
#         num_agents=p,
#         num_enemies=e,
#         num_data_centers=dc,
#         min_hack_level=min_hack,
#         max_hack_level=max_hack,
#         min_combat_level=min_combat,
#         max_combat_level=max_combat,
#         min_security_level=min_security,
#         max_security_level=max_security,
#         min_threat_level=min_threat,
#         max_threat_level=max_threat,
#         force_combat_first=fcf,
#     ):
#         continue
    
#     # Build environment ID following lb-foraging pattern
#     # Format: Hacking-{size}x{size}-{players}p-{enemies}e-{dcs}dc{flags}-v0
#     flags = ""
#     if not comm:
#         flags += "-nocomm"
#     if fcf:
#         flags += "-combatfirst"
    
#     flags += f"-lambda{rl}"
    
#     env_id = "Hacking-{0}x{0}-{1}p-{2}e-{3}dc{4}-v0".format(
#         s, p, e, dc, flags
#     )

#     register(
#         id=env_id,
#         entry_point="lbh.lbh_env:LBHEnv",
#         kwargs={
#             "grid_size": s,
#             "max_steps": 100,
#             "min_hack_level": min_hack,
#             "max_hack_level": max_hack,
#             "min_combat_level": min_combat,
#             "max_combat_level": max_combat,
#             "min_security_level": min_security,
#             "max_security_level": max_security,
#             "min_threat_level": min_threat,
#             "max_threat_level": max_threat,
#             "reward_lambda": rl,
#             "num_agents": p,
#             "num_enemies": e,
#             "num_data_centers": dc,
#             "comm_allowed": comm,
#             "force_combat_first": fcf,
#         },
#     )

register(
    id="Hacking-4p-Hard-v0",
    entry_point="lbh.lbh_env:LBHEnv",
    kwargs={
        "grid_size": 8,
        "max_steps": 150, # Increased slightly to allow for the sequence
        
        # --- Levels ---
        "min_hack_level": 0, "max_hack_level": 70,
        "min_combat_level": 1, "max_combat_level": 5,
        "min_security_level": 1, "max_security_level": 50,
        "min_threat_level": 2, "max_threat_level": 12,
        
        # --- Rewards ---
        "reward_lambda": 0.3,      # Keep Hacking Valuable
        "shaping_weight": 0.001,   # Keep Navigation Hint
        "xp_reward_weight": 0.005, # Keep Combat Incentive
        
        # --- Entity Counts ---
        "num_agents": 4,
        "num_enemies": 6,
        "num_data_centers": 4,
        
        "comm_allowed": True,
        "force_combat_first": False, # CHANGED: Allow hacking small DCs immediately
        
        # --- AGENTS: Give them a tiny bit of starting skill ---
        # Hack=1 allows them to grab the "Free Sample" DC immediately.
        "agent_configs": [
            {"combat": 5, "hack": 1}, # Strong Warrior, Weak Hacker
            {"combat": 5, "hack": 1}, # Strong Warrior, Weak Hacker
            {"combat": 2, "hack": 2}, # Balanced
            {"combat": 2, "hack": 2}, # Balanced
        ],

        # --- ENEMIES: Mix of easy and hard ---
        "enemy_configs": [
            {"threat": 2}, # Easy XP
            {"threat": 2}, # Easy XP
            {"threat": 4},
            {"threat": 4},
            {"threat": 8}, # Boss 1
            {"threat": 12},# Boss 2
        ],

        # --- DATA CENTERS: The Gradient ---
        "dc_configs": [
            # Tier 1: Free Sample (Security 1)
            # Agents can hack this IMMEDIATELY at spawn.
            # Teaches them: "Data Center Interaction = GOOD."
            {"security": 1}, 
            
            # Tier 2: Easy Goal (Security 6)
            # Requires killing ONE "Threat 2" enemy (shared XP).
            # Teaches them: "Kill -> Hack."
            {"security": 6},
            
            # Tier 3: Medium Goal (Security 15)
            # Requires clearing the small enemies.
            {"security": 15},
            
            # Tier 4: The Boss Goal (Security 40)
            # Requires clearing the map.
            {"security": 40},
        ],
    },
)

register(
    id="Hacking-2p-Easy-v0",
    entry_point="lbh.lbh_env:LBHEnv",
    kwargs={
        "grid_size": 6,
        "max_steps": 50,
        
        # --- Missing Required Arguments Added Here ---
        "min_hack_level": 0,
        "max_hack_level": 10,       # High enough to cover the 8 needed
        "min_combat_level": 1,
        "max_combat_level": 5,      # Covers the agent combat level (4)
        "min_security_level": 1,
        "max_security_level": 10,   # Covers the DC security (8)
        "min_threat_level": 1,
        "max_threat_level": 10,     # Covers the Enemy threat (8)

        # --- Rewards ---
        "reward_lambda": 0.3,
        "shaping_weight": 0.001,
        "xp_reward_weight": 0.005,
        
        # --- Entities ---
        "num_agents": 2,
        "num_enemies": 1,
        "num_data_centers": 1,
        
        "comm_allowed": True,
        "force_combat_first": True,
        
        # --- AGENTS ---
        "agent_configs": [
            {"combat": 4, "hack": 0},
            {"combat": 4, "hack": 0},
        ],

        # --- ENEMY (Threat 8) ---
        "enemy_configs": [
            {"threat": 8} 
        ],

        # --- DATA CENTER (Security 8) ---
        "dc_configs": [
            {"security": 8} 
        ],
    },
)

register(
    id="Hacking-3p-Medium-v0",
    entry_point="lbh.lbh_env:LBHEnv",
    kwargs={
        "grid_size": 7,             # Slightly larger than Easy (6), smaller than Hard (8)
        "max_steps": 75,            # Enough time for 2 fights + 1 hack
        
        # --- Level Bounds ---
        "min_hack_level": 0, "max_hack_level": 20,
        "min_combat_level": 1, "max_combat_level": 5,
        "min_security_level": 1, "max_security_level": 20,
        "min_threat_level": 1, "max_threat_level": 10,

        # --- Rewards ---
        "reward_lambda": 0.3,
        "shaping_weight": 0.001,
        "xp_reward_weight": 0.005,
        
        # --- Entities ---
        "num_agents": 3,
        "num_enemies": 2,
        "num_data_centers": 2,      # 1 Bait (Easy), 1 Goal (Hard)
        
        "comm_allowed": True,
        "force_combat_first": True,
        
        # --- AGENTS ---
        # 3 Balanced Agents (Combat 4)
        "agent_configs": [
            {"combat": 4, "hack": 0},
            {"combat": 4, "hack": 0},
            {"combat": 4, "hack": 0},
        ],

        # --- ENEMIES ---
        "enemy_configs": [
            # Enemy 1: The "Solo" kill. 
            # Any single agent can kill this. 
            {"threat": 4}, 
            
            # Enemy 2: The "Team" kill.
            # Requires 2 agents to coordinate.
            # XP Gain: 8 / 3 = 2 with remainder 2
            # First 2 agents get 3 XP, last agent gets 2 XP
            {"threat": 8},
        ],

        # --- DATA CENTERS ---
        "dc_configs": [
            # DC 1: The "Bait" / Progress Check.
            # Security 3.
            # Unlocked immediately after killing Enemy 1.
            # Teaches agents: "Killing unlocks Hacking."
            {"security": 3}, 
            
            # DC 2: The Real Goal.
            # Security 10.
            # After Enemy 1: Team Hack Skill is ~4. (Fail).
            # After Enemy 2: Team Hack Skill is ~12. (Success).
            # Forces agents to finish BOTH fights.
            {"security": 10}, 
        ],
    },
)