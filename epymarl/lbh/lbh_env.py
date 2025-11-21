import gymnasium
from gymnasium import spaces
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


# --- Constants for Actions ---
# Actions are discrete, 0-5
ACTION_NONE = 0
ACTION_MOVE_NORTH = 1
ACTION_MOVE_SOUTH = 2
ACTION_MOVE_EAST = 3
ACTION_MOVE_WEST = 4
ACTION_INTERACT = 5

class LBHEnv(gymnasium.Env):
    """
    Level-Based Hacking (LBH) Multi-Agent Environment.

    This environment, as described in the proposal, is a grid world
    where agents must cooperate to defeat enemies to level up their hacking skill,
    and then cooperate to hack data centers to win.

    It is designed with an API similar to lb-foraging (LBF) for compatibility
    with multi-agent learning frameworks like EPyMARL.

    Key LBF/EPyMARL API Methods:
    - get_obs()
    - get_obs_size()
    - get_n_agents()
    - get_total_actions()
    - get_avail_actions()
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        grid_size: int,
        max_steps: int,
        min_hack_level: int,
        max_hack_level: int,
        min_combat_level: int,
        max_combat_level: int,
        min_security_level: int,
        max_security_level: int,
        min_threat_level: int,
        max_threat_level: int,
        reward_lambda: float,
        num_agents: int,
        num_enemies: int,
        num_data_centers: int,
        agent_configs: Optional[List[Dict[str, int]]] = None,
        enemy_configs: Optional[List[Dict[str, int]]] = None,
        dc_configs: Optional[List[Dict[str, int]]] = None,
        comm_allowed: bool = True,
        force_combat_first: bool = False,
        shaping_weight: float = 0.0, # Weight for the dense reward
        xp_reward_weight: float = 0.0, # Weight for the XP reward
    ):

        """
        Initializes the LBH Environment.

        Args:
            grid_size (int): The width and height of the square grid (e.g., 8 for 8x8).
            max_steps (int): Maximum number of steps per episode.
            reward_lambda (float): Weight for the combat reward (lambda in the proposal).
                                   (1 - lambda) is the weight for the hacking reward.
            num_agents (int): Number of agents to spawn.
            num_enemies (int): Number of enemies to spawn.
            num_data_centers (int): Number of data centers to spawn.
            agent_configs (Optional[List[Dict]]): Optional per-agent overrides.
                Missing fields (or entire configs) are randomized between the
                configured min/max levels.
            enemy_configs (Optional[List[Dict]]): Optional per-enemy overrides.
            dc_configs (Optional[List[Dict]]): Optional per-data-center overrides.
            comm_allowed (bool): If True, agents can observe other agents' stats.
                                 If False, other agents' stats are masked.
            force_combat_first (bool): If True, initial hacking skills are capped
                                       so they cannot exceed the weakest data
                                       center security.
            shaping_weight (float): Weight for the dense reward.
            xp_reward_weight (float): Weight for the XP reward.
        """

        super().__init__()

        self.grid_size = (grid_size, grid_size)
        self._max_steps = max_steps
        self.reward_lambda = reward_lambda
        self.comm_allowed = comm_allowed
        self.force_combat_first = force_combat_first
        self.shaping_weight = shaping_weight
        self.xp_reward_weight = xp_reward_weight
        
        if min_hack_level > max_hack_level:
            raise ValueError("min_hack_level must be <= max_hack_level")
        if min_combat_level > max_combat_level:
            raise ValueError("min_combat_level must be <= max_combat_level")
        if min_security_level > max_security_level:
            raise ValueError("min_security_level must be <= max_security_level")
        if min_threat_level > max_threat_level:
            raise ValueError("min_threat_level must be <= max_threat_level")
        if max_hack_level < max_security_level:
            raise ValueError(
                "max_hack_level must be >= max_security_level to ensure data "
                "centers are hackable after leveling."
            )

        self.min_hack_level = min_hack_level
        self.max_hack_level = max_hack_level

        self.min_combat_level = min_combat_level
        self.max_combat_level = max_combat_level

        self.min_security_level = min_security_level
        self.max_security_level = max_security_level

        self.min_threat_level = min_threat_level
        self.max_threat_level = max_threat_level

        if num_agents <= 0:
            raise ValueError("num_agents must be > 0")
        if num_enemies < 0:
            raise ValueError("num_enemies must be >= 0")
        if num_data_centers < 0:
            raise ValueError("num_data_centers must be >= 0")

        self.n_agents = num_agents
        self.n_enemies = num_enemies
        self.n_data_centers = num_data_centers

        # Initialize agent and enemy configs (optionally overridden)
        self.agent_init_configs = self._prepare_entity_configs(
            self.n_agents, agent_configs
        )
        self.enemy_init_configs = self._prepare_entity_configs(
            self.n_enemies, enemy_configs
        )
        self.dc_init_configs = self._prepare_entity_configs(
            self.n_data_centers, dc_configs
        )

        # Only perform worst-case validations if custom configs are not provided
        # If configs are provided, actual values will be used instead of worst-case scenarios
        has_custom_agent_configs = agent_configs is not None
        has_custom_enemy_configs = enemy_configs is not None
        has_custom_dc_configs = dc_configs is not None

        # Validate that enemies are defeatable in worst case
        # Worst case: all agents have min_combat_level, strongest enemy has max_threat_level
        # All agents must be able to defeat the strongest enemy together
        # Skip if custom configs are provided (actual values will be used)
        if (not has_custom_agent_configs and not has_custom_enemy_configs and 
            self.n_agents > 0 and max_threat_level > self.n_agents * min_combat_level):
            raise ValueError(
                f"max_threat_level ({max_threat_level}) must be <= "
                f"n_agents * min_combat_level ({self.n_agents * min_combat_level}) "
                f"to ensure all enemies are defeatable even when all agents have "
                f"minimum combat level."
            )
        
        # Validate that at least one enemy is defeatable by at least one agent
        # This ensures agents can level up. In worst case: at least one enemy must have
        # threat_level <= max_combat_level (so a single max-level agent can defeat it)
        # OR at least one enemy must have threat_level <= n_agents * min_combat_level
        # (so all agents together can defeat it)
        # Skip if custom configs are provided (actual values will be used)
        if (not has_custom_agent_configs and not has_custom_enemy_configs and
            self.n_agents > 0 and self.n_enemies > 0):
            # Check if at least one enemy can be defeated by a single agent
            single_agent_can_defeat = min_threat_level <= max_combat_level
            # Check if at least one enemy can be defeated by all agents together
            all_agents_can_defeat = min_threat_level <= self.n_agents * min_combat_level
            
            if not (single_agent_can_defeat or all_agents_can_defeat):
                raise ValueError(
                    f"No enemy is defeatable! At least one enemy must be defeatable. "
                    f"Current constraints: min_threat_level={min_threat_level}, "
                    f"max_combat_level={max_combat_level}, "
                    f"n_agents * min_combat_level={self.n_agents * min_combat_level}. "
                    f"Either min_threat_level must be <= max_combat_level (single agent) "
                    f"or <= n_agents * min_combat_level (all agents together)."
                )

        # Validate that after defeating all enemies, agents can hack all data centers
        # Worst case: all agents start with min_hack_level, all enemies have min_threat_level
        # Total XP from all enemies = n_enemies * min_threat_level
        # After all combat, total team hacking = n_agents * min_hack_level + n_enemies * min_threat_level
        # This must be >= max_security_level to ensure all DCs are hackable
        # Skip if custom configs are provided (actual values will be used)
        if (not has_custom_agent_configs and not has_custom_enemy_configs and 
            not has_custom_dc_configs and
            self.n_agents > 0 and self.n_enemies > 0 and self.n_data_centers > 0):
            worst_case_total_hacking = self.n_agents * min_hack_level + self.n_enemies * min_threat_level
            if worst_case_total_hacking < max_security_level:
                raise ValueError(
                    f"After defeating all enemies, worst-case total team hacking "
                    f"({worst_case_total_hacking}) must be >= max_security_level "
                    f"({max_security_level}) to ensure all data centers are hackable. "
                    f"Consider increasing min_hack_level, min_threat_level, or n_enemies, "
                    f"or decreasing max_security_level."
                )

        # Total threat and security levels for rewards normalization
        self.total_threat_level = 0
        self.total_security_level = 0

        # Interal state variables
        self.agents = []
        self.enemies = []
        self.data_centers = []
        self._grid = np.zeros(self.grid_size, dtype=np.int32)
        self._step_count = 0
        self._active_enemies = 0
        self._active_data_centers = 0

        # --- Define Gymnasium Spaces ---

        # 1. Action Space: Tuple of Discrete (6) for each agent
        # (NONE, N, S, E, W, INTERACT)
        self.action_space = spaces.Tuple(
            [spaces.Discrete(6) for _ in range(self.n_agents)]
        )

        # 2. Observation Space: Tuple of Box for each agent
        # The observation is a flat vector:
        # [Enemy 1 (x, y, threat), ..., Enemy N (x, y, threat),
        # DC 1 (x, y, security), ..., DC K (x, y, security),
        # Agent 1 (x, y, combat, hack), ..., Agent M (x, y, combat, hack)]
        # Note: The agent list is permuted so that the current agent is always first 
        # in its own observation's agent list.
        self._obs_size = (
            (self.n_enemies * 3) + (self.n_data_centers * 3) + (self.n_agents * 4)
        )
        obs_box = spaces.Box(
            low = -1.0,
            high = max(grid_size, max_hack_level, 
                    max_combat_level, max_security_level, 
                    max_threat_level),
            shape = (self._obs_size,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Tuple([obs_box] * self.n_agents)

        # 3. Random number generator
        self.np_random = None

    def get_obs_size(self) -> int:
        return self._obs_size
    
    # def get_state_size(self) -> int:
    #     return self._obs_size
    
    def get_n_agents(self) -> int:
        return self.n_agents
    
    def get_total_actions(self) -> int:
        return 6
    
    def get_agent_avail_actions(self, agent_id: int) -> List[int]:
        """
        Returns the available actions for the agent.
        Format: [None, MOVE_N, MOVE_S, MOVE_E, MOVE_W, INTERACT]
        1 = available, 0 = unavailable
        """

        avail = [1] * 6

        # --- Action masking ---
        # Only allow INTERACG if adjacent to an active target
        agent = self.agents[agent_id]
        can_interact = False

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # N, S, E, W
            nx, ny = agent.x + dx, agent.y + dy
            if 0 <= nx < self.grid_size[1] and 0 <= ny < self.grid_size[0]:
                cell_val = self._grid[ny, nx]

                if cell_val == 2: # Enemy
                    if self._is_active_target_at(nx, ny, self.enemies):
                        can_interact = True
                        break
                elif cell_val == 3: # Data Center
                    if self._is_active_target_at(nx, ny, self.data_centers):
                        can_interact = True
                        break

        if not can_interact:
            avail[ACTION_INTERACT] = 0
        
        return avail
    
    def _is_active_target_at(self, x: int, y: int, entity_list: List[Any]) -> bool:
        """
        Checks if an active target is at the given position.
        """
        for e in entity_list:
            if e.active and e.x == x and e.y == y:
                return True
        return False

    
    def get_avail_actions(self) -> List[List[int]]:
        return [self.get_agent_avail_actions(i) for i in range(self.n_agents)]
    
    def get_obs(self) -> List[np.ndarray]:
        """
        Returns the list of observations for all agents.
        """
        return self._get_obs()

    # def get_state(self) -> np.ndarray:
    #     """
    #     Returns the global state vector.
    #     Just define this as agent 0's observation as it's fully observable.
    #     """
    #     return self._get_obs()[0]


    # --- Gymnasium Core API ---

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tuple[np.ndarray, ...], Dict[str, Any]]:
        """
        Resets the environment to the initial state.
        """
        if seed is not None:
            super().reset(seed=seed)
        
        # Initialize the random number generator
        if self.np_random is None:
            self.np_random, _ = gymnasium.utils.seeding.np_random(seed)
        
        self._step_count = 0
        self.agents = []
        self.enemies = []
        self.data_centers = []
        self._grid = np.zeros(self.grid_size, dtype=int) # 0: empty, 1: agent, 2: enemy, 3: data center
        self.total_threat_level = 0
        self.total_security_level = 0

        # Place all entities randomly on the grid without overlapping
        n_entities = self.n_agents + self.n_enemies + self.n_data_centers
        if n_entities > self.grid_size[0] * self.grid_size[1]:
            raise ValueError(
                "Not enough space on the grid for all entities."
            )
        
        positions = self.np_random.choice(
            self.grid_size[0] * self.grid_size[1], n_entities, replace=False
        )
        positions_xy = [
            (pos % self.grid_size[0], pos // self.grid_size[1]) for pos in positions
        ]

        # Resolve entity stats prior to placement
        resolved_enemy_threats: List[int] = []
        self.total_threat_level = 0
        for base_config in self.enemy_init_configs:
            config = base_config or {}
            threat_level = self._resolve_level(
                config.get("threat"),
                self.min_threat_level,
                self.max_threat_level,
            )
            resolved_enemy_threats.append(threat_level)
            self.total_threat_level += threat_level

        resolved_dc_security: List[int] = []
        self.total_security_level = 0
        min_dc_security = None
        for base_config in self.dc_init_configs:
            config = base_config or {}
            security_level = self._resolve_level(
                config.get("security"),
                self.min_security_level,
                self.max_security_level,
            )
            resolved_dc_security.append(security_level)
            self.total_security_level += security_level
            if (min_dc_security is None) or (security_level < min_dc_security):
                min_dc_security = security_level

        hack_cap = None
        if self.force_combat_first and min_dc_security is not None:
            hack_cap = min_dc_security - 1
            if hack_cap < self.min_hack_level:
                raise ValueError(
                    "min_hack_level must be strictly less than the minimum data center "
                    "security level to enforce combat-first progression."
                )

        resolved_agent_stats: List[Tuple[int, int]] = []
        effective_hack_max = hack_cap if hack_cap is not None else self.max_hack_level
        for base_config in self.agent_init_configs:
            config = base_config or {}
            combat_skill = self._resolve_level(
                config.get("combat"),
                self.min_combat_level,
                self.max_combat_level,
            )
            hacking_skill = self._resolve_level(
                config.get("hack"),
                self.min_hack_level,
                effective_hack_max,
            )
            resolved_agent_stats.append((combat_skill, hacking_skill))

        # 1. Initialize agents
        pos_idx = 0
        for i, (combat_skill, hacking_skill) in enumerate(resolved_agent_stats):
            x, y = positions_xy[pos_idx]
            self.agents.append(
                _Agent(
                    id=i,
                    x=x,
                    y=y,
                    combat_skill=combat_skill,
                    hacking_skill=hacking_skill,
                )
            )
            self._grid[y, x] = 1
            pos_idx += 1
        
        # 2. Initialize enemies
        self._active_enemies = self.n_enemies
        for i, threat_level in enumerate(resolved_enemy_threats):
            x, y = positions_xy[pos_idx]
            self.enemies.append(
                _Enemy(
                    id=i,
                    x=x,
                    y=y,
                    threat_level=threat_level,
                )
            )
            self._grid[y, x] = 2
            pos_idx += 1
        
        # 3. Initialize data centers
        self._active_data_centers = self.n_data_centers
        for i, security_level in enumerate(resolved_dc_security):
            x, y = positions_xy[pos_idx]
            self.data_centers.append(
                _DataCenter(
                    id=i,
                    x=x,
                    y=y,
                    security_level=security_level,
                )
            )
            self._grid[y, x] = 3
            pos_idx += 1
        
        # Runtime validation: Verify that at least one enemy is actually defeatable
        # with the current agent combat skills
        if self.n_agents > 0 and self.n_enemies > 0:
            agent_combat_skills = [a.combat_skill for a in self.agents]
            enemy_threats = [e.threat_level for e in self.enemies]
            
            # Check if any single agent can defeat any enemy
            max_agent_combat = max(agent_combat_skills)
            min_enemy_threat = min(enemy_threats)
            single_agent_can_defeat = max_agent_combat >= min_enemy_threat
            
            # Check if all agents together can defeat any enemy
            total_combat = sum(agent_combat_skills)
            all_agents_can_defeat = total_combat >= min_enemy_threat
            
            if not (single_agent_can_defeat or all_agents_can_defeat):
                raise RuntimeError(
                    f"Invalid game state after reset: No enemy is defeatable! "
                    f"Agent combat skills: {agent_combat_skills}, "
                    f"Enemy threat levels: {enemy_threats}. "
                    f"Max agent combat ({max_agent_combat}) < min enemy threat ({min_enemy_threat}) "
                    f"and total combat ({total_combat}) < min enemy threat ({min_enemy_threat}). "
                    f"Agents cannot level up - game is unwinnable!"
                )
        
        return tuple(self._get_obs()), self._get_info()
    
    def step(
        self, actions: Tuple[int]
    ) -> Tuple[Tuple[np.ndarray, ...], List[float], bool, bool, Dict[str, Any]]:
        """
        Executes a step in the environment given the actions of all agents.
        """
        
        self._step_count += 1
        rewards = [0.0] * self.n_agents

        # 0. Calculate distance to nearest target
        dists_before = [0.0] * self.n_agents
        if self.shaping_weight > 0.0:
            for i, agent in enumerate(self.agents):
                dists_before[i] = self._get_min_dist_to_target(agent)


        # 1. Handle Movement
        # Agents move first, then interactions are resolved
        # Collect all intended moves first to handle conflicts properly
        intended_moves = {}  # Maps (new_x, new_y) -> list of agent indices trying to move there
        move_vectors = {}  # Maps agent index -> (dx, dy, new_x, new_y)
        
        for i in range(self.n_agents):
            agent = self.agents[i]
            action = actions[i]

            if action in [ACTION_NONE, ACTION_INTERACT]:
                continue
                
            dx, dy = 0, 0
            if action == ACTION_MOVE_NORTH:
                dy = 1
            elif action == ACTION_MOVE_SOUTH:
                dy = -1
            elif action == ACTION_MOVE_EAST:
                dx = 1
            elif action == ACTION_MOVE_WEST:
                dx = -1
            
            new_x, new_y = agent.x + dx, agent.y + dy
            
            # Only consider valid moves
            if self._is_valid_move(new_x, new_y):
                move_vectors[i] = (dx, dy, new_x, new_y)
                target_pos = (new_x, new_y)
                if target_pos not in intended_moves:
                    intended_moves[target_pos] = []
                intended_moves[target_pos].append(i)
        
        # Apply moves, handling conflicts: if multiple agents try to move to same cell, none move
        for i in range(self.n_agents):
            if i not in move_vectors:
                continue
            
            dx, dy, new_x, new_y = move_vectors[i]
            target_pos = (new_x, new_y)
            
            # Only move if this is the only agent trying to move to this cell
            if len(intended_moves[target_pos]) == 1:
                agent = self.agents[i]
                self._grid[agent.y, agent.x] = 0
                agent.x, agent.y = new_x, new_y
                self._grid[new_y, new_x] = 1
        
        # 1.5 Reward Shaping: Calculate distance to nearest target after movement
        # Prevent from sparse rewards by encouraging agents to move closer to targets.
        if self.shaping_weight > 0.0:
            for i, agent in enumerate(self.agents):
                dists_after = self._get_min_dist_to_target(agent)
                rewards[i] += self.shaping_weight * (dists_before[i] - dists_after)
        
        # 2. Handle Interactions (Simultaneous)
        # Find all agents performing 'INTERACT' and group them by target
        enemy_interactions: Dict[int, List[int]] = {
            e.id: [] for e in self.enemies if e.active
        }
        dc_interactions: Dict[int, List[int]] = {
            dc.id: [] for dc in self.data_centers if dc.active
        }
        for i in range(self.n_agents):
            if actions[i] != ACTION_INTERACT:
                continue
            
            agent = self.agents[i]

            # Priority: Attack enemies before hacking
            adjacent_enemies = self._get_adjacent_entities(agent, 2, self.enemies)
            if adjacent_enemies:
                # Select weakest enemy to attack
                target_enemy = min(adjacent_enemies, key=lambda e: e.threat_level)
                enemy_interactions[target_enemy.id].append(agent.id)
            else:
                adjacent_dcs = self._get_adjacent_entities(agent, 3, self.data_centers)
                if adjacent_dcs:
                    # Select least secure DC to hack
                    target_dc = min(adjacent_dcs, key=lambda dc: dc.security_level)
                    dc_interactions[target_dc.id].append(agent.id)
        
        # 3. Resolve combat
        for enemy_id, agent_ids in enemy_interactions.items():

            if not agent_ids:
                continue

            enemy = self.enemies[enemy_id]
            if not enemy.active:
                continue
            
            participating_agents = [self.agents[aid] for aid in agent_ids]
            total_combat = sum(a.combat_skill for a in participating_agents)

            if total_combat >= enemy.threat_level:
                # Agent team wins combat
                enemy.active = False
                self._grid[enemy.y, enemy.x] = 0
                self._active_enemies -= 1

                # Distribute XP (Hacking Skill increases)
                # Ensure all XP is distributed evenly, with remainder going to first agents
                base_xp = enemy.threat_level // len(participating_agents)
                remainder = enemy.threat_level % len(participating_agents)

                # Distribute rewards
                # We use the LBF-style reward described in the text:
                # R_combat = (threat * agent_combat) / (sum_participating_combat)
                for idx, agent in enumerate(participating_agents):
                    # First 'remainder' agents get one extra XP
                    xp_gain = base_xp + (1 if idx < remainder else 0)
                    agent.hacking_skill = int(agent.hacking_skill + xp_gain)
                    if self.total_threat_level > 0 and total_combat > 0:
                        # Normalize combat reward so total combat reward is 1
                        r_combat = (enemy.threat_level * agent.combat_skill) / (
                            self.total_threat_level * total_combat
                        )
                        rewards[agent.id] += self.reward_lambda * r_combat
                    if self.xp_reward_weight > 0.0:
                        # Reward XP gain for combat
                        # To address the issue of sparse rewards, we reward XP gain for combat.
                        rewards[agent.id] += self.xp_reward_weight * xp_gain

        # 4. Resolve hacking
        for dc_id, agent_ids in dc_interactions.items():
            if not agent_ids:
                continue

            dc = self.data_centers[dc_id]
            if not dc.active:
                continue
            
            participating_agents = [self.agents[aid] for aid in agent_ids]
            total_hacking = sum(a.hacking_skill for a in participating_agents)

            if total_hacking >= dc.security_level:
                # Agent team wins hacking
                dc.active = False
                self._grid[dc.y, dc.x] = 0
                self._active_data_centers -= 1
                
                # Distribute rewards
                # We use the LBF-style reward described in the text:
                # R_hacking = (security * agent_hacking) / (sum_participating_hacking)
                if self.total_security_level > 0 and total_hacking > 0:
                    for agent in participating_agents:
                        r_hack = (dc.security_level * agent.hacking_skill) / (
                            self.total_security_level * total_hacking
                        )
                        rewards[agent.id] += (1 - self.reward_lambda) * r_hack
        
        # 5. Set termination and truncation
        terminated = (self._active_data_centers == 0)
        truncated = (self._step_count >= self._max_steps)

        info = self._get_info()
        if truncated:
            info["episode_limit"] = True
        
        return tuple(self._get_obs()), rewards, terminated, truncated, info
    
    def render(self, mode: str = "human"):
        """
        Renders the environment in a human-readable format.
        (0, 0) is bottom-left corner.
        """
        if mode != "human":
            raise NotImplementedError(f"Render mode {mode} not implemented")


        grid_repr = [
            ["." for _ in range(self.grid_size[1])]
            for _ in range(self.grid_size[0])
        ]

        # Place entities on grid
        for enemy in self.enemies:
            if enemy.active:
                grid_repr[enemy.y][enemy.x] = "E"
        for dc in self.data_centers:
            if dc.active:
                grid_repr[dc.y][dc.x] = "D"
        for agent in self.agents:
            grid_repr[agent.y][agent.x] = str(agent.id)

        print(f"--- Step {self._step_count} ---")
        # Print grid (reversed to show (0, 0) at bottom-left)
        for row in reversed(grid_repr):
            print(" ".join(row))
        print("\nAgent Stats:")
        for agent in self.agents:
            print(
                f"Agent {agent.id}: Pos=({agent.x}, {agent.y}), "
                f"Combat={agent.combat_skill}, "
                f"Hacking={agent.hacking_skill}"
            )
        print("\nEnemy Stats:")
        for enemy in self.enemies:
            print(
                f"Enemy {enemy.id}: Pos=({enemy.x}, {enemy.y}), "
                f"Threat={enemy.threat_level}, "
                f"Active={enemy.active}"
            )
        print("\nData Center Stats:")
        for dc in self.data_centers:
            print(
                f"Data Center {dc.id}: Pos=({dc.x}, {dc.y}), "
                f"Security={dc.security_level}, "
                f"Active={dc.active}"
            )
        print("\nEnvironment Stats:")
        print(f"  Active Enemies: {self._active_enemies}/{self.n_enemies}")
        print(f"  Active Data Centers: {self._active_data_centers}/{self.n_data_centers}")
        print("-" * (self.grid_size[1] * 2 + 1))

    def close(self):
        pass

    def _get_obs(self) -> list[np.ndarray]:
        """
        Returns the list of observations for all agents.
        """
        
        # 1. Create feature arrays for all entities
        enemy_features = np.full((self.n_enemies, 3), -1.0, dtype=np.float32)
        for i, enemy in enumerate(self.enemies):
            if enemy.active:
                enemy_features[i] = [enemy.x, enemy.y, enemy.threat_level]
            else:
                enemy_features[i] = [-1.0, -1.0, 0.0]

        dc_features = np.full((self.n_data_centers, 3), -1.0, dtype=np.float32)
        for i, dc in enumerate(self.data_centers):
            if dc.active:
                dc_features[i] = [dc.x, dc.y, dc.security_level]
            else:
                dc_features[i] = [-1.0, -1.0, 0.0]

        agent_features = np.full((self.n_agents, 4), 0.0, dtype=np.float32)
        for i, agent in enumerate(self.agents):
            agent_features[i] = [
                agent.x, agent.y, agent.combat_skill, agent.hacking_skill
            ]
        
        # 2. Create the base observation vector (enemies + dcs)
        base_obs = np.concatenate(
            [enemy_features.flatten(), dc_features.flatten()]
        )

        # 3. Create agent-specific observations (base obs + ordered agent features)
        agent_obs_list = []
        for i in range(self.n_agents):
            # Order agent features: [self, other_agent_1, other_agent_2, ...]
            ordered_agent_features = np.zeros_like(agent_features)

            ordered_agent_features[0] = agent_features[i]

            idx = 1
            for j in range(self.n_agents):
                if i == j:
                    continue
                if self.comm_allowed:
                    ordered_agent_features[idx] = agent_features[j]
                else:
                    # Mask other agent's stats if communication is not allowed
                    ordered_agent_features[idx] = [-1.0, -1.0, 0.0, 0.0]
                
                idx += 1
            
            full_obs = np.concatenate(
                [base_obs, ordered_agent_features.flatten()]
            ).astype(np.float32)

            agent_obs_list.append(full_obs)

        return agent_obs_list

    def _prepare_entity_configs(
        self,
        count: int,
        configs: Optional[List[Dict[str, int]]],
    ) -> List[Dict[str, int]]:
        """
        Aligns optional configs list with required entity count.
        """
        if configs is None:
            return [{} for _ in range(count)]

        if len(configs) < count:
            configs = configs + [{} for _ in range(count - len(configs))]
        elif len(configs) > count:
            configs = configs[:count]

        return [dict(cfg) for cfg in configs]

    def _get_min_dist_to_target(self, agent: "_Agent") -> float:
        """
        Returns the minimum distance to any target entity (enemy or data center).
        """
        targets = []
        if self._active_enemies > 0:
            targets = [e for e in self.enemies if e.active]
        
        elif self._active_data_centers > 0:
            targets = [dc for dc in self.data_centers if dc.active]
        
        if not targets:
            return 0.0
        
        dists = [abs(agent.x - t.x) + abs(agent.y - t.y) for t in targets]

        return min(dists)
        
    def _resolve_level(
        self,
        provided_value: Optional[int],
        min_level: int,
        max_level: int,
    ) -> int:
        """
        Returns a valid level value either from the provided config or sampled uniformly.
        """
        if provided_value is not None:
            return int(np.clip(provided_value, min_level, max_level))

        if min_level == max_level:
            return int(min_level)

        if self.np_random is None:
            # Fallback RNG if called outside reset (should not generally happen)
            self.np_random, _ = gymnasium.utils.seeding.np_random()

        return int(self.np_random.integers(min_level, max_level + 1))


    def _get_adjacent_entities(
        self, agent: "_Agent", entity_type: int, entity_list: List
    ) -> List:
        """
        Finds activte entities of a specific type adjacent to the agent.
        """
        adjacent_entities = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # N, S, E, W
            nx, ny = agent.x + dx, agent.y + dy

            # Check bounds
            if not (0 <= nx < self.grid_size[1] and 0 <= ny < self.grid_size[0]):
                continue

            # Check entity type on grid
            if self._grid[ny, nx] == entity_type:
                for entity in entity_list:
                    if entity.active and entity.x == nx and entity.y == ny:
                        adjacent_entities.append(entity)
                        break

        return adjacent_entities
    
    def _is_valid_move(self, x: int, y: int) -> bool:
        """
        Checks if a move is valid (within grid bounds and not occupied by another entity).
        """
        #  Check bounds
        if not (0 <= x < self.grid_size[1] and 0 <= y < self.grid_size[0]):
            return False
        # Check occupation
        if self._grid[y, x] != 0:
            return False
        return True

    
    def _get_info(self) -> Dict[str, Any]:
        return {
            "step": self._step_count,
            "active_enemies": self._active_enemies,
            "active_data_centers": self._active_data_centers,
        }

class _Agent:
    def __init__(self, id: int, x: int, y: int, combat_skill: int, hacking_skill: int):
        self.id = id
        self.x = x
        self.y = y
        self.combat_skill = combat_skill
        self.hacking_skill = hacking_skill

class _Enemy:
    def __init__(self, id: int, x: int, y: int, threat_level: int):
        self.id = id
        self.x = x
        self.y = y
        self.threat_level = threat_level
        self.active = True # Becomes inactive when defeated

class _DataCenter:
    def __init__(self, id: int, x: int, y: int, security_level: int):
        self.id = id
        self.x = x
        self.y = y
        self.security_level = security_level
        self.active = True # Becomes inactive when hacked
