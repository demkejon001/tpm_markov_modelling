import abc
import enum
import numpy as np


class Action(enum.IntEnum):
    left = 3
    up = 0
    right = 1
    down = 2


class GridObject(abc.ABC):
    """An object that can be placed in the GridEnv.

    Subclasses should register themselves in the list of GridObject above."""

    def __init__(self, color, size=0.4):
        """Constructs.

        Args:
            color (str): valid PIL color.
        """
        self._color = color
        self._size = size

    @property
    def color(self):
        return self._color

    @property
    def size(self):
        return self._size


class Wall(GridObject):
    """An object that cannot be passed through."""

    def __init__(self):
        super().__init__("black", 1)

    @property
    def type(self):
        return "Wall"


class Lava(GridObject):
    """An object that cannot be passed through."""

    def __init__(self):
        super().__init__("red", 1)

    @property
    def type(self):
        return "Wall"
    

class Goal(GridObject):
    """An object that cannot be passed through."""

    def __init__(self):
        super().__init__("green", 1)

    @property
    def type(self):
        return "Goal"


class GridEnv:
    """A grid world to move around in.

    The observations are np.ndarrays of shape (height, width, num_objects + 2),
    where obs[y, x, obj_val] = 1 denotes that obj_val is present at (x, y).

    obj_val:
        - value 0, ..., num_objects - 1 denote the objects
        - value num_objects denotes the agent
        - value num_objects + 1 denotes a wall
    """

    def __init__(self, max_steps=20, width=7, height=7):
        """Constructs the environment with dynamics according to env_id.

        Args:
            env_id (int): a valid env_id in TransportationGridEnv.env_ids()
            wrapper (function): see create_env.
            max_steps (int): maximum horizon of a single episode.
        """
        self._max_steps = max_steps
        self._grid = [[None for _ in range(width)] for _ in range(height)]
        self._width = width
        self._height = height

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def agent_pos(self):
        return self._agent_pos

    @property
    def steps_remaining(self):
        return self._max_steps - self._steps

    def text_description(self):
        return "grid"

    def get(self, position):
        return self._grid[position[0]][position[1]]

    def place(self, obj, position, exist_ok=False):
        existing_obj = self.get(position)
        if existing_obj is not None and not exist_ok:
            raise ValueError(
                    "Object {} already exists at {}.".format(existing_obj, position))
        self._grid[position[0]][position[1]] = obj

    def _place_objects(self):
        self._agent_pos = np.array([1, 1])

    def _gen_obs(self):
        obs = np.ones((3, self._height, self._width)).astype(np.float32) * 255
        for x in range(self._width):
            for y in range(self._height):
                obj = self.get((x, y))
                if obj is not None:
                    if isinstance(obj, Wall):
                        obs[:, x, y] = [0, 0, 0]
                    elif isinstance(obj, Lava):
                        obs[:, x, y] = [255, 0, 0]
                    elif isinstance(obj, Goal):
                        obs[:, x, y] = [0, 255, 0]
                    else:
                        raise ValueError

        # Add agent
        obs[:, self._agent_pos[0], self._agent_pos[1]] = [0, 0, 255]
        return obs

    def reset(self):
        self._steps = 0
        self._grid = [[None for _ in range(self.width)]
                      for _ in range(self.height)]
        self._place_objects()
        return self._gen_obs(), {}

    def step(self, action):
        self._steps += 1

        original_pos = np.array(self._agent_pos)
        if action == Action.left:
            self._agent_pos[1] -= 1
        elif action == Action.up:
            self._agent_pos[0] -= 1
        elif action == Action.right:
            self._agent_pos[1] += 1
        elif action == Action.down:
            self._agent_pos[0] += 1

        reward = 0
        terminated = False
        truncated = False

        # Can't walk through wall
        obj = self.get(self._agent_pos)
        if obj is not None and isinstance(obj, Wall):
            self._agent_pos = original_pos
        if obj is not None and isinstance(obj, Lava):
            self._grid[self._agent_pos[0]][self._agent_pos[1]] = None
            reward = -1
            terminated = True
        if obj is not None and isinstance(obj, Goal):
            self._grid[self._agent_pos[0]][self._agent_pos[1]] = None
            reward = 1
            terminated = True

        self._agent_pos[0] = max(min(self._agent_pos[0], self.width - 1), 0)
        self._agent_pos[1] = max(min(self._agent_pos[1], self.height - 1), 0)

        truncated = self._steps == self._max_steps
        return self._gen_obs(), reward, terminated, truncated, {}


class Distshift(GridEnv):
    def __init__(self, max_steps=20, width=7, height=7, shift=False):
        super().__init__(max_steps, width, height)
        self.wall_positions = []
        for i in range(width):
            for j in range(height):
                if i == 0 or i == (width - 1) or j == 0 or j == (height - 1):
                    self.wall_positions.append((i, j))
        self.lava_positions = [(1, 3), (1, 4), (5, 3), (5, 4)]
        if shift:
            self.lava_positions = [(1, 3), (1, 4), (2, 3), (2, 4)]
        self.goal_positions = [(1, 5)]
        
    def get_start_positions(self):
        self.reset()
        start_positions = []
        for x in range(self._width):
            for y in range(self._height):
                obj = self.get((x, y))
                if obj is None:
                    start_positions.append((x, y))
        return start_positions        

    def _place_objects(self):
        super()._place_objects()
        for (r, c) in self.wall_positions:
            self.place(Wall(), (r, c))
        for (r, c) in self.lava_positions:
            self.place(Lava(), (r, c))
        for (r, c) in self.goal_positions:
            self.place(Goal(), (r, c))

    def reset(self, agent_start_pos=(1, 1)):
        self._steps = 0
        self._grid = [[None for _ in range(self.width)]
                      for _ in range(self.height)]
        self._place_objects()
        self._agent_pos = np.array(agent_start_pos)
        return self._gen_obs(), {}
    
    
def get_vertical_lava_combinations(lava_length=2):
    vertical_lava_combinations = []
    for x in range(1, 7-lava_length):
        for y in range(1, 6):
            if (x, y) != (1, 1) and (x, y) != (1, 5):
                vertical_lava = []
                for i in range(lava_length):
                    vertical_lava.append((x + i, y))
                vertical_lava_combinations.append(vertical_lava)
    return vertical_lava_combinations


def get_horizontal_lava_combinations(lava_length=2):
    horizontal_lava_combinations = []
    for x in range(1, 6):
        for y in range(1, 7-lava_length):
            if (x, y) != (1, 1) and (x, y+1) != (1, 5):
                horizontal_lava = []
                for i in range(lava_length):
                    horizontal_lava.append((x, y + i))
                horizontal_lava_combinations.append(horizontal_lava)
    return horizontal_lava_combinations
    
    
class RandomLavaDistshift(Distshift):
    def __init__(self, seed=42, n_lava=2, lava_length=2):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        lava_positions = self.get_lava_positions(lava_length)
        self.set_lava_positions(lava_positions, n_lava)
        
    def get_lava_positions(self, lava_length):
        raise NotImplementedError
                    
    def set_lava_positions(self, lava_positions, n_lava):
        self.lava_positions = []
        selected_lava_positions = set()
        for lava_idx in self.rng.choice(len(lava_positions), size=n_lava, replace=False):
            for lava_position in lava_positions[lava_idx]:
                selected_lava_positions.add(lava_position)
        self.lava_positions = list(selected_lava_positions)
        
        
class RandomLength2LavaDistshift(RandomLavaDistshift):
    def get_lava_positions(self, lava_length):
        lava_positions = []
        lava_positions += get_vertical_lava_combinations(lava_length)
        lava_positions += get_horizontal_lava_combinations(lava_length)
        return lava_positions
    
    
class RandomVerticalLavaDistshift(RandomLavaDistshift):
    def get_lava_positions(self, lava_length):
        lava_positions = []
        lava_positions += get_vertical_lava_combinations(lava_length)
        return lava_positions
    
        
class RandomHorizontalLavaDistshift(RandomLavaDistshift):
    def get_lava_positions(self, lava_length):
        lava_positions = []
        lava_positions += get_horizontal_lava_combinations(lava_length)
        return lava_positions

        
class RandomLength1LavaDistshift(RandomLavaDistshift):
    def get_lava_positions(self, lava_length):
        lava_positions = []
        for x in range(1, 6):
            for y in range(1, 6):
                if (x, y) != (1, 1) and (x, y) != (1, 5):
                    lava_positions.append([(x, y)])
        return lava_positions
