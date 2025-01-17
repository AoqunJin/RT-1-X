from collections import deque
import gym
import gym.spaces
import numpy as np


def stack_and_pad(history: deque, num_obs: int):
    """
    Converts a list of observation dictionaries (`history`) into a single observation dictionary
    by stacking the values. Adds a padding mask to the observation that denotes which timesteps
    represent padding based on the number of observations seen so far (`num_obs`).
    """
    horizon = len(history)
    full_obs = {k: np.stack([dic[k] for dic in history]) for k in history[0]}
    pad_length = horizon - min(num_obs, horizon)
    timestep_pad_mask = np.ones(horizon)
    timestep_pad_mask[:pad_length] = 0
    full_obs["timestep_pad_mask"] = timestep_pad_mask
    return full_obs


def space_stack(space: gym.Space, repeat: int):
    """
    Creates new Gym space that represents the original observation/action space
    repeated `repeat` times.
    """

    if isinstance(space, gym.spaces.Box):
        return gym.spaces.Box(
            low=np.repeat(space.low[None], repeat, axis=0),
            high=np.repeat(space.high[None], repeat, axis=0),
            dtype=space.dtype,
        )
    elif isinstance(space, gym.spaces.Discrete):
        return gym.spaces.MultiDiscrete([space.n] * repeat)
    elif isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict(
            {k: space_stack(v, repeat) for k, v in space.spaces.items()}
        )
    else:
        raise ValueError(f"Space {space} is not supported by Octo Gym wrappers.")


class HistoryWrapper(gym.Wrapper):
    """
    Accumulates the observation history into `horizon` size chunks. If the length of the history
    is less than the length of the horizon, we pad the history to the full horizon length.
    A `timestep_pad_mask` key is added to the final observation dictionary that denotes which timesteps
    are padding.
    """

    def __init__(self, env: gym.Env, horizon: int):
        super().__init__(env)
        self.horizon = horizon

        self.history = deque(maxlen=self.horizon)
        self.num_obs = 0

        self.observation_space = space_stack(self.env.observation_space, self.horizon)

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        self.num_obs += 1
        self.history.append(obs)  # With history
        # self.history[-1] = obs  # w/o history
        assert len(self.history) == self.horizon
        full_obs = stack_and_pad(self.history, self.num_obs)

        return full_obs, reward, done, trunc, info

    # reset with history (black obs)
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.num_obs = 1
        for i in range(self.horizon):
            ap_obs = {k: (np.zeros_like(v) if i < self.horizon - 1 else v) for k, v in obs.items()}
            self.history.append(ap_obs)

        full_obs = stack_and_pad(self.history, self.num_obs)

        return full_obs, info
    
    # reset with history (same obs)
    # def reset(self, **kwargs):
    #     obs, info = self.env.reset(**kwargs)
    #     self.num_obs = 1
    #     self.history.extend([obs] * self.horizon)
    #     full_obs = stack_and_pad(self.history, self.num_obs)

    #     return full_obs, info
