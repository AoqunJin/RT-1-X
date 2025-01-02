from typing import Any
import functools
import copy

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import NamedSharding
import flax
from flax.training import checkpoints
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import wandb

from models.rt1 import RT1, tokenize_action, detokenize_action

from envs.metaworld_env import MetaworldEnv, _env_dict
from envs.mw_tools import POLICIES
from utils.gym_wrappers import HistoryWrapper
from utils.visualization_lib import save_images_vertically_with_border
from data.transform import convert_dtype_and_crop_images

# Constants
SEQUENCE_LENGTH = 15
NUM_ACTION_TOKENS = 11
LAYER_SIZE = 256
VOCAB_SIZE = 512
NUM_IMAGE_TOKENS = 81
CHECKPOINT_LOAD_DIR = "/home/sora/workspace/models/rt_1_x_jax_metaworld_ml10_20e/checkpoint_9999"
WANDB_PROJECT_NAME = "rt_1_x_jax"
WANDB_RUN_NAME = "eval_metaworld_ml10_20e_train_10k"
BENCHMARK = _env_dict.ML10_V2
TEST_TYPE = "train"


def main():

    for name in BENCHMARK[TEST_TYPE].keys():

        env = MetaworldEnv(name)
        env = HistoryWrapper(env, horizon=15)
        expert = POLICIES[name]()

        # running rollouts
        total_return = 0
        total_accuracy = 0
        for i in range(20):
            obs, info = env.reset()
            images = []
            episode_return = 0.0

            for j in range(500):
                images.append(obs["image_primary"]) # [15, h, w, 3]
                action = np.clip(expert.get_action(info["state"]), -1, 1)
                obs, reward, done, trunc, info = env.step(action)

                episode_return += reward
                if done or trunc: break

            save_images_vertically_with_border(np.array(images), border_size=10, output_path=f"./tmp/{name}_{i}.png")
            total_return += episode_return
            total_accuracy += int(done)

        print(f"Environment: {name}, Average return: {total_return / 20}, Average accuracy: {total_accuracy / 20}")

if __name__ == "__main__":
    main()
