import functools
import copy

import jax
from jax.experimental import mesh_utils
from jax.sharding import NamedSharding
import numpy as np
import tensorflow_hub as hub
import wandb
from tqdm import tqdm

from envs.metaworld_env import MetaworldEnv, _env_dict
from envs.mw_tools import POLICIES
from utils.gym_wrappers import HistoryWrapper
from data.transform import convert_dtype_and_crop_images
from models.rt1 import RT1
from utils.model_utils import load_model
from utils.jax_utils import reshard, configure_jax
from utils.eval_utils import (
    evaluate,
    create_eval_state,
    RT1Policy,
    get_eval_sample_batch
)


# Constants
SEQUENCE_LENGTH = 15
NUM_ACTION_TOKENS = 11
LAYER_SIZE = 256
VOCAB_SIZE = 512
NUM_IMAGE_TOKENS = 81
DECODER_TYPE = "kpt"  # ["greedy" | "kpt"]
CHECKPOINT_LOAD_DIR = "/path/to/models/rt_1_x_jax_output/checkpoint_x"
# EMBED_LOAD_DIR = "/path/to/models/universal-sentence-encoder-tensorflow2-large-v2"
EMBED_LOAD_DIR = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
WANDB_PROJECT_NAME = "metaworld_eval"
WANDB_RUN_NAME = "ml10_50e"
BENCHMARK = _env_dict.ML10_V2
TEST_TYPE = "train"  # ["train" | "test"]


def main():
    wandb.init(name=WANDB_RUN_NAME, project=WANDB_PROJECT_NAME, mode="offline")
    configure_jax()

    def _put_to_devices(x):
        per_device_arrays = np.split(x, local_device_count, axis=0)
        return jax.device_put(per_device_arrays, local_devices)

    def _form_gda(local_data, global_shape):
        arrays = _put_to_devices(local_data)
        return jax.make_array_from_single_device_arrays(global_shape, sharding, arrays)

    ###########################################################################
    #                       Prepare the dataset.                              #
    ###########################################################################
    # To initialize weights for the model and run a forward pass.
    sample_batch = get_eval_sample_batch()
    global_batch_size = 1

    ###########################################################################
    #                    Creating mesh and shardings.                         #
    ###########################################################################

    num_devices = len(jax.devices())
    mesh = jax.sharding.Mesh(mesh_utils.create_device_mesh((num_devices,)), ("data",))
    local_devices = mesh.local_devices
    local_device_count = jax.local_device_count()
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    replicate_sharding = NamedSharding(mesh, jax.sharding.PartitionSpec())

    global_data_shape = jax.tree_map(
        lambda x: (global_batch_size,) + x.shape[1:], sample_batch
    )
    sample_batch = jax.tree_map(_form_gda, sample_batch, global_data_shape)

    ###########################################################################
    #                          Prepare the model.                             #
    ###########################################################################

    rt1x_model = RT1(
        num_image_tokens=NUM_IMAGE_TOKENS,
        num_action_tokens=NUM_ACTION_TOKENS,
        layer_size=LAYER_SIZE,
        vocab_size=VOCAB_SIZE,
        # Use token learner to reduce tokens per image to 81.
        use_token_learner=True,
        # RT-1-X uses (-2.0, 2.0) instead of (-1.0, 1.0).
        world_vector_range=(-2.0, 2.0),
    )

    agent_create_eval_state = functools.partial(create_eval_state, model=rt1x_model)

    agent_create_eval_state_jit = jax.jit(
        agent_create_eval_state,
        out_shardings=replicate_sharding,
    )

    rng = jax.random.PRNGKey(0)  # Pseudo-random number generator
    rng, agent_rng = jax.random.split(rng)
    state = agent_create_eval_state_jit(batch=sample_batch, rng=agent_rng)

    # Create the eval step.
    agent_eval = functools.partial(evaluate, model=rt1x_model)
    jitted_eval_step = jax.jit(
        agent_eval,
        out_shardings=(replicate_sharding, replicate_sharding),
    )

    # The state should be resharded since we may have loaded pretrained weights
    # that need to be converted to jax.Arrays.
    state_repl = reshard(state, shardings=replicate_sharding)
    # The RNG must be replicated.
    rng_repl = reshard(rng, shardings=replicate_sharding)

    # Load model
    if CHECKPOINT_LOAD_DIR:
        print(f"Loading model from {CHECKPOINT_LOAD_DIR}")
        state_repl = load_model(CHECKPOINT_LOAD_DIR, state_repl)

    ###########################################################################
    #                            Model evaluating.                            #
    ###########################################################################

    embed = hub.load(EMBED_LOAD_DIR)
    policy = RT1Policy(
        rt1x_model,
        {"params": state_repl.params, "batch_stats": state_repl.batch_stats},
        seqlen=15,
        decoder_type=DECODER_TYPE
    )

    for name in BENCHMARK[TEST_TYPE].keys():
        env = MetaworldEnv(name)
        env = HistoryWrapper(env, horizon=15)
        # expert = POLICIES[name]()

        # running rollouts
        total_return = 0
        total_accuracy = 0
        for i in tqdm(range(50)):
            obs, info = env.reset()

            # create task specification --> use model utility to create task dict with correct entries
            language_instruction = env.get_task()[
                "language_instruction"
            ]  # ['pick up the cube and hand it over']
            embedding = embed(language_instruction)  # [1, 512]
            natural_language_embedding = np.zeros((15, 512))  # [15, 512]
            terminate_episode = np.zeros((15, 3))
            terminate_episode[:, 1] = 1

            images = []
            episode_return = 0.0

            for j in range(500):
                images.append(obs["image_primary"][-1])

                natural_language_embedding = np.vstack(
                    (natural_language_embedding[1:], copy.deepcopy(embedding))
                )
                observation = {
                    "image": convert_dtype_and_crop_images(
                        copy.deepcopy(obs["image_primary"]), (300, 300), training=False
                    ).numpy(),
                    "natural_language_embedding": natural_language_embedding,
                    # 'natural_language_embedding': jnp.ones((15, 512)),
                }
                # print(observation['image'].shape, observation['natural_language_embedding'].shape)
                model_output = policy.action(observation)
                agent_action = np.concatenate(
                    (
                        model_output["world_vector"] / 2,
                        model_output["gripper_closedness_action"],
                    )
                )
                obs, reward, done, trunc, info = env.step(agent_action)
                # print('Agent action:', agent_action)
                episode_return += reward
                if done or trunc:
                    break

            total_return += episode_return
            total_accuracy += int(done)

            # log rollout video to wandb -- subsample temporally 2x for faster logging
            if i % 5 == 0:
                wandb.log({"rollout_video": wandb.Video(np.array(images).transpose(0, 3, 1, 2)[::10])})

            # print('Trunc:', trunc, 'Done:', done)

        print(f"Environment: {name}, Average return: {total_return / 50}, Average accuracy: {total_accuracy / 50}")
        wandb.log({name: {"average_return": total_return / 50, "average_accuracy": total_accuracy / 50,}})


if __name__ == "__main__":
    main()
