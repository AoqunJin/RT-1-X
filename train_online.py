import copy
import jax
from jax.experimental import mesh_utils
from jax.sharding import NamedSharding
import optax
import numpy as np
import optax
import functools
import tensorflow_hub as hub
import wandb

from envs.metaworld_env import MetaworldEnv, _env_dict
from envs.mw_tools import POLICIES
from utils.gym_wrappers import HistoryWrapper
from data.transform import convert_dtype_and_crop_images
from utils.model_utils import save_model, load_model
from utils.jax_utils import reshard, configure_jax
from models.rt1 import RT1
from utils.train_utils import (
    create_train_state,
    train
)
from utils.eval_utils import (
    RT1Policy,
    get_eval_sample_batch
)


# Constants
SEQUENCE_LENGTH = 15
NUM_ACTION_TOKENS = 11
LAYER_SIZE = 256
VOCAB_SIZE = 512
NUM_IMAGE_TOKENS = 81
PER_DEVICE_MICRO_BATCH_SIZE = 2  # 4-> 48 GB | 2-> 24 GB
GRADIENT_ACCUMULATION_STEPS = 4  # 2 * 4 = 8
LEARNING_RATE = 1e-4
CHECKPOINT_LOAD_DIR = "/path/to/models/rt_1_x_jax"
NUM_TRAIN_STEPS = 40_000  # actual should be > 1M TODO
SAVE_CHECKPOINT_EVERY_STEPS = 40_000  # TODO
LOG_LOSS_EVERY_STEPS = 100  # TODO
BENCHMARK = _env_dict.ML10_V2
TEST_TYPE = "train"  # ["train" | "test"]
DECODER_TYPE = "kpt"  # ["greedy" | "kpt"]
WANDB_PROJECT_NAME = "metaworld_train"
WANDB_RUN_NAME = "ml10_50e"


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

    optimizer = optax.adam(learning_rate=1e-4, eps=1e-7)
    
    if GRADIENT_ACCUMULATION_STEPS > 1:
        optimizer = optax.MultiSteps(
            optimizer, GRADIENT_ACCUMULATION_STEPS
        )
    
    agent_create_train_state = functools.partial(
        create_train_state, model=rt1x_model, optimizer=optimizer
    )

    create_train_state_jit = jax.jit(
        agent_create_train_state,
        out_shardings=replicate_sharding,
    )

    rng = jax.random.PRNGKey(0)  # Pseudo-random number generator
    rng, agent_rng = jax.random.split(rng)
    state = create_train_state_jit(batch=sample_batch, rng=agent_rng)
        
    agent_train = functools.partial(train, model=rt1x_model, optimizer=optimizer)
    jitted_train_step = jax.jit(
        agent_train,
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
    #                             Model training.                             #
    ###########################################################################
    
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    policy = RT1Policy(
        rt1x_model,
        {"params": state_repl.params, "batch_stats": state_repl.batch_stats},
        seqlen=15,
    )
    
    for name in BENCHMARK[TEST_TYPE].keys():
        env = MetaworldEnv(name)
        env = HistoryWrapper(env, horizon=15)
        expert = POLICIES[name]()

        # running rollouts
        total_return = 0
        total_accuracy = 0
        for i in range(500):
            obs, info = env.reset()

            # create task specification --> use model utility to create task dict with correct entries
            language_instruction = env.get_task()[
                "language_instruction"
            ]  # ['pick up the cube and hand it over']
            embedding = embed(language_instruction)  # [1, 512]
            natural_language_embedding = np.zeros((15, 512))  # [15, 512]
            action_labels = np.zeros((15, 4))
            terminate_episode = np.zeros((15, 3))
            terminate_episode[:, 1] = 1  # TODO 
            metrics_train_sum = None
            images = []
            episode_return = 0.0

            for j in range(500):
                ###########################################################################
                #                             Model inference.                            #
                ###########################################################################
                
                images.append(obs["image_primary"][-1])

                natural_language_embedding = np.vstack(
                    (natural_language_embedding[1:], copy.deepcopy(embedding))
                )
                observation = {
                    "image": convert_dtype_and_crop_images(
                        copy.deepcopy(obs["image_primary"]), (300, 300), training=False
                    ).numpy(),
                    "natural_language_embedding": natural_language_embedding,
                }
                # print(observation['image'].shape, observation['natural_language_embedding'].shape)
                model_output = policy.action(observation, decoder_type=DECODER_TYPE)
                agent_action = np.concatenate(
                    (
                        model_output["world_vector"] / 2,
                        model_output["gripper_closedness_action"],
                    )
                )

                ###########################################################################
                #                             Model training.                             #
                ###########################################################################
                
                expert_action = np.clip(expert.get_action(info["state"]), -1, 1)
                action_labels = np.vstack((action_labels[1:], copy.deepcopy(expert_action)))
                batch = {
                    "observation": {
                        "image": np.expand_dims(observation["image"], axis=0),
                        "natural_language_embedding": np.expand_dims(
                            observation["natural_language_embedding"], axis=0
                        ),
                    },
                    "action": {
                        "base_displacement_vector": np.zeros(
                            (1, 15, 2), dtype=np.float32
                        ),
                        "base_displacement_vertical_rotation": np.zeros(
                            (1, 15, 1), dtype=np.float32
                        ),
                        "gripper_closedness_action": np.expand_dims(
                            action_labels[:, 3:4], axis=0
                        ),
                        "rotation_delta": np.zeros((1, 15, 3), dtype=np.float32),
                        "terminate_episode": np.expand_dims(
                            copy.deepcopy(terminate_episode), axis=0
                        ),
                        "world_vector": np.expand_dims(action_labels[:, :3], axis=0),
                    },
                }
                batch = jax.tree_map(_form_gda, batch, global_data_shape)
                rng_repl = jax.random.fold_in(rng_repl, j)
                state_repl, metrics_train_update = jitted_train_step(
                    state=state_repl, batch=batch, rng=rng_repl
                )
                
                if metrics_train_sum is None: metrics_train_sum = metrics_train_update
                else: metrics_train_sum = jax.tree_map(lambda x, y: x + y, metrics_train_sum, metrics_train_update)
                ###########################################################################
                
                obs, reward, done, trunc, info = env.step(agent_action)
                episode_return += reward
                if done or trunc:
                    break
            
            metrics_train_update = jax.tree_map(lambda x: x / j, metrics_train_sum)
            metrics_train_update = jax.device_get(metrics_train_update)
            metrics_train_update.update({
                "episode_return": episode_return,
                "done": bool(done)
            })
            print(metrics_train_update)
            # log rollout video to wandb -- subsample temporally 2x for faster logging
            if i % 5 == 0:
                wandb.log({"rollout_video": wandb.Video(np.array(images).transpose(0, 3, 1, 2)[::10])})
                
                
if __name__ == "__main__":
    main()
    