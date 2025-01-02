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


def configure_jax():
    """Configures JAX settings for GPU memory and backend."""
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.Session(config=config)
    jax.config.update("jax_platform_name", "gpu")
    jax.config.update("jax_xla_backend", "cuda")


def reshard(tree, shardings):
    """Take an arbitrarily sharded pytree and shard it according to `shardings`.

    From `big_vision.utils.reshard`. See that doc for full details.

    Args:
      tree: a pytree of arrays.
      shardings: a (prefix) pytree of jax array shardings.

    Returns:
      A pytree of global jax arrays that follows provided shardings.
    """

    def _make_global_arr(x, shard, shape):
        # Avoid unnecessary copies and transfers:
        if hasattr(x, "sharding") and x.sharding.is_equivalent_to(
            shard, len(shape)
        ):  # pylint: disable=line-too-long
            return x
        if not getattr(x, "is_fully_addressable", True):
            raise RuntimeError(
                "Trying to reshard a non-fully-addressable array. "
                "Please see the doc-comment for detailed explanation."
            )
        x = jax.device_get(x)  # Might be on local devices.
        xs = [
            jax.device_put(x[s], device=d)
            for d, s in shard.addressable_devices_indices_map(shape).items()
        ]
        return jax.make_array_from_single_device_arrays(shape, shard, xs)

    shapes = jax.tree_map(np.shape, tree)
    shardings = tree_broadcast(shardings, tree)
    return jax.tree_map(_make_global_arr, tree, shardings, shapes)


def tree_broadcast(prefix, target):
    """Broadcasts a prefix tree to a full tree.

    See big_vision.utils.tree_broadcast.

    Args:
      prefix: prefix pytree.
      target: boradcast target for a prefix tree.

    Returns:
      prefix tree broadcasted to a target tree.
    """

    def _broadcast(leaf, subtree):
        return jax.tree_map(lambda _: leaf, subtree)

    return jax.tree_map(_broadcast, prefix, target)


@flax.struct.dataclass
class EvalState:
    step: int
    params: Any
    batch_stats: Any


def create_eval_state(model, batch, rng):
    """Creates the train state and initial metrics for agent."""
    obs_input = batch["observation"]
    act_input = batch["action"]

    rng, rng2, rng3 = jax.random.split(rng, 3)
    variables = model.init(
        {"params": rng, "random": rng3},
        obs=obs_input,
        act=act_input,
        train=False,
    )

    params = flax.core.unfreeze(variables["params"])
    batch_stats = flax.core.unfreeze(variables["batch_stats"])

    eval_state = EvalState(
        step=0,
        params=flax.core.unfreeze(params),
        batch_stats=batch_stats,
    )
    return eval_state


def evaluate(batch, state, model, rng):
    """Performs a single training step."""
    rng, loss_rng = jax.random.split(rng)

    def loss_fn(params):
        variables = {"params": params, "batch_stats": state.batch_stats}
        per_example_loss, accuracy, new_variables = rt1_loss(
            model, batch=batch, variables=variables, rng=loss_rng
        )
        loss = jnp.mean(per_example_loss)
        return loss, (accuracy, new_variables["batch_stats"])

    (loss, (accuracy, new_batch_stats)) = loss_fn(state.params)

    loss = jnp.mean(loss)

    metrics_update = {
        "loss": loss, "accuracy": accuracy
    }
    return state, metrics_update


def rt1_loss(
    model,
    batch,
    variables,
    rng,
):
    """Implements the RT-1 loss."""
    observation = batch["observation"]
    action = batch["action"]

    bs = observation["image"].shape[0]
    seqlen = observation["image"].shape[1]

    # First, we encode the observations using the model.encode method.
    # This will give us an observation encoding (for the entire sequence).
    rng, params_rng = jax.random.split(rng)
    rng, dropout_rng = jax.random.split(rng)
    rng, sd_rng = jax.random.split(rng)
    rng, random_rng = jax.random.split(rng)
    logits, new_variables = model.apply(
        variables,
        obs=observation,
        act=action,
        train=True,
        mutable=["batch_stats"],
        rngs={
            "params": params_rng,
            "dropout": dropout_rng,
            "random": random_rng,
        },
    )

    vocab_size = model.vocab_size

    # `action` is dict of (B, T, ...), we combine actions into B*T batch to
    # tokenize.
    action = jax.tree_map(lambda x: jnp.reshape(x, (bs * seqlen, -1)), action)
    labels = tokenize_action(action, vocab_size=vocab_size, world_vector_range=model.world_vector_range)
    labels = jax.tree_map(lambda x: jnp.reshape(x, (bs, seqlen, -1)), labels)
    labels = labels[:, :, :, None]  # labels should be (B, seqlen, 11, 1)

    # Get num_action_tokens tokens for the action prediction. By default,
    # RT-1 computes the loss for all `seqlen * num_action_tokens`, not just
    # the final timestep's action.
    # In the default RT-1 setup (8 img, 11 act tokens), we have:
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    # |-----image tokens------|-------------action tokens--------------|
    #                      |----------------logits------------------|
    # For each time step, we want [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] for
    # the logits, for the "next token" prediction.
    num_image_tokens = model.num_image_tokens
    num_action_tokens = model.num_action_tokens
    time_step_tokens = num_image_tokens + num_action_tokens
    logits = jnp.reshape(logits, (bs, seqlen, time_step_tokens, vocab_size))
    logits = logits[:, :, num_image_tokens - 1 : -1]

    logp = jax.nn.log_softmax(logits)
    loglik = jnp.take_along_axis(logp, labels, axis=-1)
    loglik = jnp.mean(loglik, axis=(1, 2, 3))
    
    # Compute accuracy
    predictions = jnp.argmax(logits, axis=-1)
    correct = jnp.equal(predictions, jnp.squeeze(labels, axis=-1))
    accuracy = jnp.mean(correct.astype(jnp.float32))

    return -loglik, accuracy, new_variables


# Save function
def save_model(state, save_dir, step):
    checkpoints.save_checkpoint(
        ckpt_dir=save_dir,
        target=state,  # Save the training state
        step=step,
        overwrite=True,
    )
    print(f"Model saved at step {step} in {save_dir}")


# Load function
def load_model(save_dir, state):
    restored_state = checkpoints.restore_checkpoint(
        ckpt_dir=save_dir, target=state  # Initialize with the same structure
    )
    if restored_state == state:
        raise ValueError(f"Failed to load model from '{save_dir}'. The state is the same as the default one.")
    print(f"Model restored from {save_dir}")
    return restored_state


class RT1Policy:
    """Runs inference with a RT-1 policy."""

    def __init__(self, model, variables=None, seqlen=15, rng=None):
        self.model = model
        self.variables = variables
        self.seqlen = seqlen
        if rng is None:
            self.rng = jax.random.PRNGKey(0)
        else:
            self.rng = rng
        self._run_action_inference_jit = jax.jit(self._run_action_inference)

    def _run_action_inference(self, observation, rng):
        """A jittable function for running inference."""

        # We add zero action tokens so that the shape is (seqlen, 11).
        # Note that in the vanilla RT-1 setup, where
        # `include_prev_timesteps_actions=False`, the network will not use the
        # input tokens and instead uses zero action tokens, thereby not using the
        # action history. We still pass it in for simplicity.
        act_tokens = jnp.zeros((1, 6, 11))

        # Add a batch dim to the observation.
        batch_obs = jax.tree_map(lambda x: jnp.expand_dims(x, 0), observation)

        _, random_rng = jax.random.split(rng)

        output_logits = self.model.apply(
            self.variables,
            batch_obs,
            act=None,
            act_tokens=act_tokens,
            train=False,
            rngs={"random": random_rng},
        )

        time_step_tokens = self.model.num_image_tokens + self.model.num_action_tokens
        output_logits = jnp.reshape(
            output_logits, (1, self.seqlen, time_step_tokens, -1)
        )
        action_logits = output_logits[:, -1, ...]
        action_logits = action_logits[:, self.model.num_image_tokens - 1 : -1]

        action_logp = jax.nn.softmax(action_logits)
        action_token = jnp.argmax(action_logp, axis=-1)

        # Detokenize the full action sequence.
        detokenized = detokenize_action(
            action_token, self.model.vocab_size, self.model.world_vector_range
        )

        detokenized = jax.tree_map(lambda x: x[0], detokenized)

        return detokenized

    def action(self, observation):
        """Outputs the action given observation from the env."""
        # Assume obs has no batch dimensions.
        observation = copy.deepcopy(observation)

        # Jax does not support string types, so remove it from the dict if it
        # exists. TODO natural_language_embedding
        if "natural_language_instruction" in observation:
            del observation["natural_language_instruction"]

        # image = observation['image']
        # # Resize using TF image resize to avoid any issues with using different
        # # resize implementation, since we also use tf.image.resize in the data
        # # pipeline. Also scale image to [0, 1].
        # image = tf.image.resize(image, (300, 300)).numpy()
        # image /= 255.0
        # observation['image'] = image

        self.rng, rng = jax.random.split(self.rng)
        action = self._run_action_inference_jit(observation, rng)
        action = jax.device_get(action)

        # Use the base pose mode if the episode if the network outputs an invalid
        # `terminate_episode` action.
        if np.sum(action["terminate_episode"]) == 0:
            action["terminate_episode"] = np.zeros_like(action["terminate_episode"])
            action["terminate_episode"][-1] = 1
        return action


def main():
    wandb.init(name=WANDB_RUN_NAME, project=WANDB_PROJECT_NAME)
    configure_jax()

    def _put_to_devices(x):
        per_device_arrays = np.split(x, local_device_count, axis=0)
        return jax.device_put(per_device_arrays, local_devices)

    def _form_gda(local_data, global_shape):
        arrays = _put_to_devices(local_data)
        return jax.make_array_from_single_device_arrays(global_shape, sharding, arrays)

    ##################################################################################################################
    #                                     Prepare the dataset.                                                       #
    ##################################################################################################################
    # Initialize random weights for the model and run a forward pass.
    obs = {
        "image": jnp.ones((1, 15, 300, 300, 3)),
        "natural_language_embedding": jnp.ones((1, 15, 512)),
    }
    act = {
        "world_vector": jnp.ones((1, 15, 3)),
        "rotation_delta": jnp.ones((1, 15, 3)),
        "gripper_closedness_action": jnp.ones((1, 15, 1)),
        "base_displacement_vertical_rotation": jnp.ones((1, 15, 1)),
        "base_displacement_vector": jnp.ones((1, 15, 2)),
        "terminate_episode": jnp.ones((1, 15, 3), dtype=jnp.int32),
    }
    sample_batch = {"observation": obs, "action": act}
    global_batch_size = 1

    ##################################################################################################################
    #                                    Creating mesh and shardings.                                                #
    ##################################################################################################################

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

    ##################################################################################################################
    #                                     Prepare the model.                                                         #
    ##################################################################################################################

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

    ##################################################################################################################
    #                                     Model evaluating.                                                          #
    ##################################################################################################################

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
        for i in range(20):
            obs, info = env.reset()

            # create task specification --> use model utility to create task dict with correct entries
            language_instruction = env.get_task()[
                "language_instruction"
            ]  # ['pick up the cube and hand it over']
            embedding = embed(language_instruction)  # [1, 512]
            natural_language_embedding = np.zeros((15, 512))  # [15, 512]
            action_labels = np.zeros((15, 4))
            terminate_episode = np.zeros((15, 3))
            terminate_episode[:, 1] = 1

            images = []
            episode_return = 0.0

            for j in range(500):
                images.append(obs["image_primary"][0])

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

                ######################################################################################################
                #                                      Expert inference.                                             #
                ######################################################################################################

                # TODO Simulate expert action
                action = np.clip(expert.get_action(info["state"]), -1, 1)
                action_labels = np.vstack((action_labels[1:], copy.deepcopy(action)))
                # print('Expert action:', action, 'Agent action:', agent_action)
                # print('Action diff:', action - agent_action)

                # obs, reward, done, trunc, info = env.step(action)

                # episode_return += reward
                # if done or trunc: break

                ######################################################################################################
                #                                           Loss.                                                    #
                ######################################################################################################
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

                _, metrics_update = jitted_eval_step(
                    state=state_repl, batch=batch, rng=rng_repl
                )
                print(metrics_update)
                # print(model_output)
                ######################################################################################################

            total_return += episode_return
            total_accuracy += int(done)

            # log rollout video to wandb -- subsample temporally 2x for faster logging
            if i % 5 == 0:
                wandb.log({"rollout_video": wandb.Video(np.array(images).transpose(0, 3, 1, 2)[::10])})

            # print('Trunc:', trunc, 'Done:', done)

        print(f"Environment: {name}, Average return: {total_return / 20}, Average accuracy: {total_accuracy / 20}")
        wandb.log({name: {"average_return": total_return / 20, "average_accuracy": total_accuracy / 20,}})


if __name__ == "__main__":
    main()
