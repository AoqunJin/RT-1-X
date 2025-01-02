from typing import Any
import os
import flax
from flax.training import checkpoints
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import NamedSharding
import optax
import numpy as np
import optax
import functools
import tensorflow as tf
import wandb

from data.datasets import (
    get_trajectory_dataset,
    DATASET_NAME_TO_TRAJECTORY_DATASET_KWARGS,
)
from data.transform import prepare_for_model_input
from models.rt1 import RT1, tokenize_action
from utils.visualization_lib import save_images_vertically_with_border, save_batch

# Constants
SEQUENCE_LENGTH = 15
NUM_ACTION_TOKENS = 11
LAYER_SIZE = 256
VOCAB_SIZE = 512
NUM_IMAGE_TOKENS = 81
PER_DEVICE_BATCH_SIZE = 4
LEARNING_RATE = 1e-4
CHECKPOINT_LOAD_DIR = "/home/sora/workspace/models/rt_1_x_jax"
CHECKPOINT_OUT_DIR = "/home/sora/workspace/models/rt_1_x_jax_metaworld_ml10_20e"
NUM_TRAIN_STEPS = 10_000  # actual should be > 1M
SAVE_CHECKPOINT_EVERY_STEPS = 5_000
LOG_LOSS_EVERY_STEPS = 100
WANDB_PROJECT_NAME = "rt_1_x_jax"
WANDB_RUN_NAME = "train_metaworld_ml10_20e"
DATASET_NAME_TO_WEIGHTS = {
    "rt_1_metaworld_ml10_20e": 100,
    # "rt_1_metaworld_ml10_40e": 100,
    # "rt_1_metaworld_ml10_100e": 100,
    # "rt_1_metaworld_ml45_20e": 100
}

def configure_jax():
    """Configures JAX settings for GPU memory and backend."""
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.Session(config=config)
    jax.config.update("jax_platform_name", "gpu")
    jax.config.update("jax_xla_backend", "cuda")


def initialize_datasets():
    """Initializes and prepares datasets for training."""

    DATASET_NAME_TO_TRAJECTORY_DATASET = {
        k: get_trajectory_dataset(**DATASET_NAME_TO_TRAJECTORY_DATASET_KWARGS[k])
        for k in DATASET_NAME_TO_WEIGHTS.keys()
    }
    datasets = [
        dataset.shuffle(10) for dataset in DATASET_NAME_TO_TRAJECTORY_DATASET.values()
    ]
    weights = [
        float(DATASET_NAME_TO_WEIGHTS[name])
        for name in DATASET_NAME_TO_TRAJECTORY_DATASET.keys()
    ]

    train_dataset = tf.data.Dataset.sample_from_datasets(datasets, weights=weights)
    train_dataset = prepare_for_model_input(
        train_dataset, target_height=300, target_width=300, training=True
    )
    return train_dataset


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
class TrainState:
    step: int
    params: Any
    opt_state: optax.OptState
    batch_stats: Any


def create_train_state(model, batch, rng, optimizer):
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

    train_state = TrainState(
        step=0,
        params=flax.core.unfreeze(params),
        opt_state=optimizer.init(params),
        batch_stats=batch_stats,
    )
    return train_state


def train(batch, state, model, optimizer, rng):
    """Performs a single training step."""
    rng, loss_rng = jax.random.split(rng)

    def loss_fn(params):
        variables = {"params": params, "batch_stats": state.batch_stats}
        per_example_loss, accuracy, new_variables = rt1_loss(
            model, batch=batch, variables=variables, rng=loss_rng
        )
        loss = jnp.mean(per_example_loss)
        return loss, (accuracy, new_variables["batch_stats"])

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (accuracy, new_batch_stats)), grad = grad_fn(state.params)

    loss = jnp.mean(loss)

    updates, new_opt_state = optimizer.update(grad, state.opt_state, state.params)

    new_params = optax.apply_updates(state.params, updates)
    new_state = state.replace(
        step=state.step + 1,
        params=flax.core.unfreeze(new_params),
        opt_state=flax.core.unfreeze(new_opt_state),
        batch_stats=flax.core.unfreeze(new_batch_stats),
    )

    metrics_update = {
        "loss": loss, "accuracy": accuracy
    }
    return new_state, metrics_update


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
    train_dataset = initialize_datasets()

    global_batch_size = jax.device_count() * PER_DEVICE_BATCH_SIZE
    local_batch_size = jax.local_device_count() * PER_DEVICE_BATCH_SIZE

    train_dataset = train_dataset.repeat(-1)  # repeated indefinitely | repeat(num_epochs)
    # Larger shuffle buffer leads to better performance, but consumes more RAM
    train_dataset = train_dataset.shuffle(buffer_size=10)
    train_dataset = train_dataset.batch(local_batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    train_iter = train_dataset.as_numpy_iterator()
    sample_batch = jax.tree_map(lambda x: x, next(train_iter))

    print(f"Local batch size: {local_batch_size}")
    print(f"Global batch size: {global_batch_size}")
    print(f"Devices: {jax.devices()}")
    print(f"Sample batch keys: {sample_batch.keys()}")
    def print_shapes(data, prefix=""):
        if isinstance(data, dict):
            for key, value in data.items():
                print_shapes(value, prefix=f"{prefix}{key}.")
        elif isinstance(data, (jax.Array, np.ndarray)):
            print(f"{prefix}shape: {data.shape}")
        else: print(f"{prefix}type: {type(data)}")
    print_shapes(sample_batch)
    # print(sample_batch['action']['world_vector'])
    # print(sample_batch['observation']['natural_language_embedding'])
    # save_images_vertically_with_border(sample_batch["observation"]["image"], border_size=10, output_path="sample_batch.png")
    save_batch(sample_batch, output_dir="sample_batch")
    
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

    optimizer = optax.adam(learning_rate=1e-4, eps=1e-7)

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

    # Create the train step.
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

    ##################################################################################################################
    #                                     Model training.                                                            #
    ##################################################################################################################
    # from eval import RT1Policy
    
    # policy = RT1Policy(
    #     model=rt1x_model,
    #     variables={"params": state_repl.params, "batch_stats": state_repl.batch_stats}, 
    #     seqlen=15
    # )
    
    for step in range(NUM_TRAIN_STEPS):
        try:
            batch = next(train_iter)
        except StopIteration as e:
            break

        batch = jax.tree_map(_form_gda, batch, global_data_shape)

        is_last_step = (step + 1 == NUM_TRAIN_STEPS)

        rng_repl = jax.random.fold_in(rng_repl, step)

        state_repl, metrics_update = jitted_train_step(
            state=state_repl, batch=batch, rng=rng_repl
        )
        ##################################################################################################################
        #                                               test                                                             #
        ##################################################################################################################      
        # model_output = policy.action({
        #     "image": batch["observation"]["image"][0],
        #     "natural_language_embedding": batch["observation"]["natural_language_embedding"][0],
        # })
        # agent_action = np.concatenate((model_output['world_vector'], model_output['gripper_closedness_action']))
        # target_action = np.concatenate((batch["action"]["world_vector"][0][-1], batch["action"]["gripper_closedness_action"][0][-1]))
        # print('Agent action:', agent_action, 'Target action:', target_action)
        ##################################################################################################################

        if (step + 1) % LOG_LOSS_EVERY_STEPS == 0 or is_last_step:
            metrics_update = jax.device_get(metrics_update)
            print(f"Metrics: step={step}, {metrics_update}")
            wandb.log(
                flax.traverse_util.flatten_dict({"training": metrics_update}, sep="/"),
                step=step,
            )

        # Save the model checkpoint periodically
        if (step + 1) % SAVE_CHECKPOINT_EVERY_STEPS == 0:
            print(f"Saving model at step {step}")
            save_model(state_repl, os.path.join(CHECKPOINT_OUT_DIR, f"checkpoint_{step}"), step)

    print(f"Training finished at step {step}")

if __name__ == "__main__":
    main()
    