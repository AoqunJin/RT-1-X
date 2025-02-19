import os
from absl import app, flags
import flax
import jax
from jax.experimental import mesh_utils
from jax.sharding import NamedSharding
import optax
import numpy as np
import optax
import functools
import tensorflow as tf
import wandb

from utils.model_utils import save_model, load_model
from utils.jax_utils import reshard, configure_jax
from utils.eval_utils import evaluate
from utils.visualization_lib import save_batch
from models.rt1 import RT1
from utils.train_utils import (
    initialize_datasets,
    create_train_state,
    train
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
CHECKPOINT_OUT_DIR = "/path/to/models/rt_1_x_jax_output"
NUM_TRAIN_STEPS = 40_000  # actual should be > 1M
SAVE_CHECKPOINT_EVERY_STEPS = 40_000
VAL_PER_STEPS = 1_000
NUM_VAL_STEPS = 100
LOG_LOSS_EVERY_STEPS = 100
DATASET_NAME_TO_WEIGHTS = {
    "train": {
        "metaworld_ml10_50e": 1.0,
        # "metaworld_ml10_100e": 1.0,
        # "metaworld_ml45_50e": 1.0,
        # "metaworld_ml45_100e": 1.0,
        # "libero_10": 1.0,
        # "libero_90": 1.0,
        # "libero_goal": 1.0,
        # "libero_object": 1.0,
        # "libero_spatial": 1.0,
    },
    # "val": {
    #     "metaworld_ml10_50e": 1.0,
    # }
}
WANDB_PROJECT_NAME = "metaworld_train"
WANDB_RUN_NAME = "ml10_50e"


def main(_):
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
    train_dataset, val_dataset = initialize_datasets(DATASET_NAME_TO_WEIGHTS)

    global_batch_size = jax.device_count() * PER_DEVICE_MICRO_BATCH_SIZE
    local_batch_size = jax.local_device_count() * PER_DEVICE_MICRO_BATCH_SIZE

    train_dataset = train_dataset.repeat(-1)  # repeated indefinitely | repeat(num_epochs)
    # Larger shuffle buffer leads to better performance, but consumes more RAM
    train_dataset = train_dataset.shuffle(buffer_size=1000)
    train_dataset = train_dataset.batch(local_batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    train_iter = train_dataset.as_numpy_iterator()
    
    if val_dataset is not None:
        val_dataset = val_dataset.repeat(-1)
        val_dataset = val_dataset.shuffle(buffer_size=1000)
        val_dataset = val_dataset.batch(local_batch_size, drop_remainder=True)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

        val_iter = val_dataset.as_numpy_iterator()
    
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
    # print_shapes(sample_batch)
    # print(sample_batch['action']['world_vector'])
    # print(sample_batch['observation']['natural_language_embedding'])
    # save_batch(sample_batch, output_dir="sample_batch")
    
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
    
    if val_dataset is not None:
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
    #                             Model training.                             #
    ###########################################################################
    
    metrics_train_sum = None
    for step in range(NUM_TRAIN_STEPS):
        batch = next(train_iter)
        batch = jax.tree_map(_form_gda, batch, global_data_shape)
        is_last_step = (step + 1 == NUM_TRAIN_STEPS)

        rng_repl = jax.random.fold_in(rng_repl, step)

        state_repl, metrics_train_update = jitted_train_step(
            state=state_repl, batch=batch, rng=rng_repl
        )
        if metrics_train_sum is None: metrics_train_sum = metrics_train_update
        else: metrics_train_sum = jax.tree_map(lambda x, y: x + y, metrics_train_sum, metrics_train_update)
        
        if val_dataset is not None and (step + 1) % VAL_PER_STEPS == 0:
            metrics_eval_sum = None
            for val_step in range(NUM_VAL_STEPS):
                val_batch = next(val_iter)
                val_batch = jax.tree_map(_form_gda, val_batch, global_data_shape)
                rng_repl = jax.random.fold_in(rng_repl, val_step)
                
                _, metrics_eval_update = jitted_eval_step(
                    state=state_repl, batch=val_batch, rng=rng_repl
                )
                if metrics_eval_sum is None: metrics_eval_sum = metrics_eval_update
                else: metrics_eval_sum = jax.tree_map(lambda x, y: x + y, metrics_eval_sum, metrics_eval_update)
            metrics_eval_update = jax.tree_map(lambda x: x / NUM_VAL_STEPS, metrics_eval_sum)
        
        if (step + 1) % LOG_LOSS_EVERY_STEPS == 0 or is_last_step:
            metrics_train_update = jax.tree_map(lambda x: x / NUM_VAL_STEPS, metrics_train_sum)
            metrics_train_update = jax.device_get(metrics_train_update)
            log_dict = {"training": metrics_train_update}
            metrics_train_sum = None
            
            if val_dataset is not None and (step + 1) % VAL_PER_STEPS == 0:
                metrics_eval_update = jax.device_get(metrics_eval_update)
                log_dict.update({"val":metrics_eval_update})
                
            print(f"Metrics: step={step}, {log_dict}")
            wandb.log(
                flax.traverse_util.flatten_dict(log_dict, sep="/"),
                step=step,
            )

        # Save the model checkpoint periodically
        if (step + 1) % SAVE_CHECKPOINT_EVERY_STEPS == 0:
            print(f"Saving model at step {step}")
            save_model(state_repl, CHECKPOINT_OUT_DIR, step)

    print(f"Training finished at step {step}")

if __name__ == "__main__":
    app.run(main)
    