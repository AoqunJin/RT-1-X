from typing import Any
import flax
import jax
import jax.numpy as jnp
import optax
import optax
import tensorflow as tf

from utils.model_utils import rt1_loss
from data.datasets import (
    get_trajectory_dataset,
    DATASET_NAME_TO_TRAJECTORY_DATASET_KWARGS,
)
from data.transform import prepare_for_model_input


def initialize_datasets(dataset_name_to_weight, train_split="train", val_split="test"):
    """Initializes and prepares datasets for training."""

    name_to_dataset = {
        k: get_trajectory_dataset(**DATASET_NAME_TO_TRAJECTORY_DATASET_KWARGS[k], split=train_split)
        for k in dataset_name_to_weight["train"].keys()
    }
    datasets = [dataset for dataset in name_to_dataset.values()]
    weights = [
        float(dataset_name_to_weight["train"][name])
        for name in name_to_dataset.keys()
    ]

    train_dataset = tf.data.Dataset.sample_from_datasets(datasets, weights=weights)
    train_dataset = prepare_for_model_input(
        train_dataset, target_height=300, target_width=300, training=True
    )
    
    # Prepare validation dataset if present
    val_dataset = None
    if "val" in dataset_name_to_weight:
        name_to_dataset = {
            k: get_trajectory_dataset(**DATASET_NAME_TO_TRAJECTORY_DATASET_KWARGS[k], split=val_split)
            for k in dataset_name_to_weight["val"].keys()
        }
        
        val_datasets = [dataset for dataset in name_to_dataset.values()]
        val_weights = [
            float(dataset_name_to_weight["val"][name])
            for name in name_to_dataset.keys()
        ]
        val_dataset = tf.data.Dataset.sample_from_datasets(
            val_datasets, weights=val_weights
        )
        val_dataset = prepare_for_model_input(
            val_dataset, target_height=300, target_width=300, training=False
        )

    return train_dataset, val_dataset


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
