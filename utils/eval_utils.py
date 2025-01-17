from typing import Any
import copy
import numpy as np
import jax
import jax.numpy as jnp
import flax

from utils.model_utils import rt1_loss
from models.rt1 import detokenize_action


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


def nucleus_filtering(logits: jnp.ndarray, p: float) -> jnp.ndarray:
    """
    Nucleus Filtering (Top-p) implementation in JAX
    
    Args:
        logits: Shape [..., vocab_size] array of logits
        p: Sum of probabilities of tokens to pick
        
    Returns:
        Shape [..., vocab_size] array of filtered log probabilities
    """
    # Get probabilities P(x_i | x_{1:i-1})
    probs = jax.nn.softmax(logits, axis=-1)
    
    # Sort probabilities in descending order
    sorted_probs = jnp.sort(probs, axis=-1)[..., ::-1]
    sorted_indices = jnp.argsort(probs, axis=-1)[..., ::-1]
    
    # Get cumulative sum of probabilities in sorted order
    cum_sum_probs = jnp.cumsum(sorted_probs, axis=-1)
    
    # Find cumulative sums less than p and prepend ones
    nucleus = cum_sum_probs < p
    prefix_shape = nucleus.shape[:-1] + (1,)
    prefix_ones = jnp.ones(prefix_shape, dtype=nucleus.dtype)
    nucleus = jnp.concatenate([prefix_ones, nucleus[..., :-1]], axis=-1)
    
    # Get log probabilities and mask out non-nucleus (use top 1-p)
    sorted_log_probs = jnp.log(sorted_probs)
    sorted_log_probs = jnp.where(nucleus, sorted_log_probs, float('-inf'))
    
    # Convert back to original vocabulary order
    flat_logits = jnp.full_like(logits, float('-inf'))
    return flat_logits.at[jnp.arange(logits.shape[0])[:, None], sorted_indices].set(sorted_log_probs)


def combined_sampling(logits, key, k=None, p=None, temperature=1.0):
    """
    Combined Top-k -> Top-p -> Temperature sampling strategy with reshaping for arbitrary dimensions.

    Args:
        logits (jnp.ndarray): Model output logits of shape (..., vocab_size).
        key (jax.random.PRNGKey): JAX random seed key.
        k (int or None): Top-k truncation threshold. Int \in [1, vocab_size].
        p (float or None): Top-p cumulative probability threshold. Float \in (0, 1].
        temperature (float): Temperature parameter.

    Returns:
        next_token (jnp.ndarray): Sampled token indices with shape matching logits[:-1].
    """
    original_shape = logits.shape  # Save original shape
    vocab_size = logits.shape[-1]  # Last dimension is vocab size

    # 1. Flatten all dimensions except vocab_size
    flat_logits = logits.reshape(-1, vocab_size)  # Shape: (batch_size_flat, vocab_size)

    # 2. Top-k filtering
    if k is not None:
        top_k_values, top_k_indices = jax.lax.top_k(flat_logits, k)  # Shape: (batch_size_flat, k)
        mask = jnp.full_like(flat_logits, -jnp.inf)
        flat_logits = mask.at[jnp.arange(flat_logits.shape[0])[:, None], top_k_indices].set(top_k_values)

    # 3. Top-p (Nucleus) filtering
    if p is not None:
        flat_logits = nucleus_filtering(flat_logits, p)
        
    # 4. Temperature scaling
    scaled_logits = flat_logits / temperature

    # 5. Sampling
    next_token = jax.random.categorical(key, scaled_logits, axis=-1)  # Shape: (batch_size_flat,)

    # 6. Reshape back to original batch shape
    next_token = next_token.reshape(original_shape[:-1])  # Remove vocab_size dimension

    return next_token


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

    def _run_action_inference(self, observation, rng, decoder_type):
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

        # action_logp = jax.nn.softmax(action_logits)
        if decoder_type == "greedy":
            action_token = jnp.argmax(action_logits, axis=-1)
        elif decoder_type == "kpt":  # Top-k -> Top-p -> Temperature sampling
            _, sampe_rng = jax.random.split(rng)
            action_token = combined_sampling(action_logits, sampe_rng, k=5, p=0.9, temperature=1)
        else:
            raise NotImplementedError
        
        # Detokenize the full action sequence.
        detokenized = detokenize_action(
            action_token, self.model.vocab_size, self.model.world_vector_range
        )

        detokenized = jax.tree_map(lambda x: x[0], detokenized)

        return detokenized

    def action(self, observation, decoder_type="greedy"):
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
        action = self._run_action_inference_jit(observation, rng, decoder_type)
        action = jax.device_get(action)

        # Use the base pose mode if the episode if the network outputs an invalid
        # `terminate_episode` action.
        if np.sum(action["terminate_episode"]) == 0:
            action["terminate_episode"] = np.zeros_like(action["terminate_episode"])
            action["terminate_episode"][-1] = 1
        return action
    

def get_eval_sample_batch():
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
    return sample_batch
