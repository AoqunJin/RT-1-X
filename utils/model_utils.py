import jax
from flax.training import checkpoints
import jax.numpy as jnp

from models.rt1 import tokenize_action


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
