import jax
import tensorflow as tf
import numpy as np


def configure_jax():
    """Configures JAX settings for GPU memory and backend."""
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.Session(config=config)
    jax.config.update("jax_platform_name", "gpu")
    jax.config.update("jax_xla_backend", "cuda")


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
