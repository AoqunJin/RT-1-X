from typing import Callable, Union
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import rlds

import functools

from .transform import RLDSSpec, TrajectoryTransformBuilder, n_step_pattern_builder


StepFnMapType = Callable[[rlds.Step, rlds.Step], None]


def resize_to_resolution(
    image: Union[tf.Tensor, np.ndarray],
    target_width: int = 320,
    target_height: int = 256,
    to_numpy: bool = True,
) -> Union[tf.Tensor, np.ndarray]:
    """Resizes image and casts to uint8."""
    image = tf.image.resize_with_pad(
        image,
        target_width=target_width,
        target_height=target_height,
    )
    image = tf.cast(image, tf.uint8)
    if to_numpy:
        image = image.numpy()
    return image


def terminate_bool_to_act(terminate_episode: tf.Tensor) -> tf.Tensor:
    return tf.cond(
        terminate_episode == tf.constant(1.0),
        lambda: tf.constant([1, 0, 0], dtype=tf.int32),
        lambda: tf.constant([0, 1, 0], dtype=tf.int32),
    )


def rt_1_map_observation(
    to_step: rlds.Step,
    from_step: rlds.Step,
    from_image_feature_names: tuple[str, ...] = ("image",),
    to_image_feature_names: tuple[str, ...] = ("image",),
    resize: bool = True,
) -> None:
    """Map observation to model observation spec."""

    to_step[rlds.OBSERVATION]["natural_language_embedding"] = from_step[
        "language_embedding"
    ]

    for from_feature_name, to_feature_name in zip(
        from_image_feature_names, to_image_feature_names
    ):
        if resize:
            to_step["observation"][to_feature_name] = resize_to_resolution(
                from_step["observation"][from_feature_name],
                to_numpy=False,
                target_width=320,
                target_height=256,
            )


def metaworld_map_action(to_step: rlds.Step, from_step: rlds.Step):
    to_step[rlds.ACTION]["world_vector"] = from_step[rlds.ACTION][:3] * 2
    to_step[rlds.ACTION]["terminate_episode"] = terminate_bool_to_act(
        tf.cast(from_step[rlds.IS_TERMINAL], tf.float32)
    )
    # zero rotation
    to_step[rlds.ACTION]["rotation_delta"] = tf.zeros(3, dtype=tf.float32)
    to_step[rlds.ACTION]["gripper_closedness_action"] = tf.expand_dims(from_step[rlds.ACTION][3], axis=0)


def pad_initial_zero_steps(
    steps: tf.data.Dataset, num_zero_step: int
) -> tf.data.Dataset:
    zero_steps = steps.take(1)
    zero_steps = zero_steps.map(
        lambda x: tf.nest.map_structure(tf.zeros_like, x),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    zero_steps = zero_steps.repeat(num_zero_step)
    return rlds.transformations.concatenate(zero_steps, steps)


def pad_initial_zero_episode(
    episode: tf.data.Dataset, num_zero_step: int
) -> tf.data.Dataset:
    episode[rlds.STEPS] = pad_initial_zero_steps(episode[rlds.STEPS], num_zero_step)
    return episode


def get_trajectory_dataset(
    builder_dir: str, step_map_fn, trajectory_length: int, split="train[:10]"
):
    dataset_builder = tfds.builder_from_directory(builder_dir=builder_dir)

    dataset_builder_episodic_dataset = dataset_builder.as_dataset(split=split)

    # We need pad_initial_zero_episode because reverb.PatternDataset will skip
    # constructing trajectories where the first trajectory_length - 1 steps are
    # the final step in a trajectory. As such, without padding, the policies will
    # not be trained to predict the actions in the first trajectory_length - 1
    # steps.
    # We are padding with num_zero_step=trajectory_length-1 steps.
    dataset_builder_episodic_dataset = dataset_builder_episodic_dataset.map(
        functools.partial(
            pad_initial_zero_episode, num_zero_step=trajectory_length - 1
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    rlds_spec = RLDSSpec(
        observation_info=dataset_builder.info.features[rlds.STEPS][rlds.OBSERVATION],
        action_info=dataset_builder.info.features[rlds.STEPS][rlds.ACTION],
        step_metadata_info={
            "language_embedding": dataset_builder.info.features[rlds.STEPS][
                "language_embedding"
            ]
        },
    )

    trajectory_transform = TrajectoryTransformBuilder(
        rlds_spec,
        step_map_fn=step_map_fn,
        pattern_fn=n_step_pattern_builder(trajectory_length),
    ).build(validate_expected_tensor_spec=False)

    trajectory_dataset = trajectory_transform.transform_episodic_rlds_dataset(
        dataset_builder_episodic_dataset
    )

    return trajectory_dataset


def step_map_fn(step, map_observation: StepFnMapType, map_action: StepFnMapType):
    transformed_step = {}
    transformed_step[rlds.IS_FIRST] = step[rlds.IS_FIRST]
    transformed_step[rlds.IS_LAST] = step[rlds.IS_LAST]
    transformed_step[rlds.IS_TERMINAL] = step[rlds.IS_TERMINAL]
    transformed_step[rlds.OBSERVATION] = {}
    transformed_step[rlds.ACTION] = {
        "gripper_closedness_action": tf.zeros(1, dtype=tf.float32),
        "rotation_delta": tf.zeros(3, dtype=tf.float32),
        "terminate_episode": tf.zeros(3, dtype=tf.int32),
        "world_vector": tf.zeros(3, dtype=tf.float32),
        "base_displacement_vertical_rotation": tf.zeros(1, dtype=tf.float32),
        "base_displacement_vector": tf.zeros(2, dtype=tf.float32),
    }

    map_observation(transformed_step, step)
    map_action(transformed_step, step)

    return transformed_step


DATASET_NAME_TO_TRAJECTORY_DATASET_KWARGS = {
    # RT-1-metaworld
    "rt_1_metaworld_ml10_20e": {
        "builder_dir": "/home/sora/tensorflow_datasets/metaworld_ml10_20e/1.0.0/",
        "trajectory_length": 15,
        "step_map_fn": functools.partial(
            step_map_fn,
            map_observation=rt_1_map_observation,
            map_action=metaworld_map_action,
        ),
    },
    "rt_1_metaworld_ml10_40e": {
        "builder_dir": "/home/sora/tensorflow_datasets/metaworld_ml10_40e/1.0.0/",
        "trajectory_length": 15,
        "step_map_fn": functools.partial(
            step_map_fn,
            map_observation=rt_1_map_observation,
            map_action=metaworld_map_action,
        ),
    },
    "rt_1_metaworld_ml10_100e": {
        "builder_dir": "/home/sora/tensorflow_datasets/metaworld_ml10_100e/1.0.0/",
        "trajectory_length": 15,
        "step_map_fn": functools.partial(
            step_map_fn,
            map_observation=rt_1_map_observation,
            map_action=metaworld_map_action,
        ),
    },
    "rt_1_metaworld_ml45_20e": {
        "builder_dir": "/home/sora/tensorflow_datasets/metaworld_ml45_20e/1.0.0/",
        "trajectory_length": 15,
        "step_map_fn": functools.partial(
            step_map_fn,
            map_observation=rt_1_map_observation,
            map_action=metaworld_map_action,
        ),
    },
}
