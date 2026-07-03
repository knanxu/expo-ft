import jax
import jax.numpy as jnp
import numpy as np

from expo_ft.agents.alg.speedtune_dqn import SpeedTuneLearner
from expo_ft.data.speedtune_buffer import SpeedTuneReplayBuffer
from expo_ft.networks.rainbow_dqn import masked_expected_q_argmax


FEAT_DIM = 8


def _learner(epsilon):
    return SpeedTuneLearner.create(
        rng=jax.random.PRNGKey(0),
        feat_dim=FEAT_DIM,
        head_sizes=(7,),
        n_atoms=11,
        v_min=0.0,
        v_max=10.0,
        epsilon=epsilon,
    )


def test_masked_argmax_excludes_higher_q_locked_actions():
    support = jnp.asarray([0.0, 1.0])
    dists = [jnp.asarray([[[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]])]
    full = np.asarray(masked_expected_q_argmax(dists, support, (2,)))
    masked = np.asarray(masked_expected_q_argmax(dists, support, (1,)))
    assert full.tolist() == [[2]]
    assert masked.tolist() == [[1]]


def test_epsilon_random_sampling_never_exceeds_unlocked_bin():
    learner = _learner(epsilon=1.0)
    feat = np.ones((512, FEAT_DIM), dtype=np.float32)

    slow, learner = learner.sample_action_idxs(feat, max_action_idxs=(0,))
    opened, _ = learner.sample_action_idxs(feat, max_action_idxs=(2,))

    assert np.asarray(slow).max() == 0
    opened = np.asarray(opened)
    assert opened.min() >= 0 and opened.max() <= 2
    assert len(np.unique(opened)) >= 2


def test_greedy_and_update_accept_action_mask():
    learner = _learner(epsilon=0.0)
    feat = np.ones((4, FEAT_DIM), dtype=np.float32)
    greedy = np.asarray(learner.greedy_action_idxs(feat, max_action_idxs=(0,)))
    assert greedy.tolist() == [[0], [0], [0], [0]]

    buffer = SpeedTuneReplayBuffer(
        capacity=100, feat_dim=FEAT_DIM, n_heads=1, n_step=1, seed=0
    )
    for index in range(20):
        state = np.full(FEAT_DIM, index, dtype=np.float32)
        buffer.insert(state, [index % 2], 1.0, state + 1, True)
    batch = buffer.sample(8)
    learner, td, metrics = learner.update(batch, max_action_idxs=(1,))
    assert np.asarray(td).shape == (8,)
    assert "loss/dqn" in metrics


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} SpeedTune action-mask tests passed")


if __name__ == "__main__":
    main()
