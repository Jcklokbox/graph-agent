"""v60: common-support reward selection with canonical sensory inference."""
import numpy as np

from agents.factor_controller import FactorController
from models.factor_event_reward import SignedSelection
from models.factor_joint_reward import JointEvidence


class JointReward(JointEvidence, SignedSelection):
    share_fixed_projection = True

    def constant_context_action_values(self, b, features, phase):
        """Select an action split only on a witnessed unchanged sensor context.

        The same replay rows score the pooled and split laws. This deliberately
        does not attribute a difference to the action at a changed arrival.
        """
        if self.A < 2 or len(self.replay) < 2 * self.A or (self.slots >= 0).any():
            return None
        if phase.any():
            return None
        if any(not np.array_equal(b, arrival) or not np.array_equal(features, x) or onset.any()
               for _, _, _, arrival, x, onset in reversed(self.replay)):
            return None
        _, actions, rewards, *_ = zip(*self.replay)
        rewards = np.asarray(rewards)
        bins = np.minimum(self.B - 1, (self.B * rewards).astype(int))
        key = np.asarray(actions) * self.B + bins
        counts = np.bincount(key, minlength=self.A * self.B).reshape(self.A, self.B)
        visits = counts.sum(1)
        if (visits < 2).any():
            return None
        pooled = self._joint_gamma[counts.sum(0)].sum() - self._joint_norm[len(rewards)]
        split = (self._joint_gamma[counts].sum(1) - self._joint_norm[visits]).sum()
        if split <= pooled:
            return None

        # The transition is the single witnessed context. The observed atom
        # can update only the first action's group before the next decision.
        sums = np.bincount(key, weights=rewards, minlength=counts.size).reshape(counts.shape)
        alpha = counts + 1. / self.B
        mass = visits + 1.
        probs = alpha / mass[:, None]
        atoms = (sums + self.center / self.B) / alpha
        mean = (probs * atoms).sum(1)
        other = np.where(np.eye(self.A, dtype=bool), -np.inf, mean).max(1)
        updated = mean[:, None] + (atoms - mean[:, None]) / (1. + mass[:, None])
        return mean + (probs * np.clip(np.maximum(other[:, None], updated), 0., 1.)).sum(1)


class Agent(FactorController):
    def __init__(self, n_inputs, n_actions, value_index, seed=0, *, draws=5, posterior=False):
        super().__init__(n_inputs, n_actions, value_index, seed, draws=draws,
                         posterior=posterior, event=True, event_reward_factory=JointReward)
