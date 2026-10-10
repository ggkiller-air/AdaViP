"""Keep simulator info for whole ManiFeel episodes during evaluation."""

from __future__ import annotations

from collections import defaultdict

from manifeel.gym_util.multistep_wrapper import MultiStepWrapper


class FullInfoMultiStepWrapper(MultiStepWrapper):
    """Retain every simulator info entry for episode-level success metrics."""

    def reset(self):
        """Clear full-episode info after the base observation history resets."""
        observation = super().reset()
        self.info = defaultdict(list)
        return observation
