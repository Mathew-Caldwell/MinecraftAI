from __future__ import annotations

from dataclasses import dataclass

from .actions import Action


@dataclass
class RewardSignal:
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    won_round: bool = False
    lost_round: bool = False
    aim_on_enemy: float = 0.0
    distance_score: float = 0.0

    def total(self) -> float:
        return (
            2.0 * self.damage_dealt
            - 2.0 * self.damage_taken
            + 25.0 * float(self.won_round)
            - 25.0 * float(self.lost_round)
            + 0.2 * self.aim_on_enemy
            + 0.1 * self.distance_score
        )


class PvPRLEnvironment:
    """Placeholder for the future RL loop.

    A real implementation needs reliable round reset and reward extraction. The cleanest
    non-memory approach is a visible scoreboard/chat/title overlay, OCR, or a controlled
    training plugin on a private server that reports only match results.
    """

    def reset(self):
        raise NotImplementedError("Connect this to your private PvP arena reset flow.")

    def step(self, action: Action):
        raise NotImplementedError("Capture screen, apply action, read reward, return observation.")
