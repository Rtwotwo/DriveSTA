"""Configuration for the selected Endpoint algorithm only."""

from types import SimpleNamespace
from omegaconf import OmegaConf


class TTOConfig(SimpleNamespace):

    @classmethod
    def from_mapping(cls, value):
        if isinstance(value, cls):
            return value
        values = (
            OmegaConf.to_container(value, resolve=True)
            if OmegaConf.is_config(value)
            else dict(value)
        )
        config = cls(**values)
        if config.strategy != "goal_particle_selective":
            raise ValueError(
                "DriveSTA uses the goal_particle_selective Endpoint strategy"
            )
        if config.on_error != "raise":
            raise ValueError(
                "Evaluation must report errors instead of silently changing the method"
            )
        return config

    def to_dict(self):
        return vars(self).copy()
