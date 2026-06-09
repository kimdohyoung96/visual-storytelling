from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class CharacterLatent:
    name: str
    role: str
    identity_prompt: str
    outfit_prompt: str = ""
    signature_items: List[str] = field(default_factory=list)
    reference_image: Optional[str] = None
    negative_prompt: str = "different identity, changed face, changed outfit, inconsistent character"


@dataclass
class WorldLatent:
    scene_location: str
    time_of_day: str
    weather: str
    atmosphere: str
    environment_details: List[str] = field(default_factory=list)
    scene_transition: str = ""
    symbolic_objects: Dict[str, str] = field(default_factory=dict)

    def to_prompt(self) -> str:
        details = ", ".join(self.environment_details)
        symbols = ", ".join([f"{k} means {v}" for k, v in self.symbolic_objects.items()])
        return (
            f"location: {self.scene_location}; time: {self.time_of_day}; weather: {self.weather}; "
            f"atmosphere: {self.atmosphere}; environment details: {details}; "
            f"scene transition: {self.scene_transition}; visual symbols: {symbols}"
        )


@dataclass
class EmotionLatent:
    emotion: str
    intensity: int
    delta_from_previous: str
    facial_rule: str
    body_rule: str
    lighting_rule: str
    color_rule: str
    composition_rule: str

    def to_prompt(self) -> str:
        return (
            f"emotion: {self.emotion} intensity {self.intensity}/5; "
            f"transition: {self.delta_from_previous}; facial: {self.facial_rule}; "
            f"body: {self.body_rule}; lighting: {self.lighting_rule}; "
            f"color: {self.color_rule}; composition: {self.composition_rule}"
        )


@dataclass
class VisualControlPacket:
    frame_id: int
    positive_prompt: str
    negative_prompt: str
    adapter_weights: Dict[str, float]
    control_metadata: Dict[str, Any] = field(default_factory=dict)
    reference_images: Dict[str, str] = field(default_factory=dict)
