from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ImageUnderstanding:
    caption: str
    characters: List[str]
    setting: str
    objects: List[str]
    mood: str
    inferred_plot_hint: str = ""


@dataclass
class CharacterProfile:
    name: str
    role: str
    age_group: str = "adult"
    gender: str = "unspecified"
    face: str = ""
    hair: str = ""
    body: str = ""
    outfit: str = ""
    signature_items: List[str] = field(default_factory=list)
    color_palette: str = ""
    identity_anchor_prompt: str = ""
    negative_identity_prompt: str = (
        "different person, changed face, changed hairstyle, changed outfit, "
        "missing signature item, inconsistent character design"
    )

    def to_prompt(self) -> str:
        items = ", ".join(self.signature_items) if self.signature_items else "none"
        parts = [
            f"name: {self.name}",
            f"role: {self.role}",
            f"age group: {self.age_group}",
            f"gender: {self.gender}",
            f"face: {self.face}",
            f"hair/head: {self.hair}",
            f"body: {self.body}",
            f"outfit: {self.outfit}",
            f"signature items: {items}",
            f"color palette: {self.color_palette}",
            f"identity anchor: {self.identity_anchor_prompt}",
        ]
        return "; ".join([p for p in parts if p and not p.endswith(": ")])


@dataclass
class StorySeed:
    image_summary: Optional[ImageUnderstanding]
    text_prompt: str
    protagonist: str
    target_ending_emotion: str
    genre: str
    style: str
    setting: str
    objects: List[str]
    characters: List[str]
    mood: str
    visual_symbols: Dict[str, str]
    character_profiles: List[CharacterProfile] = field(default_factory=list)
    raw_input: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DCEPlan:
    protagonist: str
    desire: str
    fear: str
    misbelief: str
    obstacle: str
    conflict: str
    event_spine: List[str]
    turning_point: str
    target_ending_emotion: str
    ending_state: str
    moral_or_theme: str


@dataclass
class EmotionArc:
    states: List[str]
    intensities: List[int]
    rationale: str


@dataclass
class StoryboardFrame:
    frame_id: int
    caption: str
    narrative_function: str
    event: str
    protagonist_state: str
    desire_link: str
    conflict_level: int
    emotion: str
    emotion_intensity: int
    visual_focus: str
    key_objects: List[str]
    facial_cue: str
    body_cue: str
    event_cue: str
    scene_cue: str
    cinematic_cue: str
    character_identity: str = ""
    character_reference_prompt: str = ""
    emotion_delta: str = ""
    emotion_visual_rule: str = ""
    composition_rule: str = ""
    quality_rule: str = ""
    negative_prompt: str = ""
    prompt: str = ""


@dataclass
class CandidateImage:
    frame_id: int
    candidate_id: int
    image_path: str
    prompt: str
    scores: Dict[str, float] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    seed: StorySeed
    abstract: str
    dce_plan: DCEPlan
    emotion_arc: EmotionArc
    storyboard: List[StoryboardFrame]
    selected_images: List[CandidateImage]
    ending_candidates: List[CandidateImage]
    evaluation_questions: Dict[str, Any]
    evaluation: Dict[str, Any]
    final_story_markdown: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
