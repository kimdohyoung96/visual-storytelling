EMOTION_WORLD_RULES = {
    "hope": {
        "lighting": "soft warm light with visible path ahead",
        "weather": "clear or gently cloudy",
        "color": "light warm palette",
        "composition": "open space in front of protagonist",
        "environment": "background suggests possibility and forward movement",
    },
    "doubt": {
        "lighting": "cool muted side light",
        "weather": "cloudy or misty",
        "color": "blue-gray muted palette",
        "composition": "large negative space, protagonist appears small",
        "environment": "path or surroundings feel uncertain and unresolved",
    },
    "determination": {
        "lighting": "strong directional light cutting through shadow",
        "weather": "wind or clearing clouds",
        "color": "higher contrast with warm highlights",
        "composition": "low angle or centered forward motion",
        "environment": "obstacles are visible but navigable",
    },
    "sadness": {
        "lighting": "soft dim shadow",
        "weather": "rain, overcast sky, or fading dusk",
        "color": "cool muted palette",
        "composition": "negative space and isolation",
        "environment": "surroundings look empty, quiet, or emotionally distant",
    },
    "joy": {
        "lighting": "warm sunlight and clear highlights",
        "weather": "clear sky or bright after-rain atmosphere",
        "color": "vivid bright palette",
        "composition": "centered triumphant composition",
        "environment": "background feels open, celebratory, and resolved",
    },
    "relief": {
        "lighting": "soft brighter light after tension",
        "weather": "clouds clearing or gentle breeze",
        "color": "gentle warm tones",
        "composition": "goal visible, reduced visual pressure",
        "environment": "threat or obstacle recedes in the background",
    },
}


def get_world_rule(emotion: str):
    return EMOTION_WORLD_RULES.get((emotion or "").lower(), {
        "lighting": "cinematic lighting that supports the current emotion",
        "weather": "weather that fits the story event",
        "color": "color palette aligned with emotion",
        "composition": "clear story-focused composition",
        "environment": "background details support the narrative situation",
    })
