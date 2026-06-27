from __future__ import annotations

from typing import Any, Dict, List

SYSTEM_NARRATIVE = """
You are a research-grade visual storytelling planner.
All stories must be grounded only in the given text, image summary, protagonist specification, and provided simple input metadata.
Never import characters, occupations, props, or scenes from unrelated example stories.
If JSON is requested, return concise valid JSON only.
Core structure: Desire -> Conflict -> Event Chain -> Ending Emotion (DCEE).
""".strip()

SYSTEM_VLM = """
You are a strict visual narrative evaluator. Return concise valid JSON only.
Evaluate whether an image shows the planned event, required objects, emotion, world state, and protagonist identity.
""".strip()

QUALITY_SUFFIX = (
    "full-color cinematic storybook illustration, rich natural colors, emotionally meaningful color palette, "
    "clear action, clear visual evidence, expressive face, expressive body language, detailed background, "
    "coherent anatomy, cinematic lighting, sharp focus, professional illustration quality"
)

NEGATIVE_PROMPT = (
    "monochrome, black and white, grayscale, pencil sketch, line art only, colorless image, "
    "missing action, missing required object, missing visual evidence, emotionless face, weak expression, stiff pose, portrait only, "
    "empty background, low quality, blurry, bad anatomy, distorted face, watermark, text"
)

EMOTION_RENDER_BOOK: Dict[str, Dict[str, str]] = {
    "joy": {"face": "bright eyes, raised cheeks, open smile", "body": "open chest, lifted posture", "lighting": "warm sunlight", "palette": "golden yellow, warm green, sky blue", "weather": "clear or bright sky", "composition": "open composition"},
    "relief": {"face": "soft exhale, relaxed eyes, grateful smile", "body": "shoulders dropping after tension", "lighting": "soft warm light", "palette": "soft amber, warm beige, gentle green", "weather": "clouds clearing", "composition": "balanced frame"},
    "hope": {"face": "focused eyes, slightly raised brows", "body": "forward leaning stance", "lighting": "soft warm directional light", "palette": "fresh green, soft blue, warm earth tone", "weather": "clear or lightly cloudy", "composition": "visible path ahead"},
    "sadness": {"face": "downcast eyes, tightened mouth", "body": "slumped shoulders, lowered head", "lighting": "dim diffused light", "palette": "cool desaturated blue-gray with muted earth tones", "weather": "rain, overcast sky, mist", "composition": "negative space and isolation"},
    "regret": {"face": "downcast eyes, pained expression", "body": "slumped posture, hand near face or chest", "lighting": "dim side light", "palette": "muted blue-gray and faded brown", "weather": "drizzle or overcast sky", "composition": "lonely frame with visible evidence of loss"},
    "fear": {"face": "wide eyes, tense brows", "body": "defensive pose, recoiling posture", "lighting": "hard contrast with looming shadows", "palette": "cold blue, gray, desaturated green", "weather": "fog or storm", "composition": "off-center framing with visible threat"},
    "anger": {"face": "furrowed brows, clenched jaw", "body": "rigid shoulders, clenched fist", "lighting": "harsh directional light", "palette": "red-orange accents with dark browns", "weather": "wind, dramatic clouds", "composition": "diagonal tension"},
    "determination": {"face": "steady eyes, firm mouth", "body": "forward movement, strong stance", "lighting": "strong light through shadow", "palette": "earth tones with warm highlights", "weather": "wind or clearing clouds", "composition": "visible obstacle and forward momentum"},
    "doubt": {"face": "uncertain gaze, hesitant mouth", "body": "paused pose", "lighting": "muted side light", "palette": "cool gray-blue, pale green", "weather": "mist or thin clouds", "composition": "ambiguous space"},
    "anxiety": {"face": "worried eyes, pressed lips", "body": "tight shoulders, restless hands", "lighting": "uneven cool light", "palette": "cold green-gray and muted blue", "weather": "cloudy wind before rain", "composition": "constricted path"},
}


def get_emotion_rule(emotion: str) -> Dict[str, str]:
    return EMOTION_RENDER_BOOK.get((emotion or "").lower().strip(), {
        "face": "clear readable facial expression",
        "body": "clear readable body posture",
        "lighting": "cinematic lighting aligned with emotion",
        "palette": "full natural color palette aligned with emotion",
        "weather": "weather aligned with story situation",
        "composition": "composition that reveals emotional state",
    })


def emotion_rule_text(emotion: str) -> str:
    r = get_emotion_rule(emotion)
    return f"face: {r['face']}; body: {r['body']}; lighting: {r['lighting']}; palette: {r['palette']}; weather: {r['weather']}; composition: {r['composition']}"


def choose_shot_type(idx: int, total: int, narrative_function: str) -> str:
    nf = (narrative_function or "").lower()
    if idx == 0:
        return "wide establishing shot"
    if idx == total - 1:
        return "medium-wide ending shot"
    if "turning" in nf or "climax" in nf:
        return "dramatic medium close-up"
    if "conflict" in nf or "obstacle" in nf or "event" in nf:
        return "action-focused medium shot"
    if "emotion" in nf or "reaction" in nf:
        return "emotional medium close-up"
    return "medium shot"


def choose_camera_distance(shot_type: str) -> str:
    st = (shot_type or "").lower()
    if "close" in st:
        return "close"
    if "wide" in st:
        return "wide"
    return "medium"


def image_understanding_prompt(image_path: str, sample: dict) -> str:
    return f"""
Describe the input image for grounded DCEE visual storytelling.
Return JSON with: caption, characters, setting, objects, mood, inferred_plot_hint, time_of_day, weather, environment_details.
Image path: {image_path}
Text prompt: {sample.get('text_prompt')}
Protagonist: {sample.get('protagonist')}
Target ending emotion: {sample.get('target_ending_emotion')}
Only describe what is actually present or safely inferable.
""".strip()


def story_seed_prompt(sample: dict, image_summary: dict | None, forbidden_entities: List[str]) -> str:
    return f"""
Create a GROUNDED story seed for visual storytelling.
Use only the user input text, image summary, protagonist specification, and simple metadata.
Do not import unrelated template stories or hidden prior examples.

Input sample:
{sample}

Image summary:
{image_summary}

Forbidden entities unless explicitly grounded in the input/image summary:
{forbidden_entities}

Return JSON with keys:
setting, objects, characters, mood, visual_symbols, world_context, character_profiles.

Requirements:
- protagonist identity must stay stable across all frames.
- objects and characters must be concrete, drawable, and grounded.
- if the protagonist is a panda, do not invent a woodcutter.
- character_profiles must include: name, role, age_group, gender, face, hair, body, outfit, signature_items, color_palette, identity_anchor_prompt.
""".strip()


def story_abstract_prompt(seed: dict, forbidden_entities: List[str]) -> str:
    return f"""
Write one grounded abstract paragraph for a visual story.
The story must follow Desire -> Conflict -> Event Chain -> Ending Emotion.
Use only grounded entities from the seed.
Forbidden ungrounded entities: {forbidden_entities}
Seed: {seed}
Return plain text only.
""".strip()


def dcee_plan_prompt(seed: dict, abstract: str, forbidden_entities: List[str]) -> str:
    return f"""
Create one grounded DCEE plan from the seed and abstract.
Forbidden ungrounded entities: {forbidden_entities}

Return JSON with keys:
protagonist, desire, fear, misbelief, obstacle, conflict, event_chain, event_spine, turning_point,
target_ending_emotion, ending_state, moral_or_theme, planning_structure.

Requirements:
- events must be concrete and visually drawable.
- do not introduce ungrounded occupations, characters, or props.
- do not introduce woodcutter, axe, fairy, river unless grounded in the input.
- event_chain must contain short concrete event objects. Each event item may be a string or object.

Seed: {seed}
Abstract: {abstract}
""".strip()


def emotion_arc_prompt(seed: dict, abstract: str, dce_plan: dict, num_frames: int) -> str:
    return f"""
Create an emotion arc with exactly {num_frames} states for the selected DCEE plan.
Return JSON with keys: states, intensities, rationale.
The last emotion must match the target ending emotion or a direct synonym.
Seed: {seed}
Abstract: {abstract}
DCEE plan: {dce_plan}
""".strip()


def next_story_sentence_prompt(seed: dict, dce_plan: dict, emotion_arc: dict, story_so_far: list, previous_frame: dict | None, frame_index: int, num_frames: int, forbidden_entities: List[str]) -> str:
    target_emotion = ""
    intens = ""
    if isinstance(emotion_arc, dict):
        states = emotion_arc.get("states", []) or []
        intensities = emotion_arc.get("intensities", []) or []
        if frame_index < len(states):
            target_emotion = states[frame_index]
        if frame_index < len(intensities):
            intens = intensities[frame_index]
    return f"""
Generate ONLY the next story sentence for frame {frame_index+1} of {num_frames}.
The sentence must be EASY TO DRAW.
Use one clear action, one clear place, visible objects, and a readable emotion.

Grounding rules:
- Use only grounded entities from the seed and previous story.
- Forbidden ungrounded entities: {forbidden_entities}
- Do not import hidden examples such as a woodcutter story.
- If the protagonist is a panda, keep the panda as the protagonist.
- The sentence must naturally continue the story-so-far.

Desired output JSON keys:
sentence, subject, action, object, location, weather, atmosphere, emotion, emotion_intensity,
visible_cause, required_objects, background_elements, supporting_cast, continuity_notes.

Requirements:
- sentence must be a single image-friendly sentence.
- exactly one primary visible action.
- specify visible cause of the emotion.
- required_objects must be concrete drawable items.
- background_elements must help the scene (forest, stream, bamboo, hill, etc. only if grounded).
- keep continuity with previous frame.
- frame emotion target: {target_emotion}
- frame emotion intensity target: {intens}

Seed: {seed}
DCEE plan: {dce_plan}
Story so far: {story_so_far}
Previous frame summary: {previous_frame}
""".strip()


# ---------------------------------------------------------------------
# Backward-compatible prompt functions
# ---------------------------------------------------------------------
def dcee_branch_plan_prompt(seed: dict, abstract: str, num_candidates: int = 4) -> str:
    return dcee_plan_prompt(seed, abstract, forbidden_entities=[])


def dcee_candidate_selection_prompt(seed: dict, abstract: str, candidates: list) -> str:
    return f"""
Select the best DCEE candidate for grounded visual storytelling.
Return JSON with selected_candidate_id, scores, reason.
Seed: {seed}
Abstract: {abstract}
Candidates: {candidates}
""".strip()


def dce_plan_prompt(seed: dict, abstract: str) -> str:
    return dcee_plan_prompt(seed, abstract, forbidden_entities=[])


def emotion_delta_text(prev_emotion: str | None, cur_emotion: str, intensity: int) -> str:
    if not prev_emotion:
        return f"establish {cur_emotion} with visible intensity {intensity}/5"
    if prev_emotion == cur_emotion:
        return f"maintain {cur_emotion}, intensity {intensity}/5"
    return f"visible transition from {prev_emotion} to {cur_emotion}; show it through face, body, color, light, weather and evidence"


def storyboard_prompt(seed: dict, abstract: str, dce_plan: dict, emotion_arc: dict, num_frames: int) -> str:
    return f"""
Create a {num_frames}-frame grounded storyboard.
Each frame must include concrete event, visible evidence, emotion, location, and required objects.
Do not introduce ungrounded template-story entities.
Seed: {seed}
Abstract: {abstract}
DCEE plan: {dce_plan}
Emotion arc: {emotion_arc}
Return JSON array only.
""".strip()


def canonicalize_storyboard_prompt(seed: dict, dce_plan: dict, storyboard: list) -> str:
    return f"""
Canonicalize this storyboard for image generation.
Replace vague references with explicit grounded entities.
Do not add ungrounded new characters or props.
Seed: {seed}
DCEE plan: {dce_plan}
Storyboard: {storyboard}
Return the same JSON array.
""".strip()


def eval_questions_prompt(dce_plan: dict, emotion_arc: dict, storyboard: list) -> str:
    return f"""
Generate VQA-style questions for grounded DCEE visual storytelling evaluation.
Return JSON with global_questions, frame_questions, ending_questions.
Questions must cover:
- exact story sentence alignment
- event visibility
- required object visibility
- emotional cause visibility
- protagonist consistency
- world/background consistency
DCEE plan: {dce_plan}
Emotion arc: {emotion_arc}
Storyboard: {storyboard}
""".strip()


def frame_prompt(frame: dict, dce_plan: dict, emotion_arc: dict, memory: dict, style: str, input_image_summary: dict | None) -> str:
    return f"""
{style}, full-color cinematic storybook illustration.
Exact story sentence: {frame.get('story_sentence') or frame.get('caption')}
Visible action: {frame.get('event')}
Visible cause: {frame.get('event_grounding')}
Required objects: {frame.get('must_show') or frame.get('key_objects')}
Emotion: {frame.get('emotion')} intensity {frame.get('emotion_intensity')}/5; {frame.get('emotion_visual_rule')}
World: {frame.get('scene_location')}, {frame.get('time_of_day')}, {frame.get('weather')}, {frame.get('atmosphere')}, {frame.get('environment_details')}
Memory: {memory}
Quality: {QUALITY_SUFFIX}
Negative: {NEGATIVE_PROMPT}
""".strip()
