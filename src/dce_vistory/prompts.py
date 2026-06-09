from __future__ import annotations

from typing import Any, Dict

SYSTEM_NARRATIVE = """
You are a research-grade multimodal narrative planner.
Return concise valid JSON whenever JSON is requested.
""".strip()

SYSTEM_VLM = """
You are a strict visual narrative evaluator.
Return concise valid JSON only.
""".strip()

QUALITY_SUFFIX = (
    "full-color cinematic storybook illustration, rich natural color rendering, "
    "emotionally meaningful color palette, clear emotional storytelling, strong facial acting, "
    "expressive body language, coherent anatomy, high detail environment, visually readable action, "
    "cinematic lighting, sharp focus, professional illustration quality"
)

NEGATIVE_PROMPT = (
    "monochrome, black and white, grayscale, pencil sketch, charcoal sketch, line art only, "
    "low quality, blurry, bad anatomy, extra fingers, deformed hands, distorted face, "
    "emotionless face, stiff pose, empty background, washed-out colors, flat lighting, "
    "weak expression, colorless image, random text, watermark, logo, cropped face"
)

EMOTION_RENDER_BOOK: Dict[str, Dict[str, str]] = {
    "joy": {
        "face": "bright eyes, raised cheeks, gentle open smile, visibly happy expression",
        "body": "open chest, lifted posture, relaxed shoulders, energetic or relieved body language",
        "lighting": "warm sunlight, clean highlights, bright atmosphere",
        "palette": "golden yellow, warm green, sky blue, vivid natural colors",
        "weather": "clear sky or clouds opening after tension",
        "composition": "open composition with visible space ahead and a sense of release",
    },
    "happiness": {
        "face": "bright eyes, raised cheeks, gentle open smile, visibly happy expression",
        "body": "open chest, lifted posture, relaxed shoulders, energetic or relieved body language",
        "lighting": "warm sunlight, clean highlights, bright atmosphere",
        "palette": "golden yellow, warm green, sky blue, vivid natural colors",
        "weather": "clear sky or clouds opening after tension",
        "composition": "open composition with visible space ahead and a sense of release",
    },
    "relief": {
        "face": "soft exhale, relaxed eyes, subtle grateful smile, tension leaving the face",
        "body": "shoulders dropping after tension, calm grounded posture, hands loosening",
        "lighting": "soft warm light after shadow",
        "palette": "warm beige, gentle green, soft amber, calm blue",
        "weather": "clouds clearing, gentle breeze, quiet air",
        "composition": "balanced frame with reduced visual pressure",
    },
    "hope": {
        "face": "focused eyes, slightly raised brows, determined but gentle mouth",
        "body": "forward-leaning stance, ready movement, cautious but open posture",
        "lighting": "soft warm directional light",
        "palette": "fresh green, warm earth tones, soft blue, early sunlight",
        "weather": "clear or lightly cloudy",
        "composition": "path or destination visible in front of the protagonist",
    },
    "sadness": {
        "face": "downcast eyes, tightened mouth, heavy eyelids, visibly sorrowful expression",
        "body": "slumped shoulders, stillness, lowered head, closed body posture",
        "lighting": "dim diffused light, soft shadow",
        "palette": "cool desaturated blue-gray with muted earth colors",
        "weather": "rain, overcast sky, mist, damp air",
        "composition": "negative space and isolation around the protagonist",
    },
    "regret": {
        "face": "downcast eyes, tense mouth, pained expression, gaze avoiding the viewer",
        "body": "slumped shoulders, hand near chest or face, hesitant stillness",
        "lighting": "dim side light with soft shadow",
        "palette": "muted blue-gray, faded brown, low saturation but still full color",
        "weather": "drizzle, mist, overcast sky",
        "composition": "lonely composition with visible evidence of a wrong choice",
    },
    "fear": {
        "face": "wide eyes, tense brows, tight jaw or parted lips, alarmed expression",
        "body": "defensive pose, recoiling posture, tense shoulders, guarded hands",
        "lighting": "hard contrast, looming shadows",
        "palette": "cold blue, gray, desaturated green with dark accents",
        "weather": "fog, storm, oppressive darkness, cold wind",
        "composition": "off-center framing with visible threat or uncertainty in the scene",
    },
    "anger": {
        "face": "furrowed brows, intense stare, clenched jaw, visibly angry expression",
        "body": "tight stance, forceful gesture, clenched fist, rigid shoulders",
        "lighting": "harsh directional light, dramatic contrast",
        "palette": "red-orange accents, dark browns, high contrast warm tones",
        "weather": "wind, dramatic clouds, heated atmosphere",
        "composition": "diagonal tension, compressed frame, confrontational staging",
    },
    "determination": {
        "face": "steady eyes, focused brows, firm mouth, unwavering expression",
        "body": "forward movement, strong stance, stable posture, purposeful hands",
        "lighting": "strong light cutting through shadow",
        "palette": "earth tones with controlled warm highlights and clear contrast",
        "weather": "wind or clearing clouds",
        "composition": "forward momentum and visible obstacle",
    },
    "doubt": {
        "face": "uncertain gaze, slightly tightened brows, hesitant mouth",
        "body": "paused pose, weight shifted back, uncertain gesture, guarded posture",
        "lighting": "muted side light, partial shadow",
        "palette": "cool gray-blue with pale greens and muted browns",
        "weather": "mist, thin clouds, still air",
        "composition": "ambiguous space, partially blocked path, visual uncertainty",
    },
    "anxiety": {
        "face": "worried eyes, tense brows, pressed lips, alert expression",
        "body": "tight shoulders, restless hands, slightly hunched posture",
        "lighting": "uneven cool light with mild shadow",
        "palette": "cold green-gray, muted blue, pale brown",
        "weather": "mist, cloudy sky, wind before rain",
        "composition": "crowded foreground or constricted path",
    },
}


def get_emotion_rule(emotion: str) -> Dict[str, str]:
    return EMOTION_RENDER_BOOK.get(
        (emotion or "").lower().strip(),
        {
            "face": "clear readable emotional facial expression",
            "body": "clear readable emotional body posture",
            "lighting": "cinematic lighting aligned with the emotion",
            "palette": "full natural color palette aligned with the emotion",
            "weather": "weather aligned with the story situation",
            "composition": "composition that clearly reveals the emotional state",
        },
    )


def emotion_rule_text(emotion: str) -> str:
    r = get_emotion_rule(emotion)
    return (
        f"Facial expression: {r['face']}; Body posture: {r['body']}; "
        f"Lighting: {r['lighting']}; Color palette: {r['palette']}; "
        f"Weather/world: {r['weather']}; Composition: {r['composition']}"
    )


def emotion_delta_text(prev_emotion: str | None, cur_emotion: str, intensity: int) -> str:
    if not prev_emotion:
        return f"Establish the starting emotion as {cur_emotion} with visible intensity {intensity}/5."
    if prev_emotion == cur_emotion:
        return f"Maintain {cur_emotion}, but make intensity {intensity}/5 clearly visible."
    return (
        f"Show a visible emotional transition from {prev_emotion} to {cur_emotion}. "
        "The transition must be visible through face, posture, lighting, color, weather, and composition."
    )


def choose_shot_type(idx: int, total: int, narrative_function: str) -> str:
    nf = (narrative_function or "").lower()
    if idx == 0:
        return "wide establishing shot"
    if "climax" in nf or "turning" in nf:
        return "dramatic medium close-up"
    if "resolution" in nf or "ending" in nf or idx == total - 1:
        return "medium-wide ending shot"
    if "conflict" in nf or "obstacle" in nf:
        return "action-focused medium shot"
    if "reaction" in nf or "emotion" in nf:
        return "emotional close-up"
    return "medium shot"


def choose_camera_distance(shot_type: str) -> str:
    st = (shot_type or "").lower()
    if "close" in st:
        return "close"
    if "wide" in st:
        return "wide"
    return "medium"


def color_instruction(frame: Any) -> str:
    return (
        f"Use a full-color palette. Dominant palette: {getattr(frame, 'color_palette', 'rich natural full color')}. "
        "The illustration must be richly colored and must not be monochrome, grayscale, black-and-white, pencil-only, or sketch-only."
    )


def image_understanding_prompt(image_path: str, sample: dict) -> str:
    return f"""
Describe the input image for multimodal story planning.
Focus on characters, clothing, setting, objects, mood, inferred plot hint, time of day, weather, and environment details.

Input image path: {image_path}
Text prompt: {sample.get('text_prompt')}
Protagonist: {sample.get('protagonist')}
Target ending emotion: {sample.get('target_ending_emotion')}

Return JSON with:
caption: str
characters: list[str]
setting: str
objects: list[str]
mood: str
inferred_plot_hint: str
time_of_day: str
weather: str
environment_details: list[str]
""".strip()


def story_seed_prompt(sample: dict, image_summary: dict | None) -> str:
    return f"""
Create a multimodal story seed and character bible from the available inputs.

Text prompt: {sample.get('text_prompt')}
Protagonist: {sample.get('protagonist')}
Target ending emotion: {sample.get('target_ending_emotion')}
Genre: {sample.get('genre')}
Style: {sample.get('style')}
Image summary: {image_summary}

Return JSON with:
setting: str
objects: list[str]
characters: list[str]
mood: str
visual_symbols: object
world_context: object with keys such as region, season, time_of_day, weather_prior, atmosphere_prior, environment_prior, social_context
character_profiles: list of objects, each with:
  name, role, age_group, gender, face, hair, body, outfit, signature_items, color_palette, identity_anchor_prompt

The protagonist profile must be visually specific enough to preserve the same identity across all frames.
The world_context should be specific enough to ground background, weather, and situational details in later frames.
""".strip()


def story_abstract_prompt(seed: dict) -> str:
    return f"""
Write a story abstract in 4-6 sentences.

Seed:
{seed}

Requirements:
- Introduce the protagonist desire.
- Introduce a meaningful conflict.
- Make the ending compatible with the target ending emotion.
- Keep it visually drawable.
- Keep the protagonist identity stable across the story.
- Make scene changes, background conditions, and situation progression naturally support the story.
""".strip()


def dce_plan_prompt(seed: dict, abstract: str) -> str:
    return f"""
Create a DCE plan (Desire, Conflict, Ending / Event progression).

Seed:
{seed}

Story abstract:
{abstract}

Return valid JSON with:
protagonist, desire, fear, misbelief, obstacle, conflict, event_spine, turning_point,
target_ending_emotion, ending_state, moral_or_theme.
""".strip()


def emotion_arc_prompt(seed: dict, abstract: str, dce_plan: dict, num_frames: int) -> str:
    return f"""
Create an emotion arc planner with exactly {num_frames} steps.

Seed:
{seed}

Story abstract:
{abstract}

DCE plan:
{dce_plan}

Return valid JSON with:
states: list[str]
intensities: list[int]
rationale: str

The emotional change must be gradual and narratively motivated.
The last state must match the target ending emotion or its natural visual synonym.
""".strip()


def storyboard_prompt(seed: dict, abstract: str, dce_plan: dict, emotion_arc: dict, num_frames: int) -> str:
    return f"""
Create a {num_frames}-frame storyboard.

Seed:
{seed}

Abstract:
{abstract}

DCE plan:
{dce_plan}

Emotion arc:
{emotion_arc}

Return a JSON array. Each item must include:
frame_id, caption, narrative_function, event, protagonist_state, desire_link, conflict_level,
emotion, emotion_intensity, visual_focus, key_objects, facial_cue, body_cue, event_cue, scene_cue, cinematic_cue,
scene_location, time_of_day, weather, atmosphere, environment_details, supporting_cast, scene_transition.

Important:
- Each frame must show not only the protagonist's emotion, but also WHY the emotion occurs.
- The event, key object, or conflict evidence must be visible in the image.
- Use full-color visual storytelling, never monochrome or grayscale.
- Keep the same protagonist identity in every frame.
- Follow a desire-conflict-ending trajectory.
- The final frame must clearly express the target ending emotion.
""".strip()


def frame_prompt(frame: dict, dce_plan: dict, emotion_arc: dict, memory: dict, style: str, input_image_summary: dict | None) -> str:
    emotion = frame.get("emotion", "")
    env_details = ", ".join(frame.get("environment_details", []))
    support_cast = ", ".join(frame.get("supporting_cast", []))
    must_show = ", ".join(frame.get("must_show", []))
    emotion_evidence = ", ".join(frame.get("emotion_evidence", []))
    return f"""
Generate one frame of a visual story.

[STYLE]
{style}
full-color cinematic storybook illustration, richly colored, emotionally expressive.

[CHARACTER IDENTITY - MUST PRESERVE]
{frame.get('character_identity')}
{frame.get('character_reference_prompt')}

[FRAME EVENT]
Caption: {frame.get('caption')}
Narrative function: {frame.get('narrative_function')}
Event: {frame.get('event')}
Protagonist state: {frame.get('protagonist_state')}
Desire link: {frame.get('desire_link')}
Conflict level: {frame.get('conflict_level')}
Visual focus: {frame.get('visual_focus')}
Key objects: {frame.get('key_objects')}
Supporting cast: {support_cast}

[EMOTION - MUST BE IMMEDIATELY READABLE]
Target emotion: {emotion} (intensity {frame.get('emotion_intensity')}/5)
Emotion delta from previous frame: {frame.get('emotion_delta')}
Emotion visual rule: {frame.get('emotion_visual_rule') or emotion_rule_text(emotion)}
Facial cue: {frame.get('facial_cue')}
Body cue: {frame.get('body_cue')}
Emotion evidence that must appear in the scene: {emotion_evidence}

[WORLD / BACKGROUND / SITUATION]
Scene location: {frame.get('scene_location')}
Time of day: {frame.get('time_of_day')}
Weather: {frame.get('weather')}
Atmosphere: {frame.get('atmosphere')}
Environment details: {env_details}
Scene transition from previous frame: {frame.get('scene_transition')}

[CAMERA / COLOR]
Shot type: {frame.get('shot_type')}
Camera distance: {frame.get('camera_distance')}
Lighting style: {frame.get('lighting_style')}
Color palette: {frame.get('color_palette')}
Color instruction: full-color only, rich natural colors, never monochrome, never grayscale.

[MUST SHOW]
{must_show}

[DCE PLAN]
{dce_plan}

[EMOTION ARC]
{emotion_arc}

[INPUT IMAGE SUMMARY]
{input_image_summary}

[NARRATIVE MEMORY]
{memory}

[QUALITY]
{QUALITY_SUFFIX}

Create a single coherent full-color cinematic storybook illustration.
The protagonist's emotion must be visible through face, body, color, lighting, weather, composition, and story evidence.
The scene must clearly show the current event and why the protagonist feels this emotion.
Do not create a black-and-white drawing, grayscale sketch, pencil sketch, or monochrome image.
""".strip()


def eval_questions_prompt(dce_plan: dict, emotion_arc: dict, storyboard: list) -> str:
    return f"""
Generate evaluation questions for visual story assessment.

DCE plan:
{dce_plan}

Emotion arc:
{emotion_arc}

Storyboard:
{storyboard}

Return valid JSON with:
global_questions: list[str]
frame_questions: object
ending_questions: list[str]
""".strip()


def vlm_eval_prompt(dce_plan: dict, emotion_arc: dict, storyboard: list, questions: dict) -> str:
    return f"""
Evaluate the generated visual story by looking at the images.

DCE plan:
{dce_plan}

Emotion arc:
{emotion_arc}

Storyboard:
{storyboard}

Questions:
{questions}

Return valid JSON with:
text_image_alignment, character_consistency, image_quality, colorfulness, desire_alignment, conflict_visibility,
emotion_visibility, emotion_cause_visibility, emotion_transition_visibility, scene_consistency, world_fidelity,
event_alignment, ending_emotion_accuracy, narrative_coherence, interestingness, rationale.
""".strip()


def ending_candidate_eval_prompt(final_frame: dict, dce_plan: dict) -> str:
    return f"""
Evaluate ending candidates for the final frame.

Final frame specification:
{final_frame}

DCE plan:
{dce_plan}

Return valid JSON with:
candidate_scores: list[object]
Each item must contain:
candidate_id, identity_consistency, image_quality, colorfulness, world_fidelity,
ending_emotion_accuracy, emotion_visibility, event_alignment, visual_clarity, overall, reason.
""".strip()
