from __future__ import annotations

from typing import Any, Dict, List

SYSTEM_NARRATIVE = """
You are a research-grade visual storytelling planner.
All stories must be grounded only in the given text, input image summary, protagonist specification, and simple input metadata.
Never import characters, occupations, props, or scenes from unrelated example stories.
If JSON is requested, return concise valid JSON only.
Core structure: Desire -> Conflict -> Event Chain -> Ending Emotion (DCEE).

V23 POLICY:
- The input image is a hard identity anchor. The protagonist's species, fur/skin/clothing color, age impression, body shape, and distinctive appearance must be preserved.
- If the input subject is a white bear, keep a white bear in story and image generation; never drift to a brown bear or a different species.
- The story must stay protagonist-centered and visually drawable.
- Do not create ungrounded secondary characters, crowds, helpers, woodcutters, or unrelated animals.
- One frame corresponds to one single-scene image. Never describe split-screen, multiple panels, or multiple moments in one frame.
- Use ConsiStory-inspired subject-only consistency: preserve the protagonist identity across frames, but do not copy background/layout mistakes.
- Each frame must contain a visible event, not just a nice portrait.
""".strip()

SYSTEM_VLM = """
You are a strict visual narrative evaluator. Return concise valid JSON only.
Evaluate whether an image shows the protagonist, planned event, required props, background world, emotion, and identity consistency.
""".strip()

QUALITY_SUFFIX = (
    "full-color cinematic storybook illustration, rich natural colors, emotionally meaningful color palette, "
    "clear protagonist action, clear prop evidence, expressive face, expressive body language, detailed background, "
    "coherent anatomy, cinematic lighting, sharp focus, professional illustration quality, single coherent scene"
)

NEGATIVE_PROMPT = (
    "monochrome, black and white, grayscale, pencil sketch, line art only, colorless image, "
    "split screen, comic panel, storyboard sheet, collage, diptych, triptych, multi-panel, multiple scenes in one frame, "
    "extra character, secondary character, crowd, unrelated animal, unrelated human, duplicated protagonist, duplicate subject, second bear, two bears, multiple bears, clone, mirror duplicate, "
    "missing protagonist, missing action, missing required prop, missing visual evidence, emotionless face, weak expression, stiff pose, portrait only, "
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
Describe the input image for grounded V21 visual storytelling.
The goal is identity-faithful story and image generation.

Return JSON with keys:
caption, characters, protagonist_description, protagonist_species, protagonist_color, coat_pattern,
age_group, body_shape, distinguishing_traits, setting, objects, background_objects, mood,
inferred_plot_hint, time_of_day, weather, environment_details.

Image path: {image_path}
Text prompt: {sample.get('text_prompt')}
Protagonist: {sample.get('protagonist')}
Target ending emotion: {sample.get('target_ending_emotion')}

Important rules:
- Focus on the main protagonist identity.
- Explicitly state color/species/appearance if visually visible (for example: white bear, brown bear, black-and-white panda).
- Do not invent secondary characters.
- Keep background details concrete (forest, river, bamboo, rock, snow, rain, path, hill, etc.).
""".strip()


def story_seed_prompt(sample: dict, image_summary: dict | None, forbidden_entities: List[str], protagonist_only: bool = True) -> str:
    return f"""
Create a grounded V21 story seed for protagonist-centered visual storytelling.
Use only the user input text, input image summary, protagonist specification, and simple metadata.

Input sample:
{sample}

Image summary:
{image_summary}

Forbidden entities unless explicitly grounded in the input/image summary:
{forbidden_entities}

V21 policy:
- protagonist_only = {protagonist_only}
- The input image is the identity anchor.
- Preserve protagonist species, color, age impression, body shape, and distinctive appearance.
- The story must center on the protagonist.
- Do not invent friends, wild animal friends, helpers, enemies, humans, woodcutters, fairies, villagers, crowds, or other agents.
- Keep only simple props and grounded background objects needed for the protagonist's action.
- One frame must later map to one drawable single scene.

Return JSON with keys:
setting, objects, characters, mood, visual_symbols, world_context, character_profiles.

Requirements:
- characters must contain only the protagonist unless the input explicitly names another character.
- objects must be props/background, not new characters.
- world_context should preserve grounded environment: terrain, weather, time_of_day, flora, water, landmarks.
- character_profiles must include: name, role, age_group, gender, face, hair, body, outfit, signature_items, color_palette, species, fur_color, distinguishing_traits, identity_anchor_prompt.
- if the protagonist is a white bear, preserve white-bear identity; do not drift to a brown bear.
""".strip()


def story_abstract_prompt(seed: dict, forbidden_entities: List[str], protagonist_only: bool = True) -> str:
    return f"""
Write one grounded abstract paragraph for a V21 protagonist-centered visual story.
The story must follow Desire -> Conflict -> Event Chain -> Ending Emotion.
Use only grounded entities from the seed.

Forbidden ungrounded entities:
{forbidden_entities}

V21 policy:
- protagonist_only = {protagonist_only}
- The protagonist is the only active character unless the input explicitly includes someone else.
- Conflict must come from environment, lost object, obstacle, weather, distance, hunger, mistake, or internal emotion.
- Keep the abstract visually grounded in the input image world and protagonist identity.
- Do not create friends, helpers, enemies, humans, woodcutters, fairies, crowds, or unrelated animals.
- Each later frame must be drawable as one clear scene.

Seed:
{seed}

Return plain text only.
""".strip()


def dcee_plan_prompt(seed: dict, abstract: str, forbidden_entities: List[str], protagonist_only: bool = True) -> str:
    return f"""
Create one grounded V21 DCEE plan from the seed and abstract.

Forbidden ungrounded entities:
{forbidden_entities}

V21 protagonist-centered policy:
- protagonist_only = {protagonist_only}
- The protagonist is the only active character unless explicitly grounded.
- Keep protagonist identity visually stable.
- Do not introduce secondary characters or other agents.
- Conflict must be protagonist-centered: environment obstacle, lost prop, hunger, weather, distance, injury, fear, or internal dilemma.
- event_chain must contain concrete, visually drawable events.
- Each event should naturally become one frame with one main action.

Return JSON with keys:
protagonist, desire, fear, misbelief, obstacle, conflict, event_chain, event_spine, turning_point,
target_ending_emotion, ending_state, moral_or_theme, planning_structure.

Seed:
{seed}

Abstract:
{abstract}
""".strip()


def emotion_arc_prompt(seed: dict, abstract: str, dce_plan: dict, num_frames: int) -> str:
    return f"""
Create an emotion arc with exactly {num_frames} states for the selected V21 DCEE plan.
Return JSON with keys: states, intensities, rationale.
The last emotion must match the target ending emotion or a direct synonym.
All emotions must be visually expressible by the protagonist's face, body, pose, lighting, color, and background.
Seed: {seed}
Abstract: {abstract}
DCEE plan: {dce_plan}
""".strip()


def next_story_sentence_prompt(seed: dict, dce_plan: dict, emotion_arc: dict, story_so_far: list, previous_frame: dict | None, frame_index: int, num_frames: int, forbidden_entities: List[str], protagonist_only: bool = True) -> str:
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
Generate ONLY the next protagonist-centered story sentence for frame {frame_index+1} of {num_frames}.
The sentence must be EASY TO DRAW and must focus on the protagonist.

Grounding rules:
- Use only grounded entities from the seed, story_so_far, and previous_frame.
- Forbidden ungrounded entities: {forbidden_entities}
- protagonist_only = {protagonist_only}
- Do not create any new character, friend, animal friend, helper, enemy, human, woodcutter, fairy, crowd, or duplicate protagonist.
- Preserve protagonist identity from the input image and seed character profile.
- If the protagonist is a white bear in the input, keep a white bear.
- The protagonist must remain the subject of every sentence.
- The sentence must naturally continue story_so_far.

Visual simplicity rules:
- One sentence = one single drawable scene.
- Use one main action only.
- Use a small number of visible objects.
- Use a concrete place, weather, and emotion that an image model can show clearly.
- Avoid abstract, multi-clause, or hard-to-illustrate events.

Allowed visual elements:
- protagonist
- protagonist's simple props from input/seed
- grounded background objects such as forest, river, bamboo, rock, leaf, rain, path, hill, shadow, sunlight

Desired output JSON keys:
sentence, subject, action, object, location, weather, atmosphere, emotion, emotion_intensity,
visible_cause, required_objects, background_elements, supporting_cast, continuity_notes, action_pose, camera_composition, absent_objects.

Strict output rules:
- subject must be the protagonist only.
- supporting_cast must be [].
- sentence should be short and visually concrete.
- action_pose must describe the protagonist's body pose needed for this event.
- camera_composition must describe how to frame this single scene.
- absent_objects must list story objects that should NOT appear in this frame because they were lost, gone, or not relevant.
- required_objects must list only concrete visible objects needed in the image.
- background_elements must list location/background details actually visible in the scene.
- target emotion for this frame: {target_emotion}
- target emotion intensity: {intens}

Seed:
{seed}

DCEE plan:
{dce_plan}

Story so far:
{story_so_far}

Previous frame:
{previous_frame}
""".strip()


# ---------------------------------------------------------------------
# Backward-compatible prompt functions for evaluator / older modules
# ---------------------------------------------------------------------
def eval_questions_prompt(dce_plan: dict, emotion_arc: dict, storyboard: list) -> str:
    return f"""
Generate VQA-style questions for grounded V22 DCEE visual storytelling evaluation.
Return valid JSON with keys: global_questions, frame_questions, ending_questions.

Questions must cover:
- whether each image matches the exact story sentence
- whether each frame is one single coherent scene, not split-screen or multi-panel
- whether the protagonist identity from the input image is preserved
- whether protagonist species and visible color are preserved
- whether the event/action is visible
- whether required objects/background are visible
- whether emotional expression and emotional cause are visible
- whether frame-to-frame continuity is maintained

DCEE plan:
{dce_plan}

Emotion arc:
{emotion_arc}

Storyboard:
{storyboard}
""".strip()


def dcee_branch_plan_prompt(seed: dict, abstract: str, num_candidates: int = 4) -> str:
    return dcee_plan_prompt(seed, abstract, forbidden_entities=[], protagonist_only=True)


def dcee_candidate_selection_prompt(seed: dict, abstract: str, candidates: list) -> str:
    return f"""
Select the best grounded V21 DCEE plan candidate.
Return valid JSON with keys: selected_candidate_id, scores, reason.
Prefer the candidate that:
- preserves the protagonist identity from the input image
- avoids ungrounded secondary characters
- creates a clear Desire-Conflict-Event-Ending Emotion chain
- can be rendered as one single-scene image per frame

Seed:
{seed}

Abstract:
{abstract}

Candidates:
{candidates}
""".strip()


def dce_plan_prompt(seed: dict, abstract: str) -> str:
    return dcee_plan_prompt(seed, abstract, forbidden_entities=[], protagonist_only=True)


def emotion_delta_text(prev_emotion: str | None, cur_emotion: str, intensity: int) -> str:
    if not prev_emotion:
        return f"establish {cur_emotion} with visible intensity {intensity}/5"
    if prev_emotion == cur_emotion:
        return f"maintain {cur_emotion}, visible intensity {intensity}/5"
    return f"visible transition from {prev_emotion} to {cur_emotion}; show it through face, body, lighting, weather, and story evidence"


def storyboard_prompt(seed: dict, abstract: str, dce_plan: dict, emotion_arc: dict, num_frames: int) -> str:
    return f"""
Create a {num_frames}-frame grounded V21 storyboard.
Each frame must be one single coherent scene.
Each frame must include:
- exact protagonist identity from input image
- one visible action
- one visible emotional state
- required props/background only
- no ungrounded secondary characters
- no split-screen, comic panels, or multi-scene layout

Seed:
{seed}

Abstract:
{abstract}

DCEE plan:
{dce_plan}

Emotion arc:
{emotion_arc}

Return JSON array only.
""".strip()


def canonicalize_storyboard_prompt(seed: dict, dce_plan: dict, storyboard: list) -> str:
    return f"""
Canonicalize this V21 storyboard for image generation.
Replace vague references with explicit protagonist-centered visual details.
Preserve input-image identity.
Do not add ungrounded characters, props, or scene elements.
Ensure each frame is one single coherent scene.

Seed:
{seed}

DCEE plan:
{dce_plan}

Storyboard:
{storyboard}

Return the same JSON array.
""".strip()


def frame_prompt(frame: dict, dce_plan: dict, emotion_arc: dict, memory: dict, style: str, input_image_summary: dict | None) -> str:
    return f"""
{style}, full-color cinematic storybook illustration.
V22 STORY-LOCKED SINGLE-SCENE RENDERING:
- Generate one single coherent scene for this frame.
- Do not create split-screen, comic panels, storyboard sheets, collage, diptych, or multiple moments in one image.
- Show exactly one protagonist unless the input explicitly grounded another character.
- Preserve the protagonist identity from the input image.
- Preserve species, visible color, body shape, age impression, and distinctive traits.
- Render only the exact story content specified below.
- Do not add extra objects, props, animals, people, or scene elements that are not listed.

Exact story sentence: {frame.get('story_sentence') or frame.get('caption')}
Image-friendly sentence: {frame.get('image_sentence') or frame.get('story_sentence') or frame.get('caption')}
Visible protagonist action: {frame.get('event')}
Visible cause/evidence: {frame.get('event_grounding')}
Allowed visual inventory only: {frame.get('must_show') or frame.get('key_objects')}
Protagonist identity lock: {frame.get('character_reference_prompt')}
Emotion: {frame.get('emotion')} intensity {frame.get('emotion_intensity')}/5; {frame.get('emotion_visual_rule')}
World/background: {frame.get('scene_location')}, {frame.get('time_of_day')}, {frame.get('weather')}, {frame.get('atmosphere')}, {frame.get('environment_details')}
Continuity memory: {memory}
Input image summary: {input_image_summary}
Quality: {QUALITY_SUFFIX}
Negative: {NEGATIVE_PROMPT}
""".strip()



def candidate_selection_prompt(frame: dict, candidate_notes: list) -> str:
    return f"""
Select the best candidate for this V22 visual storytelling frame.
Return JSON with selected_candidate_id and reason.

Selection priority order:
1. exactly one protagonist only, no duplicate/second protagonist
2. one single coherent scene, no split-screen/multi-panel
3. exact story sentence alignment
4. main action visibility
5. required object/background visibility
6. emotion visibility
7. protagonist identity consistency from input image
8. image quality only after all story constraints

Frame:
{frame}

Candidate notes:
{candidate_notes}
""".strip()
