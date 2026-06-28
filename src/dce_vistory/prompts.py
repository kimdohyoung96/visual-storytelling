from __future__ import annotations

from typing import Any, Dict, List

SYSTEM_NARRATIVE = """
You are a research-grade visual storytelling planner.
All stories must be grounded only in the given text, image summary, protagonist specification, and simple input metadata.
Never import characters, occupations, props, or scenes from unrelated example stories.
If JSON is requested, return concise valid JSON only.
Core structure: Desire -> Conflict -> Event Chain -> Ending Emotion (DCEE).

V30 POLICY:
- All generated text fields must be in English.
- Protagonist-only visual storytelling is the default.
- The story must center on the protagonist only.
- Do not create secondary characters, animal friends, villagers, humans, helpers, enemies, woodcutters, fairies, or crowds unless the user explicitly provides them in input.
- Background objects and simple props are allowed, but they must serve the protagonist's action.
- All conflicts must come from the environment, the protagonist's goal, the protagonist's mistake, a lost object, or the protagonist's internal emotional state.
- Never resolve or drive the story through another agent.
- If an object is difficult to render or would confuse the input-image identity, remove it from the story rather than making it a new character.
""".strip()

SYSTEM_VLM = """
You are a strict visual narrative evaluator. Return concise valid JSON only.
Evaluate whether an image shows the protagonist, planned event, required props, emotion, world state, and identity consistency.
""".strip()

QUALITY_SUFFIX = (
    "full-color cinematic storybook illustration, rich natural colors, emotionally meaningful color palette, "
    "clear protagonist action, clear prop evidence, expressive face, expressive body language, detailed background, "
    "coherent anatomy, cinematic lighting, sharp focus, professional illustration quality"
)

NEGATIVE_PROMPT = (
    "monochrome, black and white, grayscale, pencil sketch, line art only, colorless image, "
    "extra character, secondary character, crowd, unrelated animal, unrelated human, duplicated protagonist, "
    "extra prop, unrelated object, unrelated background object, extra scene element not mentioned in the story, "
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
Describe the input image for grounded V19 protagonist-only DCEE visual storytelling.
Return JSON with: caption, characters, protagonist_description, setting, objects, background_objects, mood, inferred_plot_hint, time_of_day, weather, environment_details.
Image path: {image_path}
Text prompt: {sample.get('text_prompt')}
Protagonist: {sample.get('protagonist')}
Target ending emotion: {sample.get('target_ending_emotion')}
Only describe what is actually present or safely inferable.
Do not invent secondary characters.
""".strip()


def story_seed_prompt(sample: dict, image_summary: dict | None, forbidden_entities: List[str], protagonist_only: bool = True) -> str:
    return f"""
Create a GROUNDED V19 story seed for protagonist-only visual storytelling.
Use only the user input text, image summary, protagonist specification, and simple metadata.

Input sample:
{sample}

Image summary:
{image_summary}

Forbidden entities unless explicitly grounded in the input/image summary:
{forbidden_entities}

V28 protagonist-only caption-grounded policy:
- protagonist_only = {protagonist_only}
- The story must center on the protagonist only.
- Do not invent friends, wild animal friends, helpers, enemies, humans, woodcutters, fairies, villagers, crowds, or other agents.
- Keep only simple props and background objects needed for the protagonist's action.
- If the input contains only one protagonist image, do not create additional character agents.

Return JSON with keys:
setting, objects, characters, mood, visual_symbols, world_context, character_profiles.

Requirements:
- characters must contain only the protagonist unless the input explicitly names another character.
- objects must be props/background, not new characters.
- protagonist identity must stay stable across all frames.
- if the protagonist is a panda, do not invent a woodcutter, animal friends, or other pandas.
- character_profiles must include: name, role, age_group, gender, face, hair, body, outfit, signature_items, color_palette, identity_anchor_prompt.
""".strip()


def story_abstract_prompt(seed: dict, forbidden_entities: List[str], protagonist_only: bool = True) -> str:
    return f"""
Write one grounded abstract paragraph for a V19 protagonist-only visual story.
The story must follow Desire -> Conflict -> Event Chain -> Ending Emotion.
Use only grounded entities from the seed.

Forbidden ungrounded entities:
{forbidden_entities}

V28 policy:
- protagonist_only = {protagonist_only}
- The protagonist is the only active character.
- Conflict must come from environment, lost object, obstacle, weather, distance, hunger, mistake, or internal emotion.
- Do not create friends, wild animal friends, helpers, enemies, humans, woodcutters, fairies, crowds, or other agents.
- Props/background objects are allowed only if they support the protagonist's visible action.

Seed:
{seed}

Return plain text only.
""".strip()


def dcee_plan_prompt(seed: dict, abstract: str, forbidden_entities: List[str], protagonist_only: bool = True) -> str:
    return f"""
Create one grounded V19 DCEE plan from the seed and abstract.

Forbidden ungrounded entities:
{forbidden_entities}

V28 protagonist-only caption-grounded policy:
- protagonist_only = {protagonist_only}
- The protagonist is the only active character.
- Do not introduce secondary characters or other agents.
- Conflict must be protagonist-centered: environment obstacle, lost prop, hunger, weather, distance, injury, fear, or internal dilemma.
- Do not introduce woodcutter, axe, fairy, animal friends, wild animal friends, villagers, helpers, or enemies unless explicitly grounded in input.

Return JSON with keys:
protagonist, desire, fear, misbelief, obstacle, conflict, event_chain, event_spine, turning_point,
target_ending_emotion, ending_state, moral_or_theme, planning_structure.

Requirements:
- events must be concrete and visually drawable.
- each event must show what the protagonist does, sees, loses, finds, carries, reaches, drops, searches, sits near, or walks away from.
- event_chain must not depend on another character's action.
- event_chain must contain short concrete event objects.

Seed:
{seed}

Abstract:
{abstract}
""".strip()


def emotion_arc_prompt(seed: dict, abstract: str, dce_plan: dict, num_frames: int) -> str:
    return f"""
Create an emotion arc with exactly {num_frames} states for the selected V19 protagonist-only DCEE plan.
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
Generate ONLY the next protagonist-centered frame description in English for frame {frame_index+1} of {num_frames}.
The output sentence will be used directly for image generation, so it must be visually concrete, single-scene, story-faithful, and protagonist-only.

Grounding rules:
- All output text must be English.
- Use only grounded entities from the seed, current DCEE plan, and previous story.
- Forbidden ungrounded entities: {forbidden_entities}
- protagonist_only = {protagonist_only}
- The protagonist is the only character/agent allowed in the story.
- Do not create any new character, friend, animal friend, helper, enemy, human, child, woodcutter, fairy, crowd, or duplicate protagonist.
- The protagonist must remain the subject of every sentence.
- The sentence must naturally continue story_so_far.
- Each frame must be one single scene with one main visible action.
- Conflict must come from environment, object state, search, movement, weather, distance, obstacle, loss, or internal emotion.
- If the sentence mentions an object, that object must be concrete, grounded, and drawable.
- If the object is not grounded in input/seed/story history, do not mention it.
- Keep the frame easy to draw: one protagonist, one action, one place, one emotional cue.
- Prefer literal, visually specific wording over abstract narration.

Allowed visual elements:
- protagonist
- protagonist's simple grounded props from input/seed
- grounded background objects such as forest, river, bamboo, rock, leaf, rain, path, hill, shadow, sunlight, water, sky

Desired output JSON keys:
sentence, image_caption_en, subject, action, action_en, object, location, location_en, weather, atmosphere, emotion, emotion_intensity,
visible_cause, visible_cause_en, required_objects, required_objects_en, background_elements, background_elements_en, supporting_cast, continuity_notes.

Strict output rules:
- sentence must be an English frame caption.
- image_caption_en must be the same English frame caption, rewritten only if needed for cleaner image generation.
- subject must be the protagonist only.
- supporting_cast must be [].
- sentence and image_caption_en must contain exactly one protagonist and exactly one primary visible action by the protagonist.
- sentence and image_caption_en must include one concrete place/background.
- required_objects / required_objects_en must contain only visible props/background objects required by this exact caption.
- background_elements / background_elements_en must contain only visible scene elements required by this exact caption.
- do not include humans, children, extra animals, reflections that look like another protagonist, or a second protagonist in any field.
- prefer full-body or medium-wide situations that are easy to render as a story frame.
- frame emotion target: {target_emotion}
- frame emotion intensity target: {intens}

Seed:
{seed}

DCEE plan:
{dce_plan}

Story so far:
{story_so_far}

Previous selected frame summary / feedback:
{previous_frame}
""".strip()


# ---------------------------------------------------------------------
# Backward-compatible prompt functions
# ---------------------------------------------------------------------
def dcee_branch_plan_prompt(seed: dict, abstract: str, num_candidates: int = 4) -> str:
    return dcee_plan_prompt(seed, abstract, forbidden_entities=[])


def dcee_candidate_selection_prompt(seed: dict, abstract: str, candidates: list) -> str:
    return f"""
Select the best DCEE candidate for grounded protagonist-only visual storytelling.
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
Create a {num_frames}-frame grounded protagonist-only storyboard.
Each frame must include concrete protagonist event, visible evidence, emotion, location, and required props/background.
Do not introduce secondary characters or ungrounded template-story entities.
Seed: {seed}
Abstract: {abstract}
DCEE plan: {dce_plan}
Emotion arc: {emotion_arc}
Return JSON array only.
""".strip()


def canonicalize_storyboard_prompt(seed: dict, dce_plan: dict, storyboard: list) -> str:
    return f"""
Canonicalize this storyboard for protagonist-only image generation.
Replace vague references with explicit grounded protagonist action, props, and background.
Do not add ungrounded new characters or props.
Seed: {seed}
DCEE plan: {dce_plan}
Storyboard: {storyboard}
Return the same JSON array.
""".strip()


def eval_questions_prompt(dce_plan: dict, emotion_arc: dict, storyboard: list) -> str:
    return f"""
Generate VQA-style questions for grounded protagonist-only DCEE visual storytelling evaluation.
Return JSON with global_questions, frame_questions, ending_questions.
Questions must cover:
- exact story sentence alignment
- protagonist event visibility
- required prop/background visibility
- emotional cause visibility
- protagonist consistency
- world/background consistency
DCEE plan: {dce_plan}
Emotion arc: {emotion_arc}
Storyboard: {storyboard}
""".strip()



def frame_prompt(frame: dict, dce_plan: dict, emotion_arc: dict, memory: dict, style: str, input_image_summary: dict | None) -> str:
    caption = frame.get('image_caption_en') or frame.get('image_sentence') or frame.get('story_sentence') or frame.get('caption')
    return f"""
{style}, full-color cinematic storybook illustration.
V30 ENGLISH CAPTION-GROUNDED RENDERING RULES:
- Render the exact frame caption as one single coherent image.
- The caption is the primary contract; do not replace it with a generic portrait or unrelated scene.
- Show exactly one protagonist and no secondary characters.
- Render only the current caption content and listed visual inventory.
- Do not add extra objects, props, animals, humans, or scene elements that are not listed.
- Use the input-image protagonist identity consistently.
- Prefer medium or medium-wide composition that shows protagonist action, required object, and background.
Exact frame caption: {caption}
Story sentence: {frame.get('story_sentence') or frame.get('caption')}
Visible protagonist action: {frame.get('event')}
Visible cause/evidence: {frame.get('event_grounding')}
Allowed visual inventory only: {frame.get('must_show') or frame.get('key_objects')}
Emotion: {frame.get('emotion')} intensity {frame.get('emotion_intensity')}/5; {frame.get('emotion_visual_rule')}
World/background: {frame.get('scene_location')}, {frame.get('time_of_day')}, {frame.get('weather')}, {frame.get('atmosphere')}, {frame.get('environment_details')}
Continuity memory: {memory}
Quality: {QUALITY_SUFFIX}
Negative: {NEGATIVE_PROMPT}
""".strip()
