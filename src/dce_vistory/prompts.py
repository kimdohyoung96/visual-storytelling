from __future__ import annotations

from typing import Any, Dict

SYSTEM_NARRATIVE = """
You are a research-grade visual storytelling planner. Return concise valid JSON whenever JSON is requested.
Plan stories for ending-controllable causal visual storytelling. The core structure is DCEE:
Desire -> Conflict -> Event Chain -> Ending Emotion. Each event must be visually drawable and must explain or intensify the protagonist's emotion.
""".strip()

SYSTEM_VLM = """
You are a strict visual narrative evaluator. Return concise valid JSON only.
Evaluate whether an image shows the planned DCEE event, evidence, emotion, world state, and character identity.
""".strip()

QUALITY_SUFFIX = (
    "full-color cinematic storybook illustration, rich natural colors, emotionally meaningful color palette, "
    "clear event grounding, clear visual evidence, expressive face, expressive body language, detailed background, "
    "coherent anatomy, cinematic lighting, sharp focus, professional illustration quality"
)

NEGATIVE_PROMPT = (
    "monochrome, black and white, grayscale, pencil sketch, charcoal sketch, line art only, colorless image, "
    "missing event, missing visual evidence, emotionless face, weak expression, stiff pose, portrait only, "
    "empty background, low quality, blurry, bad anatomy, distorted face, watermark, text"
)

EMOTION_RENDER_BOOK: Dict[str, Dict[str, str]] = {
    "joy": {"face":"bright eyes, raised cheeks, open smile", "body":"open chest, lifted posture, relaxed shoulders", "lighting":"warm sunlight, clean highlights", "palette":"golden yellow, warm green, sky blue", "weather":"clear sky or clouds opening", "composition":"open composition with visible space ahead"},
    "relief": {"face":"soft exhale, relaxed eyes, grateful subtle smile", "body":"shoulders dropping after tension, grounded posture", "lighting":"soft warm light after shadow", "palette":"warm beige, gentle green, soft amber", "weather":"clouds clearing, gentle breeze", "composition":"balanced frame with reduced pressure"},
    "hope": {"face":"focused eyes, slightly raised brows, determined gentle mouth", "body":"forward-leaning stance, ready movement", "lighting":"soft warm directional light", "palette":"fresh green, warm earth tones, soft blue", "weather":"clear or lightly cloudy", "composition":"visible path or destination"},
    "sadness": {"face":"downcast eyes, tightened mouth, heavy eyelids", "body":"slumped shoulders, lowered head, closed posture", "lighting":"dim diffused light, soft shadow", "palette":"cool desaturated blue-gray with muted earth colors", "weather":"rain, overcast sky, mist", "composition":"negative space and isolation"},
    "regret": {"face":"downcast eyes, tense mouth, pained expression", "body":"slumped shoulders, hand near chest or face", "lighting":"dim side light", "palette":"muted blue-gray, faded brown, low saturation but full color", "weather":"drizzle, mist, overcast sky", "composition":"lonely frame with visible evidence of a wrong choice"},
    "fear": {"face":"wide eyes, tense brows, tight jaw", "body":"defensive pose, recoiling posture", "lighting":"hard contrast, looming shadows", "palette":"cold blue, gray, desaturated green", "weather":"fog, storm, oppressive darkness", "composition":"off-center framing with visible threat"},
    "anger": {"face":"furrowed brows, intense stare, clenched jaw", "body":"rigid shoulders, clenched fist, forceful gesture", "lighting":"harsh directional light", "palette":"red-orange accents, dark browns", "weather":"wind and dramatic clouds", "composition":"diagonal tension and compressed frame"},
    "determination": {"face":"steady eyes, focused brows, firm mouth", "body":"forward movement, strong stance", "lighting":"strong light cutting through shadow", "palette":"earth tones with warm highlights", "weather":"wind or clearing clouds", "composition":"forward momentum and visible obstacle"},
    "doubt": {"face":"uncertain gaze, tightened brows, hesitant mouth", "body":"paused pose, weight shifted back", "lighting":"muted side light", "palette":"cool gray-blue with pale greens", "weather":"mist, thin clouds", "composition":"ambiguous space, partially blocked path"},
    "anxiety": {"face":"worried eyes, tense brows, pressed lips", "body":"tight shoulders, restless hands", "lighting":"uneven cool light", "palette":"cold green-gray, muted blue", "weather":"cloudy wind before rain", "composition":"constricted path or crowded foreground"},
}

def get_emotion_rule(emotion: str) -> Dict[str, str]:
    return EMOTION_RENDER_BOOK.get((emotion or '').lower().strip(), {
        'face':'readable emotional facial expression', 'body':'readable emotional body posture',
        'lighting':'cinematic lighting aligned with emotion', 'palette':'full natural color palette aligned with emotion',
        'weather':'weather aligned with story situation', 'composition':'composition that reveals emotional state'
    })

def emotion_rule_text(emotion: str) -> str:
    r = get_emotion_rule(emotion)
    return f"face: {r['face']}; body: {r['body']}; lighting: {r['lighting']}; palette: {r['palette']}; weather: {r['weather']}; composition: {r['composition']}"

def emotion_delta_text(prev_emotion: str | None, cur_emotion: str, intensity: int) -> str:
    if not prev_emotion:
        return f"establish {cur_emotion} with visible intensity {intensity}/5"
    if prev_emotion == cur_emotion:
        return f"maintain {cur_emotion}, intensity {intensity}/5"
    return f"visible transition from {prev_emotion} to {cur_emotion}; show it through face, body, color, light, weather and evidence"

def choose_shot_type(idx: int, total: int, narrative_function: str) -> str:
    nf=(narrative_function or '').lower()
    if idx == 0: return 'wide establishing shot'
    if 'climax' in nf or 'turning' in nf: return 'dramatic medium close-up'
    if 'ending' in nf or 'resolution' in nf or idx == total-1: return 'medium-wide ending shot'
    if 'conflict' in nf or 'obstacle' in nf: return 'action-focused medium shot'
    if 'reaction' in nf or 'emotion' in nf: return 'emotional close-up'
    return 'medium shot'

def choose_camera_distance(shot_type: str) -> str:
    st=(shot_type or '').lower()
    if 'close' in st: return 'close'
    if 'wide' in st: return 'wide'
    return 'medium'

def image_understanding_prompt(image_path: str, sample: dict) -> str:
    return f"""
Describe the input image for DCEE visual storytelling.
Return JSON with: caption, characters, setting, objects, mood, inferred_plot_hint, time_of_day, weather, environment_details.
Image path: {image_path}
Text prompt: {sample.get('text_prompt')}
Protagonist: {sample.get('protagonist')}
Target ending emotion: {sample.get('target_ending_emotion')}
""".strip()

def story_seed_prompt(sample: dict, image_summary: dict | None) -> str:
    return f"""
Create a story seed, character bible, object/evidence list, and world context.
Input sample: {sample}
Image summary: {image_summary}
Return JSON with: setting, objects, characters, mood, visual_symbols, world_context, character_profiles.
character_profiles must include name, role, face, hair, body, outfit, signature_items, color_palette, identity_anchor_prompt.
""".strip()

def story_abstract_prompt(seed: dict) -> str:
    return f"""
Write a 4-6 sentence abstract for an ending-controllable visual story. The abstract must imply Desire, Conflict, Event Chain, and Ending Emotion.
Seed: {seed}
""".strip()

def dcee_branch_plan_prompt(seed: dict, abstract: str, num_candidates: int = 4) -> str:
    return f"""
Generate {num_candidates} alternative DCEE candidate plans for the same seed and target ending emotion.
Each candidate must contain: candidate_id, desire, conflict, conflict_escalation, event_chain, turning_point, ending_emotion, ending_state, rationale.
Each event_chain item must contain: event_id, event, causal_role, visual_grounding, emotion_effect, key_objects, evidence_objects.
The candidates should explore different Desire->Conflict routes before choosing events. The best event chain should make the target ending emotion visually believable.
Seed: {seed}
Abstract: {abstract}
Return JSON: {{"candidates": [...]}}
""".strip()

def dcee_candidate_selection_prompt(seed: dict, abstract: str, candidates: list) -> str:
    return f"""
Select the best DCEE candidate for visual storytelling. Score each candidate using:
causal_coherence, ending_emotion_fit, event_drawability, evidence_visibility, conflict_strength, novelty, frame_coverage.
Return JSON: {{"selected_candidate_id": "...", "scores": [...], "reason": "..."}}
Seed: {seed}
Abstract: {abstract}
Candidates: {candidates}
""".strip()

def dce_plan_prompt(seed: dict, abstract: str) -> str:
    # Backward compatible name, now DCEE.
    return dcee_branch_plan_prompt(seed, abstract, num_candidates=1)

def emotion_arc_prompt(seed: dict, abstract: str, dce_plan: dict, num_frames: int) -> str:
    return f"""
Create an emotion arc with exactly {num_frames} states for the selected DCEE event chain.
Return JSON: {{"states": [...], "intensities": [...], "valence_curve": [...], "arousal_curve": [...], "suspense_curve": [...], "rationale": "..."}}
The last emotion must match the target ending emotion or a natural synonym.
Seed: {seed}
Abstract: {abstract}
DCEE plan: {dce_plan}
""".strip()

def storyboard_prompt(seed: dict, abstract: str, dce_plan: dict, emotion_arc: dict, num_frames: int) -> str:
    return f"""
Create a {num_frames}-frame DCEE storyboard. Return a JSON array.
Each frame must include: frame_id, caption, narrative_function, event, event_causal_role, event_grounding, protagonist_state, desire_link, conflict_level, emotion, emotion_intensity, visual_focus, key_objects, evidence_objects, facial_cue, body_cue, event_cue, scene_cue, cinematic_cue, scene_location, time_of_day, weather, atmosphere, environment_details, supporting_cast, scene_transition.
Every frame must show the DCEE event, the visual evidence, and why the event causes the emotion.
Seed: {seed}
Abstract: {abstract}
DCEE plan: {dce_plan}
Emotion arc: {emotion_arc}
""".strip()

def canonicalize_storyboard_prompt(seed: dict, dce_plan: dict, storyboard: list) -> str:
    return f"""
Canonicalize the storyboard for diffusion generation. Replace pronouns and vague references with explicit entity/object names. Convert abstract events into drawable visual events.
Return the same JSON array with clarified event, key_objects, evidence_objects, and must_show fields.
Seed: {seed}
DCEE plan: {dce_plan}
Storyboard: {storyboard}
""".strip()

def eval_questions_prompt(dce_plan: dict, emotion_arc: dict, storyboard: list) -> str:
    return f"""
Generate VQA-style questions for DCEE visual storytelling evaluation.
Return JSON with global_questions, frame_questions, ending_questions.
Questions must cover event grounding, evidence visibility, event-emotion causal consistency, character consistency, world consistency, colorfulness, and ending emotion accuracy.
DCEE plan: {dce_plan}
Emotion arc: {emotion_arc}
Storyboard: {storyboard}
""".strip()

def frame_prompt(frame: dict, dce_plan: dict, emotion_arc: dict, memory: dict, style: str, input_image_summary: dict | None) -> str:
    return f"""
{style}, full-color cinematic storybook illustration.
Character: {frame.get('character_identity')} {frame.get('character_reference_prompt')}
DCEE event: {frame.get('event')}
Causal role: {frame.get('event_causal_role')}
Visual grounding/evidence: {frame.get('event_grounding')}; {frame.get('emotion_evidence')}; must show {frame.get('must_show')}
Emotion: {frame.get('emotion')} intensity {frame.get('emotion_intensity')}/5; {frame.get('emotion_visual_rule')}
World: {frame.get('scene_location')}, {frame.get('time_of_day')}, {frame.get('weather')}, {frame.get('atmosphere')}, {frame.get('environment_details')}
Camera/color: {frame.get('shot_type')}, {frame.get('camera_distance')}, {frame.get('lighting_style')}, {frame.get('color_palette')}
Memory: {memory}
Quality: {QUALITY_SUFFIX}
Never grayscale. The image must show what happened and why the protagonist feels this emotion.
""".strip()
