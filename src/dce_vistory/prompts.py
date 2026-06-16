from __future__ import annotations

from typing import Any, Dict

SYSTEM_NARRATIVE = """
You are a research-grade multimodal narrative planner.
You must explicitly model protagonist desire, conflict, causally ordered visual events, emotional transitions, and target ending emotion.
Each planned event must be visually drawable and must explain or intensify the protagonist's current emotion.
Return concise valid JSON whenever JSON is requested.
""".strip()

SYSTEM_VLM = """
You are a strict visual narrative evaluator.
Evaluate whether the image clearly shows the target emotion, the planned visual event, the cause of the emotion, protagonist identity, world state, colorfulness, and story alignment.
Return concise valid JSON only.
""".strip()

QUALITY_SUFFIX = (
    "full-color cinematic storybook illustration, rich natural color rendering, emotionally meaningful color palette, "
    "clear event-grounded emotional storytelling, strong facial acting, expressive body language, coherent anatomy, "
    "high detail environment, visually readable action, cinematic lighting, sharp focus, professional illustration quality"
)

NEGATIVE_PROMPT = (
    "monochrome, black and white, grayscale, pencil sketch, charcoal sketch, line art only, low quality, blurry, "
    "bad anatomy, extra fingers, deformed hands, distorted face, emotionless face, stiff pose, empty background, "
    "washed-out colors, flat lighting, weak expression, colorless image, random text, watermark, logo, cropped face"
)

EMOTION_RENDER_BOOK: Dict[str, Dict[str, str]] = {
    "joy": {"face":"bright eyes, raised cheeks, gentle open smile, visibly happy expression", "body":"open chest, lifted posture, relaxed shoulders, energetic or relieved body language", "lighting":"warm sunlight, clean highlights, bright atmosphere", "palette":"golden yellow, warm green, sky blue, vivid natural colors", "weather":"clear sky or clouds opening after tension", "composition":"open composition with visible space ahead and a sense of release"},
    "happiness": {"face":"bright eyes, raised cheeks, gentle open smile, visibly happy expression", "body":"open chest, lifted posture, relaxed shoulders, energetic or relieved body language", "lighting":"warm sunlight, clean highlights, bright atmosphere", "palette":"golden yellow, warm green, sky blue, vivid natural colors", "weather":"clear sky or clouds opening after tension", "composition":"open composition with visible space ahead and a sense of release"},
    "relief": {"face":"soft exhale, relaxed eyes, subtle grateful smile, tension leaving the face", "body":"shoulders dropping after tension, calm grounded posture, hands loosening", "lighting":"soft warm light after shadow", "palette":"warm beige, gentle green, soft amber, calm blue", "weather":"clouds clearing, gentle breeze, quiet air", "composition":"balanced frame with reduced visual pressure"},
    "hope": {"face":"focused eyes, slightly raised brows, determined but gentle mouth", "body":"forward-leaning stance, ready movement, cautious but open posture", "lighting":"soft warm directional light", "palette":"fresh green, warm earth tones, soft blue, early sunlight", "weather":"clear or lightly cloudy", "composition":"path or destination visible in front of the protagonist"},
    "sadness": {"face":"downcast eyes, tightened mouth, heavy eyelids, visibly sorrowful expression", "body":"slumped shoulders, stillness, lowered head, closed body posture", "lighting":"dim diffused light, soft shadow", "palette":"cool desaturated blue-gray with muted earth colors", "weather":"rain, overcast sky, mist, damp air", "composition":"negative space and isolation around the protagonist"},
    "regret": {"face":"downcast eyes, tense mouth, pained expression, gaze avoiding the viewer", "body":"slumped shoulders, hand near chest or face, hesitant stillness", "lighting":"dim side light with soft shadow", "palette":"muted blue-gray, faded brown, low saturation but still full color", "weather":"drizzle, mist, overcast sky", "composition":"lonely composition with visible evidence of a wrong choice"},
    "fear": {"face":"wide eyes, tense brows, tight jaw or parted lips, alarmed expression", "body":"defensive pose, recoiling posture, tense shoulders, guarded hands", "lighting":"hard contrast, looming shadows", "palette":"cold blue, gray, desaturated green with dark accents", "weather":"fog, storm, oppressive darkness, cold wind", "composition":"off-center framing with visible threat or uncertainty in the scene"},
    "anger": {"face":"furrowed brows, intense stare, clenched jaw, visibly angry expression", "body":"tight stance, forceful gesture, clenched fist, rigid shoulders", "lighting":"harsh directional light, dramatic contrast", "palette":"red-orange accents, dark browns, high contrast warm tones", "weather":"wind, dramatic clouds, heated atmosphere", "composition":"diagonal tension, compressed frame, confrontational staging"},
    "determination": {"face":"steady eyes, focused brows, firm mouth, unwavering expression", "body":"forward movement, strong stance, stable posture, purposeful hands", "lighting":"strong light cutting through shadow", "palette":"earth tones with controlled warm highlights and clear contrast", "weather":"wind or clearing clouds", "composition":"forward momentum and visible obstacle"},
    "doubt": {"face":"uncertain gaze, slightly tightened brows, hesitant mouth", "body":"paused pose, weight shifted back, uncertain gesture, guarded posture", "lighting":"muted side light, partial shadow", "palette":"cool gray-blue with pale greens and muted browns", "weather":"mist, thin clouds, still air", "composition":"ambiguous space, partially blocked path, visual uncertainty"},
    "anxiety": {"face":"worried eyes, tense brows, pressed lips, alert expression", "body":"tight shoulders, restless hands, slightly hunched posture", "lighting":"uneven cool light with mild shadow", "palette":"cold green-gray, muted blue, pale brown", "weather":"mist, cloudy sky, wind before rain", "composition":"crowded foreground or constricted path"},
}

def get_emotion_rule(emotion: str) -> Dict[str, str]:
    return EMOTION_RENDER_BOOK.get((emotion or '').lower().strip(), {
        "face":"clear readable emotional facial expression",
        "body":"clear readable emotional body posture",
        "lighting":"cinematic lighting aligned with the emotion",
        "palette":"full natural color palette aligned with the emotion",
        "weather":"weather aligned with the story situation",
        "composition":"composition that clearly reveals the emotional state",
    })

def emotion_rule_text(emotion: str) -> str:
    r = get_emotion_rule(emotion)
    return f"Facial expression: {r['face']}; Body posture: {r['body']}; Lighting: {r['lighting']}; Color palette: {r['palette']}; Weather/world: {r['weather']}; Composition: {r['composition']}"

def emotion_delta_text(prev_emotion: str | None, cur_emotion: str, intensity: int) -> str:
    if not prev_emotion:
        return f"Establish the starting emotion as {cur_emotion} with visible intensity {intensity}/5."
    if prev_emotion == cur_emotion:
        return f"Maintain {cur_emotion}, but make intensity {intensity}/5 clearly visible through the planned event."
    return f"Show a visible emotional transition from {prev_emotion} to {cur_emotion}; the current event must visually explain this transition."

def choose_shot_type(idx: int, total: int, narrative_function: str) -> str:
    nf = (narrative_function or '').lower()
    if idx == 0: return 'wide establishing shot'
    if 'climax' in nf or 'turning' in nf: return 'dramatic medium close-up'
    if 'resolution' in nf or 'ending' in nf or idx == total - 1: return 'medium-wide ending shot'
    if 'conflict' in nf or 'event' in nf or 'obstacle' in nf: return 'action-focused medium shot'
    if 'reaction' in nf or 'emotion' in nf: return 'emotional close-up'
    return 'medium shot'

def choose_camera_distance(shot_type: str) -> str:
    st = (shot_type or '').lower()
    if 'close' in st: return 'close'
    if 'wide' in st: return 'wide'
    return 'medium'

def image_understanding_prompt(image_path: str, sample: dict) -> str:
    return f"""
Describe the input image for multimodal story planning.
Focus on characters, clothing, setting, objects, mood, inferred plot hint, time of day, weather, and environment details.

Input image path: {image_path}
Text prompt: {sample.get('text_prompt')}
Protagonist: {sample.get('protagonist')}
Target ending emotion: {sample.get('target_ending_emotion')}

Return JSON with: caption, characters, setting, objects, mood, inferred_plot_hint, time_of_day, weather, environment_details.
""".strip()

def story_seed_prompt(sample: dict, image_summary: dict | None) -> str:
    return f"""
Create a multimodal story seed and character bible.

Text prompt: {sample.get('text_prompt')}
Protagonist: {sample.get('protagonist')}
Target ending emotion: {sample.get('target_ending_emotion')}
Genre: {sample.get('genre')}
Style: {sample.get('style')}
Image summary: {image_summary}

Return JSON with setting, objects, characters, mood, visual_symbols, world_context, and character_profiles.
The protagonist profile must be visually specific enough to preserve identity across frames.
""".strip()

def story_abstract_prompt(seed: dict) -> str:
    return f"""
Write a 4-6 sentence story abstract.

Seed: {seed}

Requirements:
- Introduce protagonist desire.
- Introduce a meaningful conflict.
- Include a chain of concrete visual events that causes the target ending emotion.
- Keep protagonist identity stable and every event visually drawable.
""".strip()

def dce_plan_prompt(seed: dict, abstract: str) -> str:
    return f"""
Create a DCEE plan: Desire, Conflict, Event Chain, and Ending Emotion.

Seed: {seed}
Story abstract: {abstract}

Return valid JSON with:
protagonist, desire, fear, misbelief, obstacle, conflict,
event_spine, turning_point, target_ending_emotion, ending_state, moral_or_theme.

Important definition:
- event_spine must be a causally ordered list of visual events.
- Each event must be drawable as an image frame.
- Each event must cause, reveal, or intensify the protagonist's emotion.
- The final event must make the ending emotion believable.
""".strip()

def emotion_arc_prompt(seed: dict, abstract: str, dce_plan: dict, num_frames: int) -> str:
    return f"""
Create an emotion arc with exactly {num_frames} steps.

Seed: {seed}
Abstract: {abstract}
DCEE plan: {dce_plan}

Return valid JSON with states, intensities, and rationale.
Each emotional transition must be justified by the corresponding event in event_spine.
The final state must match the target ending emotion or a natural visual synonym.
""".strip()

def storyboard_prompt(seed: dict, abstract: str, dce_plan: dict, emotion_arc: dict, num_frames: int) -> str:
    return f"""
Create a {num_frames}-frame event-grounded storyboard.

Seed: {seed}
Abstract: {abstract}
DCEE plan: {dce_plan}
Emotion arc: {emotion_arc}

Return a JSON array. Each item must include:
frame_id, caption, narrative_function, event, event_causal_role, protagonist_state, desire_link, conflict_level,
emotion, emotion_intensity, visual_focus, key_objects, facial_cue, body_cue, event_cue, scene_cue, cinematic_cue,
scene_location, time_of_day, weather, atmosphere, environment_details, supporting_cast, scene_transition.

Important:
- Each frame event must be visible in the image.
- Each event must explain why the protagonist feels the planned emotion.
- Include emotion_evidence implicitly through event, key_objects, and scene details.
- Use full-color visual storytelling, never monochrome or grayscale.
- The final frame must visibly ground the target ending emotion.
""".strip()

def frame_prompt(frame: dict, dce_plan: dict, emotion_arc: dict, memory: dict, style: str, input_image_summary: dict | None) -> str:
    emotion = frame.get('emotion', '')
    env_details = ', '.join(frame.get('environment_details', []))
    must_show = ', '.join(frame.get('must_show', []))
    emotion_evidence = ', '.join(frame.get('emotion_evidence', []))
    return f"""
Generate one event-grounded frame of a visual story.

[STYLE] {style}, full-color cinematic storybook illustration, richly colored.
[CHARACTER] {frame.get('character_identity')} {frame.get('character_reference_prompt')}
[EVENT] {frame.get('event')}
[EVENT CAUSAL ROLE] {frame.get('event_causal_role')}
[EMOTION] {emotion} intensity {frame.get('emotion_intensity')}/5
[EMOTION RULE] {frame.get('emotion_visual_rule') or emotion_rule_text(emotion)}
[EMOTION EVIDENCE] {emotion_evidence}
[WORLD] location={frame.get('scene_location')}; time={frame.get('time_of_day')}; weather={frame.get('weather')}; atmosphere={frame.get('atmosphere')}; details={env_details}
[CAMERA/COLOR] shot={frame.get('shot_type')}; lighting={frame.get('lighting_style')}; palette={frame.get('color_palette')}
[MUST SHOW] {must_show}
[DCEE PLAN] {dce_plan}
[MEMORY] {memory}

The image must show the event and the visual cause of the protagonist's emotion. Full-color only.
""".strip()

def eval_questions_prompt(dce_plan: dict, emotion_arc: dict, storyboard: list) -> str:
    return f"""
Generate evaluation questions for visual story assessment.
DCEE plan: {dce_plan}
Emotion arc: {emotion_arc}
Storyboard: {storyboard}
Return valid JSON with global_questions, frame_questions, ending_questions.
Questions should assess event grounding, event-emotion causal consistency, emotion visibility, emotion cause visibility, world fidelity, colorfulness, and identity consistency.
""".strip()

def vlm_eval_prompt(dce_plan: dict, emotion_arc: dict, storyboard: list, questions: dict) -> str:
    return f"""
Evaluate the generated visual story by looking at the images.
DCEE plan: {dce_plan}
Emotion arc: {emotion_arc}
Storyboard: {storyboard}
Questions: {questions}
Return valid JSON with text_image_alignment, character_consistency, image_quality, colorfulness, event_grounding, event_emotion_causal_consistency, emotion_visibility, emotion_cause_visibility, scene_consistency, world_fidelity, ending_emotion_accuracy, narrative_coherence, interestingness, rationale.
""".strip()

def ending_candidate_eval_prompt(final_frame: dict, dce_plan: dict) -> str:
    return f"""
Evaluate ending candidates for the final DCEE frame.
Final frame: {final_frame}
DCEE plan: {dce_plan}
Return JSON candidate_scores with identity_consistency, image_quality, colorfulness, world_fidelity, event_grounding, ending_emotion_accuracy, emotion_visibility, event_emotion_causal_consistency, overall, reason.
""".strip()
