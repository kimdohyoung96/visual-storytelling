
from __future__ import annotations
from typing import Any, Dict, List
import re


def clean_text(x: Any) -> str:
    return re.sub(r'\s+', ' ', str(x or '').replace('\n', ' ')).strip()


def compact_list(values, limit: int = 4) -> List[str]:
    out = []
    for v in values or []:
        t = clean_text(v)
        if t and t not in out:
            out.append(t)
    return out[:limit]


def limit_words(text: str, n: int = 75) -> str:
    words = clean_text(text).split()
    return ' '.join(words[:n])


def extract_visual_contract(packet) -> Dict[str, Any]:
    m = packet.control_metadata or {}
    character = m.get('character', {}) or {}
    world = m.get('world', {}) or {}
    emotion = m.get('emotion', {}) or {}
    contract = {
        'story_sentence': clean_text(m.get('story_sentence', '')),
        'frame_goal': clean_text(m.get('frame_goal_text', '')),
        'event': clean_text(m.get('dcee_event_text', '')),
        'event_grounding': clean_text(m.get('event_grounding', '')),
        'evidence_objects': compact_list(m.get('evidence_objects', []), 5),
        'emotion_evidence': compact_list(m.get('emotion_evidence', []), 5),
        'must_show': compact_list(m.get('must_show', []), 6),
        'character_name': clean_text(character.get('name', 'protagonist')),
        'identity_prompt': clean_text(character.get('identity_prompt', '')),
        'outfit_prompt': clean_text(character.get('outfit_prompt', '')),
        'signature_items': compact_list(character.get('signature_items', []), 4),
        'scene_location': clean_text(world.get('scene_location', '')),
        'time_of_day': clean_text(world.get('time_of_day', '')),
        'weather': clean_text(world.get('weather', '')),
        'atmosphere': clean_text(world.get('atmosphere', '')),
        'environment_details': compact_list(world.get('environment_details', []), 5),
        'emotion': clean_text(emotion.get('emotion', '')),
        'emotion_intensity': int(emotion.get('intensity', 3) or 3),
        'facial_rule': clean_text(emotion.get('facial_rule', '')),
        'body_rule': clean_text(emotion.get('body_rule', '')),
        'lighting_rule': clean_text(emotion.get('lighting_rule', '')),
        'color_rule': clean_text(emotion.get('color_rule', '')),
        'composition_rule': clean_text(emotion.get('composition_rule', '')),
        'salient_history': clean_text(m.get('history_text', '')),
        'continuity': clean_text(m.get('continuity_text', '')),
    }
    # Infer an action-centric short event from verbose text if possible.
    event_short = contract['event']
    if 'DCEE visible event:' in event_short:
        event_short = event_short.split('DCEE visible event:')[-1].split(';')[0].strip()
    contract['event_short'] = clean_text(event_short)
    return contract


def build_prompt_variant(contract: Dict[str, Any], mode: str = 'event_first') -> str:
    identity = contract['identity_prompt'] or contract['character_name']
    action = contract['event_short'] or contract['story_sentence']
    cause = contract['event_grounding'] or ', '.join(contract['emotion_evidence'])
    evidence = ', '.join(contract['must_show'] or contract['evidence_objects'] or contract['emotion_evidence'])
    world = ', '.join([x for x in [contract['scene_location'], contract['time_of_day'], contract['weather'], contract['atmosphere']] if x])
    env = ', '.join(contract['environment_details'])
    emotion = contract['emotion']
    face = contract['facial_rule'] or f'clear facial expression of {emotion}'
    body = contract['body_rule'] or f'body language showing {emotion}'
    color = contract['color_rule'] or 'rich natural colors'
    light = contract['lighting_rule'] or 'cinematic lighting'
    comp = contract['composition_rule'] or 'story-focused composition'
    story_sentence = contract['story_sentence']
    sig_items = ', '.join(contract['signature_items'])

    if mode == 'event_first':
        txt = f"full-color cinematic storybook illustration, same protagonist {identity}. Scene: {world}. Action: {action}. Story beat: {story_sentence}. Visible evidence: {evidence}. Show {emotion} with {face} and {body}. Cause visible: {cause}. Include {env}. {light}, {color}, {comp}. Not a portrait only."
    elif mode == 'evidence_first':
        txt = f"full-color narrative illustration. Same protagonist {identity}. The scene must clearly show {evidence}. These visual clues must explain why the protagonist feels {emotion}. Story beat: {story_sentence}. Action visible: {action}. Background: {world}, {env}. {face}; {body}; {light}; {color}."
    elif mode == 'emotion_first':
        txt = f"full-color expressive story illustration of the same protagonist {identity}. The protagonist feels {emotion}; show it strongly through {face} and {body}. The emotional cause must be visible: {cause}. Story beat: {story_sentence}. Event happening now: {action}. Show {evidence}. Use {world}, {light}, {color}, {comp}."
    else:  # continuity/world-first
        txt = f"full-color continuous visual storytelling frame. Keep the same protagonist identity {identity} and signature items {sig_items}. Advance the story from prior history: {contract['salient_history']}. Current story sentence: {story_sentence}. Current event: {action}. Must show {evidence}. World state: {world}, {env}. Emotion: {emotion}. {face}; {body}; {light}; {color}; {comp}."
    return limit_words(txt, 75)


def build_negative_variant(packet) -> str:
    base = clean_text(getattr(packet, 'negative_prompt', ''))
    extra = 'generic portrait, static portrait, repeated pose, missing action, missing event, missing evidence, missing hand interaction, cropped out important object, empty background, weak emotion, unclear cause, inconsistent protagonist, grayscale, monochrome, black and white, sketch, line-art only, blurry, low quality, text, watermark'
    return limit_words(base + ', ' + extra, 70)


def local_story_keywords(contract: Dict[str, Any]) -> Dict[str, List[str]]:
    return {
        'event_keywords': compact_list([contract['event_short'], *contract['evidence_objects']], 8),
        'emotion_keywords': compact_list([contract['emotion'], *contract['emotion_evidence']], 8),
        'scene_keywords': compact_list([contract['scene_location'], contract['weather'], *contract['environment_details']], 8),
    }
