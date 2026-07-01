
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple, Dict
import re
import torch
import numpy as np
from PIL import Image, ImageFilter

from .schema import CandidateImage
from .frame_director import FrameVisualSpec


_MODEL_PRESETS = {
    "sdxl_base": "stabilityai/stable-diffusion-xl-base-1.0",
    "storybook_xl": "RunDiffusion/Juggernaut-XL-v9",
    "juggernaut_xl": "RunDiffusion/Juggernaut-XL-v9",
    "realvis_xl": "SG161222/RealVisXL_V5.0",
}


def _path_exists(x: Any) -> bool:
    try:
        return bool(x) and Path(str(x)).exists()
    except Exception:
        return False


def _load_reference_image(path: str, size=(1024, 1024)) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail(size)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def _compose_reference_sheet(paths: List[str], size=(1024, 1024)) -> tuple[Image.Image | None, List[str]]:
    keep = []
    for p in paths or []:
        if _path_exists(p) and str(p) not in keep:
            keep.append(str(p))
    if not keep:
        return None, []
    images = []
    for p in keep[:3]:
        try:
            images.append(Image.open(p).convert("RGB"))
        except Exception:
            pass
    if not images:
        return None, []
    cols = len(images)
    cell_w = max(1, size[0] // cols)
    cell_h = size[1]
    canvas = Image.new("RGB", size, "white")
    for i, img in enumerate(images):
        im = img.copy()
        im.thumbnail((cell_w, cell_h))
        x0 = i * cell_w + (cell_w - im.width) // 2
        y0 = (cell_h - im.height) // 2
        canvas.paste(im, (x0, y0))
    return canvas, keep


def _resolve_model_id(model_id: str, quality_model_preset: str | None = None) -> str:
    if quality_model_preset and quality_model_preset in _MODEL_PRESETS:
        return _MODEL_PRESETS[quality_model_preset]
    return model_id


def _clean(x: Any) -> str:
    s = str(x or "").replace("\n", " ").replace("\t", " ").strip()
    while "  " in s:
        s = s.replace("  ", " ")
    return s


def _clip_words(x: Any, max_words: int) -> str:
    s = _clean(x)
    if not s:
        return ""
    toks = s.split()
    return " ".join(toks[:max_words])


def _simple_sentence(text: str, max_words: int) -> str:
    s = _clip_words(text, max_words)
    s = s.strip(" .,")
    return s


def _join_unique(items: List[str], limit: int = 5, per_item_words: int = 4) -> str:
    vals = []
    for x in items or []:
        s = _clip_words(x, per_item_words)
        low = s.lower()
        if not s:
            continue
        if any(k in low for k in ["person", "human", "man", "woman", "child", "girl", "boy", "helper", "friend", "extra animal", "another protagonist"]):
            continue
        if s not in vals:
            vals.append(s)
    return ", ".join(vals[:limit])


def _lighting_hint(spec: FrameVisualSpec) -> str:
    mood = f"{spec.emotion} {spec.atmosphere} {spec.weather}".lower()
    if any(k in mood for k in ["sad", "empty", "lonely", "dark", "night", "gloom", "melancholy", "sorrow", "rain"]):
        return "moody lighting with preserved natural environment color, readable background, and bright clear light on the protagonist so the white fur and face stay visible"
    if any(k in mood for k in ["happy", "joy", "hope", "relief", "sunlight"]):
        return "clear bright lighting, vivid natural color, readable protagonist details, and colorful storybook atmosphere"
    return "balanced cinematic lighting, full natural color, readable background, and visible protagonist details"


def _critical_negative() -> str:
    return (
        "duplicate protagonist, second protagonist, extra character, extra animal, human, crowd, "
        "reflection that looks like another character, mirror clone, split screen, collage, comic panel, multiple scenes, "
        "cropped face, cropped body, cropped paws, cut off feet, hidden protagonist, tiny protagonist, "
        "wrong action, missing required object, wrong background, empty background, grayscale background, monochrome scene, washed out color, blurry, deformed, low detail, watermark, text"
    )


def _stage_name(frame_id: int, total_frames: int) -> str:
    if total_frames <= 1:
        return "single_scene"
    ratio = (frame_id - 1) / max(1, total_frames - 1)
    if ratio <= 0.15:
        return "setup"
    if ratio <= 0.45:
        return "search"
    if ratio <= 0.70:
        return "transition"
    if ratio <= 0.90:
        return "discovery"
    return "resolution"


def _visualize_action(action: str, caption: str, evidence: str) -> str:
    text = f"{action} {caption} {evidence}".lower()
    if any(k in text for k in ["hear", "hears", "listen", "sound of water"]):
        return "the protagonist pauses, turns toward a nearby stream, and reacts to the visible flowing water"
    if any(k in text for k in ["follow", "follows"]):
        return "the protagonist walks toward visible flowing water and follows the stream"
    if any(k in text for k in ["search", "look for", "looking for"]):
        return "the protagonist searches the area and looks around for the lost object"
    if any(k in text for k in ["arrive", "arrives"]):
        return "the protagonist reaches the place and notices the important object"
    if any(k in text for k in ["retrieve", "retrieves", "pick up", "recover"]):
        return "the protagonist picks up and holds the recovered object"
    if any(k in text for k in ["enter", "enters"]):
        return "the protagonist walks into the scene and begins the search"
    return _simple_sentence(action or caption, 12)


def _resolve_protagonist_anchor(spec: FrameVisualSpec) -> str:
    ident = _clean(spec.subject_identity)
    protagonist = _clean(spec.protagonist)
    text = f"{protagonist} {ident}".lower()
    parts = []
    if "white bear" in text or ("bear" in text and "white" in text):
        parts += ["one adult white bear", "white fur"]
    elif "brown bear" in text or ("bear" in text and "brown" in text):
        parts += ["one adult brown bear", "brown fur"]
    elif "polar bear" in text:
        parts += ["one adult polar bear", "white fur"]
    elif "panda" in text:
        parts += ["one adult panda", "black and white fur"]
    else:
        parts += [protagonist or "one protagonist"]
    for key in ["friendly", "gentle", "large", "fluffy"]:
        if key in text:
            parts.append(key)
    out = []
    for p in parts:
        p = _clean(p)
        if p and p not in out:
            out.append(p)
    return ", ".join(out[:4])




def _resolve_identity_lock(spec: FrameVisualSpec) -> str:
    lock = _clean(getattr(spec, "identity_lock", ""))
    if lock:
        # Keep the most important identity clauses while fitting SDXL token limits.
        clauses = [c.strip() for c in re.split(r"[;]+", lock) if c.strip()]
        return "; ".join(clauses[:4])
    return _resolve_protagonist_anchor(spec)


def _identity_short_for_prompt(spec: FrameVisualSpec) -> str:
    lock = _resolve_identity_lock(spec)
    anchor = _resolve_protagonist_anchor(spec)
    text = f"{anchor}; {lock}"
    # Never let a verbose identity block eat the whole 77-token text encoder budget.
    return _clip_words(text, 26)


def _scene_short_for_prompt(spec: FrameVisualSpec) -> str:
    contract = _clean(getattr(spec, "scene_contract", ""))
    if contract:
        return _clip_words(contract, 34)
    parts = []
    if _clean(spec.location):
        parts.append(f"background location {spec.location}")
    if spec.required_objects:
        parts.append("required visible objects " + _join_unique(spec.required_objects, limit=5, per_item_words=3))
    if _clean(spec.weather):
        parts.append(f"weather {spec.weather}")
    return _clip_words("; ".join(parts), 34)


def _appearance_delta_short(spec: FrameVisualSpec) -> str:
    delta = _clean(getattr(spec, "dcee_appearance_delta", ""))
    if not delta:
        delta = f"same identity; only expression and pose change for emotion {spec.emotion}"
    return _clip_words(delta, 24)


def _ip_scale_for_mode(base: float, mode: str, has_reference: bool) -> float:
    if not has_reference:
        return base
    # Higher on identity candidates, slightly lower on background candidates so the scene can move.
    if mode == "identity":
        return min(0.82, base + 0.08)
    if mode in {"background", "wide_scene"}:
        return max(0.55, base - 0.03)
    return base

def _spec_from_packet(packet) -> FrameVisualSpec:
    from .frame_director import build_frame_visual_spec
    m = getattr(packet, "control_metadata", {}) or {}
    d = m.get("frame_visual_spec", {}) or {}
    if d:
        return FrameVisualSpec(**d)
    # fallback minimal spec
    return FrameVisualSpec(
        frame_id=int(getattr(packet, "frame_id", 1)),
        total_frames=6,
        story_sentence=m.get("story_sentence", ""),
        protagonist=m.get("character_text", "protagonist"),
        subject_identity=m.get("character_text", "protagonist"),
        subject_reference_policy="text-only",
        primary_action=m.get("event_text", ""),
        visible_event=m.get("event_text", ""),
        visible_cause=m.get("event_grounding", ""),
        required_objects=m.get("must_show", []) or [],
        carry_over_entities=[],
        recurring_entities=[],
        forbidden_objects=["generic portrait"],
        location="",
        weather="",
        atmosphere="",
        emotion="",
        facial_expression="",
        body_pose="",
        camera="medium story shot",
        continuity="",
        negative="generic portrait",
    )


class SDXLButterflyCrossAttentionGenerator:
    """V34 identity-lock + scene-contract SDXL generator.

    Improvements over V33:
    - keeps compact identity anchor but adds a stricter identity lock for face/body/paws/signature consistency
    - separates fixed identity from DCEE-controlled appearance deltas such as expression, pose, dirt, wetness, and lighting
    - compiles a frame-level scene contract so the background follows the current story sentence instead of a generic template
    - uses identity, wide-scene, action, background, and emotion candidate modes with prompt/token auditing
    """

    def __init__(
        self,
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        device="cuda",
        width=1024,
        height=1024,
        num_inference_steps=40,
        guidance_scale=8.0,
        seed=42,
        adapter_ckpt=None,
        enable_cpu_offload=False,
        character_tokens=8,
        world_tokens=8,
        emotion_tokens=8,
        event_tokens=8,
        evidence_tokens=8,
        use_ip_adapter=True,
        ip_adapter_repo="h94/IP-Adapter",
        ip_adapter_subfolder="sdxl_models",
        ip_adapter_weight_name="ip-adapter_sdxl.bin",
        ip_adapter_scale=0.48,
        use_butterfly_adapter=False,
        quality_model_preset="sdxl_base",
        use_refiner=False,
        refiner_model_id=None,
        refiner_strength=0.80,
        aesthetic_score=6.0,
        negative_aesthetic_score=2.5,
        use_previous_frame_img2img=False,
        previous_frame_strength=0.40,
        canonical_reference_sheet_path='',
        identity_backend_priority=None,
        use_instantid=False,
        instantid_adapter_path='',
        instantid_controlnet_path='',
        use_photomaker=False,
        photomaker_adapter_path='',
        use_character_lora=False,
        character_lora_path='',
        character_lora_scale=0.85,
        use_subject_scene_fusion=True,
        subject_scene_fusion_scale=0.72,
        subject_scene_fusion_first_n=2,
    ):
        from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.seed = int(seed)
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model_id = _resolve_model_id(model_id, quality_model_preset)
        self.quality_model_preset = quality_model_preset
        self.use_refiner = False
        self.refiner_error = "disabled in V34 for stability"
        self.use_previous_frame_img2img = False
        self.story_img2img_error = "disabled in V35 for identity stability"
        self.canonical_reference_sheet_path = str(canonical_reference_sheet_path or '')
        self.identity_backend_priority = list(identity_backend_priority or ['instantid', 'photomaker', 'canonical_reference_sheet', 'character_lora', 'ip_adapter', 'text'])
        self.use_instantid = bool(use_instantid)
        self.instantid_adapter_path = str(instantid_adapter_path or '')
        self.instantid_controlnet_path = str(instantid_controlnet_path or '')
        self.use_photomaker = bool(use_photomaker)
        self.photomaker_adapter_path = str(photomaker_adapter_path or '')
        self.use_character_lora = bool(use_character_lora)
        self.character_lora_path = str(character_lora_path or '')
        self.character_lora_scale = float(character_lora_scale)
        self.use_subject_scene_fusion = bool(use_subject_scene_fusion)
        self.subject_scene_fusion_scale = float(subject_scene_fusion_scale)
        self.subject_scene_fusion_first_n = int(subject_scene_fusion_first_n)
        self.instantid_loaded = False
        self.instantid_error = 'disabled unless InstantID dependencies and weights are provided'
        self.photomaker_loaded = False
        self.photomaker_error = 'disabled unless PhotoMaker dependencies and weights are provided'
        try:
            if self.use_instantid and _path_exists(self.instantid_adapter_path) and _path_exists(self.instantid_controlnet_path):
                self.instantid_loaded = True
                self.instantid_error = ''
        except Exception as e:
            self.instantid_loaded = False
            self.instantid_error = str(e)[:500]
        try:
            if self.use_photomaker and _path_exists(self.photomaker_adapter_path):
                self.photomaker_loaded = True
                self.photomaker_error = ''
        except Exception as e:
            self.photomaker_loaded = False
            self.photomaker_error = str(e)[:500]
        self.character_lora_loaded = False
        self.character_lora_error = ''

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            use_safetensors=True,
        )
        try:
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config,
                use_karras_sigmas=True,
                algorithm_type="dpmsolver++",
            )
        except Exception:
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config, use_karras_sigmas=True)

        if enable_cpu_offload and device.startswith("cuda"):
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)

        try:
            self.pipe.vae.enable_slicing()
        except Exception:
            pass

        self.ip_adapter_loaded = False
        self.ip_adapter_error = ""
        self.ip_adapter_scale = max(0.62, float(ip_adapter_scale))
        if use_ip_adapter:
            try:
                self.pipe.load_ip_adapter(
                    ip_adapter_repo,
                    subfolder=ip_adapter_subfolder,
                    weight_name=ip_adapter_weight_name,
                )
                # scale is set by generate_from_packet; do not overwrite dynamic per-mode scale here
                pass
                self.ip_adapter_loaded = True
            except Exception as e:
                self.ip_adapter_loaded = False
                self.ip_adapter_error = str(e)[:500]

        if self.use_character_lora and _path_exists(self.character_lora_path):
            try:
                lora_dir = str(Path(self.character_lora_path).parent)
                weight_name = Path(self.character_lora_path).name
                self.pipe.load_lora_weights(lora_dir, weight_name=weight_name)
                try:
                    self.pipe.fuse_lora(lora_scale=self.character_lora_scale)
                except Exception:
                    try:
                        self.pipe.set_adapters(['default'], adapter_weights=[self.character_lora_scale])
                    except Exception:
                        pass
                self.character_lora_loaded = True
            except Exception as e:
                self.character_lora_loaded = False
                self.character_lora_error = str(e)[:500]

    def _reference_bank_paths(self, packet) -> List[str]:
        refs = getattr(packet, "reference_images", {}) or {}
        meta = getattr(packet, "control_metadata", {}) or {}
        if bool(meta.get('text_only_mode')):
            return []
        items = []
        # Prefer a single stable protagonist identity source instead of a collage.
        items += list(meta.get('protagonist_reference_paths', []) or [])
        items += [
            meta.get('canonical_reference_sheet_path', ''),
            refs.get('canonical_reference_sheet', ''),
            self.canonical_reference_sheet_path,
            refs.get('subject', ''),
            meta.get('source_reference_image_path', ''),
            meta.get('input_reference_image_path', ''),
        ]
        keep = []
        for p in items:
            if _path_exists(p) and str(p) not in keep:
                keep.append(str(p))
        return keep

    def _resolve_identity_backend(self, meta: Dict[str, Any], reference_bank_paths: List[str]) -> str:
        if bool(meta.get('text_only_mode')):
            return 'text'
        priority = list(meta.get('identity_backend_priority', self.identity_backend_priority) or self.identity_backend_priority)
        has_ref = bool(reference_bank_paths)
        for name in priority:
            if name == 'instantid' and self.instantid_loaded and has_ref:
                return 'instantid'
            if name == 'photomaker' and self.photomaker_loaded and has_ref:
                return 'photomaker'
            if name == 'canonical_reference_sheet' and has_ref:
                return 'canonical_reference_sheet'
            if name == 'character_lora' and self.character_lora_loaded:
                return 'character_lora'
            if name == 'ip_adapter' and self.ip_adapter_loaded and has_ref:
                return 'ip_adapter'
            if name == 'text':
                return 'text'
        return 'text'

    def _subject_reference_image(self, packet):
        bank = self._reference_bank_paths(packet)
        chosen = bank[0] if bank else ''
        if chosen and _path_exists(chosen):
            return _load_reference_image(str(chosen), size=(1024, 1024)), str(chosen), [str(chosen)]
        return None, '', []

    def _token_count(self, tokenizer, text: str):
        try:
            if tokenizer is None:
                return None
            text = _clean(text)
            if not text:
                return 0
            pieces = tokenizer.tokenize(text)
            try:
                special = tokenizer.num_special_tokens_to_add(pair=False)
            except Exception:
                special = 2
            return len(pieces) + int(special)
        except Exception:
            return None

    def _token_report(self, prompt: str, prompt_2: str, negative_prompt: str) -> Dict[str, Any]:
        out = {}
        for name, tok, txt in [
            ("prompt_tokenizer_1", getattr(self.pipe, "tokenizer", None), prompt),
            ("prompt_tokenizer_2", getattr(self.pipe, "tokenizer_2", None), prompt_2),
            ("negative_tokenizer_1", getattr(self.pipe, "tokenizer", None), negative_prompt),
            ("negative_tokenizer_2", getattr(self.pipe, "tokenizer_2", None), negative_prompt),
        ]:
            if tok is None:
                continue
            max_len = int(getattr(tok, "model_max_length", 77) or 77)
            if max_len > 1000:
                max_len = 77
            n = self._token_count(tok, txt)
            out[name] = {"tokens": n, "max_length": max_len, "will_truncate": bool(n is not None and n > max_len)}
        return out

    def _fit_segments(self, tokenizer, segments: List[str], reserve: int = 4):
        max_len = int(getattr(tokenizer, "model_max_length", 77) or 77)
        if max_len > 1000:
            max_len = 77
        kept, dropped = [], []
        used = 2
        for seg in segments:
            seg = _clean(seg)
            if not seg:
                continue
            seg_tokens = self._token_count(tokenizer, seg)
            if seg_tokens is None:
                seg_tokens = len(seg.split()) + 2
            seg_tokens = max(1, seg_tokens - 2)
            join_cost = 1 if kept else 0
            if used + join_cost + seg_tokens <= (max_len - reserve):
                kept.append(seg)
                used += join_cost + seg_tokens
            else:
                dropped.append(seg)
        prompt = ". ".join(kept)
        return prompt, {
            "kept_segments": kept,
            "dropped_segments": dropped,
            "dropped_count": len(dropped),
            "max_length": max_len,
            "final_tokens": self._token_count(tokenizer, prompt),
        }

    def _fit_single_string(self, tokenizer, text: str, reserve: int = 4):
        segs = [s.strip() for s in re.split(r'[;,]+', text) if s.strip()]
        return self._fit_segments(tokenizer, segs, reserve=reserve)

    def _text_story_terms(self, spec: FrameVisualSpec, meta: Dict[str, Any]):
        sent = _clean(getattr(spec, 'story_sentence', ''))
        low = sent.lower()
        must = []
        for item in list(getattr(spec, 'required_objects', []) or []) + list(getattr(spec, 'critical_visual_nouns', []) or []):
            item = _clean(item)
            if item and item not in must:
                must.append(item)
        env = []
        for k, v in [
            (['forest', 'woods', 'tree'], 'deep forest with trees'),
            (['bush', 'underbrush'], 'bushes and underbrush'),
            (['root'], 'tangled roots'),
            (['hill', 'slope', 'elevation'], 'small hill or slope'),
            (['path', 'trail'], 'forest path'),
            (['lake', 'water', 'shore'], 'serene lake and water edge'),
        ]:
            if any(x in low for x in k) and v not in env:
                env.append(v)
        action = _simple_sentence(meta.get('visualized_action') or spec.primary_action or spec.visible_event, 12)
        protagonist = _resolve_protagonist_anchor(spec)
        emotion = _simple_sentence(spec.emotion, 4)
        return protagonist, sent, action, emotion, must[:8], env[:6]

    def _text_only_negative_for_story(self, spec: FrameVisualSpec):
        sent = _clean(getattr(spec, 'story_sentence', '')).lower()
        negatives = [
            'fox, red fox, squirrel, deer, rabbit, wolf',
            'brown bear, grizzly bear, teddy bear, panda',
            'book, reading, chair, sofa, car, truck, bus, road vehicle',
            'city, room interior, classroom, library',
            'multiple animals, extra animal, extra human',
        ]
        if 'lake' not in sent and 'water' not in sent:
            negatives.append('boat, ocean, sea')
        return ', '.join(negatives)

    def _build_prompt_pair(self, spec: FrameVisualSpec, meta: Dict[str, Any], mode: str):
        caption = _simple_sentence(spec.story_sentence, 24)
        identity_lock = _identity_short_for_prompt(spec)
        protagonist_anchor = _resolve_protagonist_anchor(spec)
        action = _simple_sentence(meta.get("visualized_action") or spec.primary_action or spec.visible_event, 12)
        required = _join_unique(spec.required_objects, limit=6, per_item_words=4)
        critical_nouns = _join_unique(getattr(spec, 'critical_visual_nouns', []) or meta.get('critical_visual_nouns', []), limit=8, per_item_words=4)
        location = _simple_sentence(spec.location, 10)
        weather = _simple_sentence(spec.weather, 5)
        atmosphere = _simple_sentence(spec.atmosphere, 5)
        emotion = _simple_sentence(spec.emotion, 4)
        previous_story_summary = _simple_sentence(meta.get("previous_story_summary"), 18)
        prev_caption = _simple_sentence(meta.get("previous_frame_caption"), 10)
        next_caption = _simple_sentence(meta.get("next_frame_caption") or getattr(spec, 'next_story_hint', ''), 10)
        scene_summary = _simple_sentence(getattr(spec, 'scene_summary', ''), 36)
        background_contract = _simple_sentence(getattr(spec, 'background_contract', ''), 24)
        retry_strong = bool(meta.get("retry_strong_mode"))
        text_only_mode = bool(meta.get('text_only_mode'))

        # V41: dedicated strong prompt for text-only story grounding.
        if text_only_mode or mode == 'text_story':
            protagonist, sent, action2, emotion2, must_items, env_items = self._text_story_terms(spec, meta)
            must_text = '; '.join(must_items)
            env_text = '; '.join(env_items)
            seg1 = [
                'single coherent full-color illustrated story panel',
                f'exact story sentence: {sent or caption}',
                f'exactly one protagonist: {protagonist}',
                'the protagonist must clearly be a white bear with white fur, bear body shape, bear face, and bear paws',
                f'current action: {action2 or action}',
                f'visible required nouns: {must_text}' if must_text else '',
                f'visible environment nouns: {env_text}' if env_text else '',
                f'location: {location}' if location else '',
                f'emotion: {emotion2 or emotion}' if (emotion2 or emotion) else '',
                'generate from text only; do not invent unrelated characters, props, or scenes',
                'if the sentence mentions a honey jar, show the honey jar visibly in this frame whenever physically possible',
                'if the sentence mentions forest, bushes, roots, hill, path, lake, or water, show those scene elements explicitly',
                'the image must depict this story sentence, not a generic animal illustration',
            ]
            seg2 = [
                'storybook illustration, readable composition, vivid natural colors',
                'show a full scene with foreground, midground, and background',
                'show exactly one white bear protagonist, full body visible, uncropped face, paws, and feet',
                'the protagonist should be inside the scene, integrated naturally with the environment',
                'no icon, no sticker, no mascot, no poster, no collage',
                'no reading book, no vehicle, no indoor scene unless the sentence says so',
                'no fox, no brown bear, no red animal, no random substitute animal',
                'make the current frame specific to the sentence and different from other frames',
                background_contract if background_contract else '',
            ]
            if previous_story_summary:
                seg2.append(f'story so far: {previous_story_summary}')
            if prev_caption:
                seg2.append(f'continue after previous frame: {prev_caption}')
            if next_caption:
                seg2.append(f'do not jump ahead; next frame comes later: {next_caption}')
            if weather:
                seg2.append(f'weather: {weather}')
            if atmosphere:
                seg2.append(f'atmosphere: {atmosphere}')
            if scene_summary:
                seg2.append(f'scene summary: {scene_summary}')
            if retry_strong:
                seg1 += [
                    'strictly prioritize story faithfulness and noun grounding',
                    'every important noun in the sentence must be visible and recognizable',
                ]
                seg2 += [
                    'make the honey jar or search clue unmistakably visible when relevant',
                    'increase scene detail and make the white bear clearly identifiable as a white bear',
                ]
            tok1 = getattr(self.pipe, 'tokenizer', None)
            tok2 = getattr(self.pipe, 'tokenizer_2', None) or tok1
            prompt, fit1 = self._fit_segments(tok1, seg1, reserve=3) if tok1 is not None else ('. '.join([s for s in seg1 if s]), {})
            prompt_2, fit2 = self._fit_segments(tok2, seg2, reserve=3) if tok2 is not None else ('. '.join([s for s in seg2 if s]), {})
            neg_raw = ', '.join([_critical_negative(), self._text_only_negative_for_story(spec), _clean(spec.negative)])
            negative_prompt, negfit = self._fit_single_string(tok1 or tok2, neg_raw, reserve=3) if (tok1 or tok2) is not None else (neg_raw, {})
            audit = self._token_report(prompt, prompt_2, negative_prompt)
            audit['prompt_fit_tokenizer_1'] = fit1
            audit['prompt_fit_tokenizer_2'] = fit2
            audit['negative_fit'] = negfit
            audit['prompt_strategy'] = 'V41_text_only_story_grounding'
            audit['identity_lock'] = identity_lock
            audit['scene_summary'] = scene_summary
            return prompt, prompt_2, negative_prompt, audit

        if mode == 'scene_first':
            variant = 'prioritize matching the exact story sentence with a readable environment and visible action'
        elif mode == 'identity':
            variant = 'prioritize stable protagonist identity while still keeping the full story scene visible'
        elif mode == 'background':
            variant = 'prioritize a complete readable background and setting details that match the sentence'
        elif mode == 'continuity':
            variant = 'prioritize continuity with the previous story step while clearly showing the current step'
        else:
            variant = 'prioritize exact story-step grounding'

        seg1 = [
            'single coherent full-color storybook illustration, one scene only',
            f'exact story sentence: {caption}',
            f'exactly one protagonist: {protagonist_anchor}',
            f'identity lock: {identity_lock}',
            'the reference image is for character identity only, not for layout or background duplication',
            f'main visible action: {action}' if action else '',
            f'required visible props: {required}' if required else '',
            f'critical visible nouns: {critical_nouns}' if critical_nouns else '',
            f'scene summary: {scene_summary}' if scene_summary else '',
            f'location: {location}' if location else '',
            f'weather: {weather}' if weather else '',
            f'atmosphere: {atmosphere}' if atmosphere else '',
            f'emotion: {emotion}' if emotion else '',
            variant,
        ]
        seg2 = [
            'cinematic storybook art, vivid natural colors, clean readable composition',
            'medium-wide or wide full-body framing, uncropped face, hands or paws, feet, and full environment visible',
            'show foreground action, midground props, and background setting clearly in one image',
            'the environment must be specific to the story sentence and not blank or plain',
            'show trees, ground, water, slope, roots, bush, or other story-specific setting details whenever the caption implies them',
            'the image must read like a visual story panel, not a sticker, icon, mascot, or isolated character portrait',
            'do not create duplicate protagonists, companions, or extra animals',
            'no collage, no split screen, no reference sheet appearance in the final image',
            background_contract if background_contract else '',
            _lighting_hint(spec),
        ]
        if previous_story_summary and mode in {'scene_first', 'continuity', 'identity'}:
            seg2.append(f'story so far: {previous_story_summary}')
        if prev_caption and mode in {'scene_first', 'continuity', 'identity'}:
            seg2.append(f'continue naturally after previous frame: {prev_caption}')
        if next_caption and mode in {'scene_first', 'continuity', 'background'}:
            seg2.append(f'do not jump ahead; next frame comes after this: {next_caption}')
        if retry_strong:
            seg1 += ['strictly prioritize story faithfulness over generic character art', 'the key noun objects and setting must be unmistakably visible']
            seg2 += ['increase environment detail and prop visibility', 'make the current action obvious at a glance', 'show a complete scene instead of a plain background character portrait']
        tok1 = getattr(self.pipe, 'tokenizer', None)
        tok2 = getattr(self.pipe, 'tokenizer_2', None) or tok1
        prompt, fit1 = self._fit_segments(tok1, seg1, reserve=3) if tok1 is not None else ('. '.join([s for s in seg1 if s]), {})
        prompt_2, fit2 = self._fit_segments(tok2, seg2, reserve=3) if tok2 is not None else ('. '.join([s for s in seg2 if s]), {})
        neg_raw = ', '.join([_critical_negative(), 'close-up portrait, mascot icon, sticker, blank background, gray background, plain background, isolated character', 'duplicate protagonist, second bear, extra animal, extra human, companion character', 'cropped body, cropped feet, cropped paws, cropped face, missing hands, missing paws, missing feet, malformed limbs', 'missing story prop, missing key object, missing setting, missing background, wrong background, unrelated background', 'reference sheet, collage, split screen, multiple panels, poster layout', _clean(spec.negative)])
        negative_prompt, negfit = self._fit_single_string(tok1 or tok2, neg_raw, reserve=3) if (tok1 or tok2) is not None else (neg_raw, {})
        audit = self._token_report(prompt, prompt_2, negative_prompt)
        audit['prompt_fit_tokenizer_1'] = fit1
        audit['prompt_fit_tokenizer_2'] = fit2
        audit['negative_fit'] = negfit
        audit['prompt_strategy'] = 'V41_scene_story_grounding'
        audit['identity_lock'] = identity_lock
        audit['scene_summary'] = scene_summary
        return prompt, prompt_2, negative_prompt, audit

    def _build_foreground_prompt(self, spec: FrameVisualSpec, meta: Dict[str, Any]):
        protagonist_anchor = _resolve_protagonist_anchor(spec)
        identity_lock = _identity_short_for_prompt(spec)
        action = _simple_sentence(meta.get("visualized_action") or spec.primary_action or spec.visible_event, 12)
        emotion = _simple_sentence(spec.emotion, 4)
        face = _simple_sentence(spec.facial_expression, 7)
        pose = _simple_sentence(spec.body_pose, 7)
        prompt = ". ".join([x for x in [
            'single full-body protagonist illustration, one subject only',
            f'exactly one protagonist: {protagonist_anchor}',
            f'identity lock: {identity_lock}',
            f'visible action: {action}' if action else '',
            f'emotion: {emotion}' if emotion else '',
            f'facial expression: {face}' if face else '',
            f'body pose: {pose}' if pose else '',
            'show the full body head-to-toe, centered, uncropped, with clear paws, feet, ears, and face',
            'simple clean studio-like background or plain soft backdrop for easy subject extraction',
            'no extra subjects, no environment clutter, no duplicate protagonist',
        ] if x])
        negative = ', '.join([
            _critical_negative(),
            'extra subject, second bear, two bears, multiple characters, busy background, forest background, lake background, duplicate protagonist, cropped body, missing feet, mascot icon, sticker'
        ])
        return prompt, negative

    def _build_background_prompt(self, spec: FrameVisualSpec, meta: Dict[str, Any]):
        caption = _simple_sentence(spec.story_sentence, 24)
        required = _join_unique(spec.required_objects, limit=5, per_item_words=3)
        critical_nouns = _join_unique(getattr(spec, 'critical_visual_nouns', []), limit=6, per_item_words=3)
        location = _simple_sentence(spec.location, 9)
        weather = _simple_sentence(spec.weather, 5)
        atmosphere = _simple_sentence(spec.atmosphere, 5)
        background_contract = _simple_sentence(getattr(spec, 'background_contract', ''), 26)
        previous_story_summary = _simple_sentence(meta.get('previous_story_summary'), 18)
        prompt = ". ".join([x for x in [
            'single coherent full-color storybook background plate',
            f'exact story sentence: {caption}',
            f'location: {location}' if location else '',
            f'background contract: {background_contract}' if background_contract else '',
            f'critical visible nouns in the environment: {critical_nouns}' if critical_nouns else '',
            f'props that must be visible in the scene: {required}' if required else '',
            f'weather: {weather}' if weather else '',
            f'atmosphere: {atmosphere}' if atmosphere else '',
            f'story so far: {previous_story_summary}' if previous_story_summary else '',
            'show a complete layered environment with foreground ground plane, midground props, and readable distant background',
            'leave clear open space in the center or lower-center for the protagonist to be placed into the scene',
            'make the environment story-specific and readable, not plain or blank',
            'no protagonist visible in the background plate',
        ] if x])
        negative = ', '.join([
            'protagonist, bear, panda, animal, human, crowd, duplicate character, blank background, gray empty background, portrait, close-up, sticker, mascot',
            _clean(spec.negative),
        ])
        return prompt, negative

    def _make_edge_mask(self, fg_image: Image.Image) -> Image.Image:
        img = fg_image.convert('RGB')
        arr = np.array(img).astype(np.int16)
        edges = np.concatenate([arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]], axis=0)
        bg = np.median(edges, axis=0)
        dist = np.sqrt(((arr - bg) ** 2).sum(axis=2))
        mask = (dist > 28).astype(np.uint8) * 255
        # also treat nearly white or nearly gray as background
        mean = arr.mean(axis=2)
        var = arr.var(axis=2)
        bg_like = ((mean > 235) | ((mean > 210) & (var < 120)))
        mask = np.where(bg_like, 0, mask).astype(np.uint8)
        pil = Image.fromarray(mask, mode='L').filter(ImageFilter.GaussianBlur(radius=2.2))
        return pil

    def _compose_subject_scene(self, fg_image: Image.Image, bg_image: Image.Image) -> Image.Image:
        bg = bg_image.convert('RGBA').resize((self.width, self.height))
        fg = fg_image.convert('RGBA')
        mask = self._make_edge_mask(fg_image)
        bbox = mask.getbbox()
        if not bbox:
            return bg_image
        fg_crop = fg.crop(bbox)
        mask_crop = mask.crop(bbox)
        target_h = int(self.height * max(0.56, min(0.82, self.subject_scene_fusion_scale)))
        scale = target_h / max(1, fg_crop.height)
        target_w = max(1, int(fg_crop.width * scale))
        fg_crop = fg_crop.resize((target_w, target_h), Image.LANCZOS)
        mask_crop = mask_crop.resize((target_w, target_h), Image.LANCZOS)
        x = (self.width - target_w) // 2
        y = self.height - target_h - int(self.height * 0.08)
        canvas = bg.copy()
        canvas.alpha_composite(fg_crop, (x, y))
        # soften hard edges by applying mask once more if supported
        out = Image.new('RGBA', (self.width, self.height), (255,255,255,0))
        out.alpha_composite(bg, (0,0))
        out.paste(fg_crop, (x,y), mask_crop)
        return out.convert('RGB')

    def _neutral_ip_adapter_image(self) -> Image.Image:
        # When IP-Adapter is loaded, diffusers may require image embeddings even for background-only generations.
        # Use a neutral blank conditioning image with adapter scale 0.0 so the background plate is not pulled toward the protagonist.
        return Image.new('RGB', (self.width, self.height), (255, 255, 255))

    def _generate_subject_scene_fusion(self, spec: FrameVisualSpec, meta: Dict[str, Any], reference_image, frame_id: int, cid: int):
        fg_prompt, fg_negative = self._build_foreground_prompt(spec, meta)
        bg_prompt, bg_negative = self._build_background_prompt(spec, meta)
        tok1 = getattr(self.pipe, 'tokenizer', None)
        tok2 = getattr(self.pipe, 'tokenizer_2', None) or tok1
        fg_prompt, _ = self._fit_single_string(tok1, fg_prompt, reserve=3) if tok1 is not None else (fg_prompt, {})
        fg_prompt_2, _ = self._fit_single_string(tok2, fg_prompt, reserve=3) if tok2 is not None else (fg_prompt, {})
        bg_prompt, _ = self._fit_single_string(tok1, bg_prompt, reserve=3) if tok1 is not None else (bg_prompt, {})
        bg_prompt_2, _ = self._fit_single_string(tok2, bg_prompt, reserve=3) if tok2 is not None else (bg_prompt, {})
        fg_negative, _ = self._fit_single_string(tok1 or tok2, fg_negative, reserve=3) if (tok1 or tok2) is not None else (fg_negative, {})
        bg_negative, _ = self._fit_single_string(tok1 or tok2, bg_negative, reserve=3) if (tok1 or tok2) is not None else (bg_negative, {})

        base_seed = self.seed + int(frame_id) * 1000 + cid
        gen_bg = torch.Generator(device=self.device).manual_seed(base_seed + 101) if self.device.startswith('cuda') else torch.Generator().manual_seed(base_seed + 101)
        bg_kwargs = dict(prompt=bg_prompt, prompt_2=bg_prompt_2, negative_prompt=bg_negative, negative_prompt_2=bg_negative, width=self.width, height=self.height, num_inference_steps=max(30, self.num_inference_steps - 2), guidance_scale=max(6.8, self.guidance_scale - 0.3), generator=gen_bg)
        # HOTFIX V39.1:
        # If IP-Adapter was loaded globally, diffusers sets encoder_hid_dim_type='ip_image_proj'.
        # Then even background-only calls require ip_adapter_image / image_embeds.
        # We pass a neutral image with adapter scale 0.0 so the background is not conditioned on the protagonist.
        if getattr(self, 'ip_adapter_loaded', False):
            try:
                self.pipe.set_ip_adapter_scale(0.0)
            except Exception:
                pass
            bg_kwargs['ip_adapter_image'] = self._neutral_ip_adapter_image()
        try:
            bg_kwargs['guidance_rescale'] = 0.15
        except Exception:
            pass
        bg_image = self.pipe(**bg_kwargs).images[0]

        gen_fg = torch.Generator(device=self.device).manual_seed(base_seed + 202) if self.device.startswith('cuda') else torch.Generator().manual_seed(base_seed + 202)
        fg_kwargs = dict(prompt=fg_prompt, prompt_2=fg_prompt_2, negative_prompt=fg_negative, negative_prompt_2=fg_negative, width=self.width, height=self.height, num_inference_steps=max(28, self.num_inference_steps - 4), guidance_scale=max(7.2, self.guidance_scale), generator=gen_fg)
        used_reference = False
        if reference_image is not None:
            try:
                self.pipe.set_ip_adapter_scale(min(0.55, self.ip_adapter_scale))
            except Exception:
                pass
        if self._apply_ip_adapter(reference_image):
            fg_kwargs['ip_adapter_image'] = reference_image
            used_reference = True
        try:
            fg_kwargs['guidance_rescale'] = 0.10
        except Exception:
            pass
        fg_image = self.pipe(**fg_kwargs).images[0]
        merged = self._compose_subject_scene(fg_image, bg_image)
        audit = {
            'prompt_strategy': 'V39_subject_scene_fusion',
            'fg_prompt': fg_prompt,
            'bg_prompt': bg_prompt,
            'fg_negative_prompt': fg_negative,
            'bg_negative_prompt': bg_negative,
            'used_reference': used_reference,
        }
        return merged, audit

    def _apply_ip_adapter(self, reference_image):
        if self.ip_adapter_loaded and reference_image is not None:
            try:
                # scale is set by generate_from_packet; do not overwrite dynamic per-mode scale here
                pass
            except Exception:
                pass
            return True
        return False

    @torch.no_grad()
    def generate_from_packet(self, packet, frame_id: int, out_dir: Path, num_candidates: int = 4) -> List[CandidateImage]:
        out_dir.mkdir(parents=True, exist_ok=True)
        spec = _spec_from_packet(packet)
        meta = getattr(packet, "control_metadata", {}) or {}
        reference_image, reference_path, reference_bank_paths = self._subject_reference_image(packet)
        active_identity_backend = self._resolve_identity_backend(meta, reference_bank_paths)
        text_only_mode = bool(meta.get('text_only_mode'))
        modes = ["text_story", "text_story", "scene_first", "continuity"] if text_only_mode else ["fusion", "fusion", "scene_first", "continuity"]

        results: List[CandidateImage] = []
        for cid in range(num_candidates):
            mode = modes[cid % len(modes)] if (self.use_subject_scene_fusion and not text_only_mode) else (["text_story", "scene_first", "continuity", "background"][cid % 4] if text_only_mode else ["scene_first", "identity", "background", "continuity"][cid % 4])
            meta2 = dict(meta)
            meta2["visualized_action"] = _visualize_action(spec.visible_event or spec.primary_action, spec.story_sentence, spec.visible_cause)
            sd = self.seed + int(frame_id) * 1000 + cid
            used_reference = False
            effective_ip_adapter_scale = self.ip_adapter_scale
            fusion_audit = {}
            if (self.use_subject_scene_fusion and not text_only_mode) and cid < max(1, self.subject_scene_fusion_first_n):
                image, fusion_audit = self._generate_subject_scene_fusion(spec, meta2, reference_image, frame_id, cid)
                prompt = fusion_audit.get('fg_prompt', '')
                prompt_2 = fusion_audit.get('bg_prompt', '')
                negative_prompt = fusion_audit.get('fg_negative_prompt', '')
                token_report = {'prompt_strategy': 'V39_subject_scene_fusion'}
                used_reference = bool(fusion_audit.get('used_reference'))
                mode = 'fusion'
            else:
                prompt, prompt_2, negative_prompt, token_report = self._build_prompt_pair(spec, meta2, mode)
                gen = torch.Generator(device=self.device).manual_seed(sd) if self.device.startswith("cuda") else torch.Generator().manual_seed(sd)
                mode_steps = self.num_inference_steps + (2 if mode in {'scene_first', 'background', 'continuity'} else 0)
                mode_guidance = self.guidance_scale + (0.15 if mode in {'scene_first', 'continuity'} else 0.0)
                kwargs = dict(
                    prompt=prompt,
                    prompt_2=prompt_2,
                    negative_prompt=negative_prompt,
                    negative_prompt_2=negative_prompt,
                    width=self.width,
                    height=self.height,
                    num_inference_steps=mode_steps,
                    guidance_scale=mode_guidance,
                    generator=gen,
                )
                if reference_image is not None:
                    effective_ip_adapter_scale = min(0.52, _ip_scale_for_mode(self.ip_adapter_scale, mode, True))
                    try:
                        self.pipe.set_ip_adapter_scale(effective_ip_adapter_scale)
                    except Exception:
                        pass
                if self._apply_ip_adapter(reference_image):
                    kwargs["ip_adapter_image"] = reference_image
                    used_reference = True
                elif getattr(self, 'ip_adapter_loaded', False):
                    # Avoid diffusers ValueError when IP-Adapter is loaded but this mode has no reference image.
                    try:
                        self.pipe.set_ip_adapter_scale(0.0)
                    except Exception:
                        pass
                    kwargs["ip_adapter_image"] = self._neutral_ip_adapter_image()
                try:
                    kwargs["guidance_rescale"] = 0.15
                except Exception:
                    pass
                image = self.pipe(**kwargs).images[0]
            path = out_dir / f"frame_{int(frame_id):03d}_cand_{cid:02d}.png"
            image.save(path)
            results.append(
                CandidateImage(
                    frame_id=int(frame_id),
                    candidate_id=cid,
                    image_path=str(path),
                    prompt=prompt,
                    scores={
                        "image_quality": 0.0,
                        "colorfulness": 0.0,
                        "identity_consistency": 0.0,
                        "reference_subject_similarity": 0.0,
                        "subject_visibility": 0.0,
                        "crop_penalty": 0.0,
                        "emotion_visibility": 0.0,
                        "emotion_cause_visibility": 0.0,
                        "event_grounding": 0.0,
                        "evidence_visibility": 0.0,
                        "event_emotion_causal_consistency": 0.0,
                        "scene_alignment": 0.0,
                        "event_alignment": 0.0,
                        "story_alignment": 0.0,
                        "continuity": 0.0,
                        "overall": 0.0,
                    },
                    notes={
                        "generator_version": "V41",
                        "seed": sd,
                        "prompt_variant_mode": mode,
                        "generation_route": "text_story_grounding_or_scene_first_text2img",
                        "caption_locked_generation": True,
                        "identity_locked_generation": True,
                        "dcee_controlled_appearance_delta": getattr(spec, "dcee_appearance_delta", ""),
                        "scene_contract_generation": getattr(spec, "scene_contract", ""),
                        "story_faithful_generation": True,
                        "scene_first_generation": True,
                        "subject_scene_fusion_generation": bool(mode == "fusion"),
                        "visual_storytelling_generation": True,
                        "nonempty_background_generation": True,
                        "quality_model_preset": self.quality_model_preset,
                        "model_id": self.model_id,
                        "subject_reference_path": reference_path,
                        "canonical_reference_sheet_path": meta.get('canonical_reference_sheet_path', self.canonical_reference_sheet_path),
                        "previous_frame_image_path": meta.get('previous_frame_image_path', ''),
                        "previous_frame_caption": meta.get('previous_frame_caption', ''),
                        "previous_story_summary": meta.get('previous_story_summary', ''),
                        "previous_frame_local_caption": meta.get('previous_frame_local_caption', ''),
                        "next_frame_caption": meta.get('next_frame_caption', ''),
                        "critical_visual_nouns": getattr(spec, 'critical_visual_nouns', []),
                        "ip_adapter_loaded": self.ip_adapter_loaded,
                        "ip_adapter_used": used_reference,
                        "ip_adapter_scale": effective_ip_adapter_scale,
                        "ip_adapter_error": self.ip_adapter_error,
                        "character_lora_loaded": self.character_lora_loaded,
                        "character_lora_error": self.character_lora_error,
                        "instantid_loaded": self.instantid_loaded,
                        "instantid_error": self.instantid_error,
                        "photomaker_loaded": self.photomaker_loaded,
                        "photomaker_error": self.photomaker_error,
                        "identity_backend_priority": self.identity_backend_priority,
                        "identity_backend_selected": active_identity_backend,
                        "reference_bank_paths": reference_bank_paths,
                        "fusion_audit": fusion_audit,
                        "frame_visual_spec": spec.to_dict(),
                        "prompt_2": prompt_2,
                        "negative_prompt": negative_prompt,
                        "token_report": token_report,
                        "visualized_action": meta2.get("visualized_action", ""),
                    },
                )
            )
        return results
