from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List
import math, numpy as np
from PIL import Image, ImageFilter, ImageStat
from .llm import BaseLLM, BaseVLM
from .prompts import SYSTEM_NARRATIVE, SYSTEM_VLM, eval_questions_prompt
from .schema import DCEPlan, EmotionArc, StoryboardFrame, CandidateImage
from .utils import extract_json, make_contact_sheet

def image_quality_proxy(path):
    try:
        img=Image.open(path).convert('RGB'); gray=img.convert('L').resize((256,256)); edges=gray.filter(ImageFilter.FIND_EDGES)
        sharp=min(1.0, ImageStat.Stat(edges).mean[0]/45.0); stat=ImageStat.Stat(gray); contrast=min(1.0, stat.stddev[0]/60.0); bright=stat.mean[0]/255.0; bscore=1.0-min(1.0,abs(bright-0.55)/0.55)
        return round(float(0.45*sharp+0.35*contrast+0.20*bscore),4)
    except Exception: return 0.5

def colorfulness_score(path):
    try:
        img=np.array(Image.open(path).convert('RGB')).astype(np.float32); r,g,b=img[:,:,0],img[:,:,1],img[:,:,2]
        rg=np.abs(r-g); yb=np.abs(0.5*(r+g)-b); cf=math.sqrt(np.std(rg)**2+np.std(yb)**2)+0.3*math.sqrt(np.mean(rg)**2+np.mean(yb)**2)
        return round(float(min(1.0, cf/60.0)),4)
    except Exception: return 0.3


def _robust_make_contact_sheet(image_paths, out_path, cols=3, thumb_size=(384,384)):
    from pathlib import Path
    from PIL import Image, ImageDraw

    valid = []
    for p in image_paths:
        if not p:
            continue
        pp = Path(p)
        if pp.exists():
            valid.append(pp)

    if not valid:
        raise ValueError("No existing selected image paths for contact sheet.")

    rows = (len(valid) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * thumb_size[0], rows * thumb_size[1]), "white")
    draw = ImageDraw.Draw(canvas)

    for idx, path in enumerate(valid):
        img = Image.open(path).convert("RGB")
        img.thumbnail(thumb_size)
        x = (idx % cols) * thumb_size[0]
        y = (idx // cols) * thumb_size[1]
        canvas.paste(img, (x, y))
        draw.text((x + 10, y + 10), f"Frame {idx + 1}", fill=(0, 0, 0))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return str(out_path)


def _compact_storyboard(storyboard):
    return [{'frame_id':f.frame_id,'event':f.event,'event_grounding':getattr(f,'event_grounding',''),'evidence_objects':getattr(f,'evidence_objects',[]),'emotion':f.emotion,'must_show':getattr(f,'must_show',[])} for f in storyboard]

class DCEQAEvaluator:
    def __init__(self, llm:BaseLLM, vlm:BaseVLM, use_vlm:bool=True, save_contact_sheet:bool=True): self.llm=llm; self.vlm=vlm; self.use_vlm=use_vlm; self.save_contact_sheet=save_contact_sheet
    def generate_questions(self, dce_plan:DCEPlan, emotion_arc:EmotionArc, storyboard:List[StoryboardFrame]) -> Dict[str,Any]:
        try: return extract_json(self.llm.generate(SYSTEM_NARRATIVE, eval_questions_prompt(asdict(dce_plan), asdict(emotion_arc), _compact_storyboard(storyboard)), temperature=0.0, max_tokens=1200))
        except Exception: return {'global_questions':['event grounding','evidence visibility','event-emotion causal consistency','ending emotion accuracy','character consistency'], 'frame_questions':{}, 'ending_questions':[]}
    def _vlm_frame_eval(self, frame, cand):
        if not self.use_vlm: return {}
        prompt=f"""
Evaluate image for DCEE visual storytelling.
Event: {frame.event}
Event grounding: {getattr(frame,'event_grounding','')}
Evidence objects: {getattr(frame,'evidence_objects',[])}
Must show: {getattr(frame,'must_show',[])}
Emotion: {frame.emotion} intensity {frame.emotion_intensity}/5
World: {getattr(frame,'scene_location','')}, {getattr(frame,'weather','')}, {getattr(frame,'atmosphere','')}
Return JSON only with scores 0-1: event_grounding, evidence_visibility, emotion_visibility, emotion_cause_visibility, event_emotion_causal_consistency, scene_alignment, event_alignment, identity_consistency, colorfulness, reason.
""".strip()
        try: return extract_json(self.vlm.generate_with_images(SYSTEM_VLM, prompt, [cand.image_path], temperature=0.0, max_tokens=500))
        except Exception as e: return {'vlm_error':str(e)[:300]}
    def rank_frame_candidates(self, frame, dce_plan, candidates, is_ending=False):
        out=[]
        for c in candidates:
            scores={'image_quality':image_quality_proxy(c.image_path),'colorfulness':colorfulness_score(c.image_path),'identity_consistency':0.74,'emotion_visibility':0.70,'emotion_cause_visibility':0.68,'event_grounding':0.68,'evidence_visibility':0.66,'event_emotion_causal_consistency':0.68,'scene_alignment':0.70,'event_alignment':0.70}
            v=self._vlm_frame_eval(frame,c)
            if v and 'vlm_error' not in v:
                for k in list(scores.keys()):
                    if k in v:
                        try: scores[k]=float(v[k])
                        except Exception: pass
                c.notes['vlm_reason']=v.get('reason','')
            elif v and 'vlm_error' in v: c.notes['vlm_error']=v['vlm_error']
            if is_ending:
                overall=0.16*scores['identity_consistency']+0.12*scores['image_quality']+0.18*scores['emotion_visibility']+0.14*scores['emotion_cause_visibility']+0.16*scores['event_grounding']+0.12*scores['evidence_visibility']+0.08*scores['event_emotion_causal_consistency']+0.04*scores['colorfulness']
            else:
                overall=0.17*scores['identity_consistency']+0.12*scores['image_quality']+0.16*scores['emotion_visibility']+0.13*scores['emotion_cause_visibility']+0.17*scores['event_grounding']+0.12*scores['evidence_visibility']+0.08*scores['scene_alignment']+0.05*scores['colorfulness']
            scores['overall']=round(float(overall),4); c.scores.update(scores); out.append(c)
        return sorted(out, key=lambda x:x.scores.get('overall',0.0), reverse=True)
    def rerank_ending_candidates(self, final_frame, dce_plan, candidates): return self.rank_frame_candidates(final_frame,dce_plan,candidates,is_ending=True)
    def evaluate_sequence(self, dce_plan, emotion_arc, storyboard, images, questions, out_dir=None):
        if not images:
            return {'warning':'No images'}

        cs = None
        contact_sheet_error = None

        if self.save_contact_sheet and out_dir:
            out_path = Path(out_dir) / 'contact_sheet.png'
            image_paths = [x.image_path for x in images]
            try:
                cs = make_contact_sheet(image_paths, out_path)
            except Exception as e:
                contact_sheet_error = f"utils.make_contact_sheet failed: {type(e).__name__}: {e}"
                try:
                    cs = _robust_make_contact_sheet(image_paths, out_path)
                    contact_sheet_error = None
                except Exception as e2:
                    contact_sheet_error += f" | fallback failed: {type(e2).__name__}: {e2}"

        n = max(1, len(images))
        keys = ['image_quality','colorfulness','identity_consistency','emotion_visibility','emotion_cause_visibility','event_grounding','evidence_visibility','event_emotion_causal_consistency','scene_alignment','event_alignment']
        data = {k: round(sum(float(x.scores.get(k, 0.0)) for x in images) / n, 4) for k in keys}
        data['ending_emotion_accuracy'] = float(images[-1].scores.get('emotion_visibility', 0.0)) if images else 0.0
        data['narrative_coherence'] = round((data.get('event_grounding',0.0) + data.get('event_emotion_causal_consistency',0.0) + data.get('scene_alignment',0.0)) / 3, 4)

        if cs:
            data['contact_sheet_path'] = str(cs)
        if contact_sheet_error:
            data['contact_sheet_error'] = contact_sheet_error

        return data

