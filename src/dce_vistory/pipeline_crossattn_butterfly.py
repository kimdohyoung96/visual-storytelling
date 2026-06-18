from __future__ import annotations
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List
from .llm import build_llm, build_vlm
from .planner import DCEPlanner
from .causal_memory import DCEECausalMemoryStore
from .anchor_bank import DCEEAnchorBank
from .evaluator import DCEQAEvaluator
from .image_understanding import ImageUnderstandingModule
from .schema import PipelineResult, CandidateImage
from .utils import save_json
from .prompts import QUALITY_SUFFIX, NEGATIVE_PROMPT
from .butterfly_adapter import ButterflyController
from .sdxl_cross_attention_generator import SDXLButterflyCrossAttentionGenerator

def _safe_asdict(obj):
    if is_dataclass(obj):
        d=asdict(obj); d.update({k:v for k,v in getattr(obj,'__dict__',{}).items() if k not in d}); return d
    return getattr(obj,'__dict__',obj)

class CrossAttentionButterflyDCEViStoryPipeline:
    def __init__(self, cfg:Dict[str,Any]):
        self.cfg=cfg; self.llm=build_llm(cfg.get('llm',{})); self.vlm=build_vlm(cfg.get('vlm',{}))
        self.planner=DCEPlanner(self.llm, temperature=float(cfg.get('llm',{}).get('temperature',0.4)), max_tokens=int(cfg.get('llm',{}).get('max_tokens',1800)))
        iu=cfg.get('image_understanding',{})
        self.image_understanding=ImageUnderstandingModule(iu.get('provider','llm_caption'), self.llm, iu.get('caption_model','Salesforce/blip-image-captioning-base'))
        ev=cfg.get('evaluation',{})
        self.evaluator=DCEQAEvaluator(self.llm,self.vlm,use_vlm=bool(ev.get('use_vlm',True)),save_contact_sheet=bool(ev.get('save_contact_sheet',True)))
        b=cfg.get('butterfly',{})
        self.controller=ButterflyController(b.get('quality_suffix',QUALITY_SUFFIX), b.get('negative_prompt',NEGATIVE_PROMPT), int(b.get('num_hypotheses',3)))
        ic=cfg.get('image_generator',{}); ac=cfg.get('cross_attention_adapters',{})
        self.image_generator=SDXLButterflyCrossAttentionGenerator(model_id=ic.get('model_id','stabilityai/stable-diffusion-xl-base-1.0'), device=ic.get('device','cuda'), width=int(ic.get('width',1024)), height=int(ic.get('height',1024)), num_inference_steps=int(ic.get('num_inference_steps',40)), guidance_scale=float(ic.get('guidance_scale',8.0)), seed=int(ic.get('seed',42)), adapter_ckpt=ac.get('adapter_ckpt'), enable_cpu_offload=bool(ic.get('enable_cpu_offload',False)), character_tokens=int(ac.get('character_tokens',8)), world_tokens=int(ac.get('world_tokens',8)), emotion_tokens=int(ac.get('emotion_tokens',8)), event_tokens=int(ac.get('event_tokens',8)), evidence_tokens=int(ac.get('evidence_tokens',8)))
    def _strengthen_packet(self, packet, frame):
        packet.positive_prompt += f"\n\nRETRY: Make DCEE event and evidence unmistakable. Event: {frame.event}. Evidence: {getattr(frame,'emotion_evidence',[])}. Emotion: {frame.emotion}. Use full color."
        packet.negative_prompt += '; missing evidence, missing event grounding, weak emotion, grayscale'
        m=packet.control_metadata or {}; m['evidence_text']=m.get('evidence_text','')+f"; MUST SHOW evidence: {getattr(frame,'must_show',[])}"; packet.control_metadata=m
        packet.adapter_weights['evidence_adapter']=min(0.45, packet.adapter_weights.get('evidence_adapter',0.2)+0.12); packet.adapter_weights['event_adapter']=min(0.40, packet.adapter_weights.get('event_adapter',0.18)+0.08); packet.adapter_weights['emotion_adapter']=min(0.45, packet.adapter_weights.get('emotion_adapter',0.28)+0.06)
        return packet
    def run(self, sample:Dict[str,Any], out_dir:Path) -> PipelineResult:
        out_dir=Path(out_dir); frames_dir=out_dir/'frames'; ending_dir=out_dir/'ending_candidates'; out_dir.mkdir(parents=True, exist_ok=True)
        image_summary=self.image_understanding.analyze(sample.get('image_path'), sample)
        seed=self.planner.build_seed(sample, image_summary); abstract=self.planner.generate_abstract(seed); dce_plan=self.planner.generate_dce_plan(seed, abstract)
        emotion_arc=self.planner.generate_emotion_arc(seed, abstract, dce_plan, int(sample.get('num_frames',6))); storyboard=self.planner.generate_storyboard(seed, abstract, dce_plan, emotion_arc)
        save_json(_safe_asdict(seed), out_dir/'seed.json'); (out_dir/'abstract.txt').write_text(abstract, encoding='utf-8'); save_json(_safe_asdict(dce_plan), out_dir/'dcee_plan.json'); save_json(_safe_asdict(dce_plan), out_dir/'dce_plan.json'); save_json(getattr(dce_plan,'dcee_candidates',[]), out_dir/'dcee_candidate_plans.json'); save_json(_safe_asdict(emotion_arc), out_dir/'emotion_arc.json'); save_json([_safe_asdict(x) for x in storyboard], out_dir/'storyboard.json')
        memory=DCEECausalMemoryStore(); memory.initialize(seed,dce_plan); anchor_bank=DCEEAnchorBank().build_from_seed_and_plan(seed,dce_plan)
        selected:List[CandidateImage]=[]; ending_candidates=[]; packet_log=[]; memory_log=[]
        ic=self.cfg.get('image_generator',{}); pc=self.cfg.get('pipeline',{})
        n=int(ic.get('num_candidates_per_frame',3)); ne=int(ic.get('num_ending_candidates',5)); retry=bool(pc.get('emotion_retry',True)); eth=float(pc.get('emotion_visibility_threshold',0.76)); cth=float(pc.get('colorfulness_threshold',0.35)); egh=float(pc.get('event_grounding_threshold',0.70)); evh=float(pc.get('evidence_visibility_threshold',0.68))
        style=sample.get('style','full-color cinematic storybook illustration'); style=('full-color '+style) if 'color' not in style.lower() else style
        prev=None
        for idx,frame in enumerate(storyboard):
            mem=memory.select(frame,dce_plan,emotion_arc,strategy=pc.get('memory_strategy','adaptive_causal'))
            anchors=anchor_bank.select_for_frame(frame)
            packet=self.controller.create_packet(frame,seed,dce_plan,mem,style,previous_frame=prev, anchors=anchors)
            last=idx==len(storyboard)-1; tdir=ending_dir if last else frames_dir; count=ne if last else n
            cands=self.image_generator.generate_from_packet(packet, frame.frame_id, tdir, count)
            ranked=self.evaluator.rerank_ending_candidates(frame,dce_plan,cands) if last else self.evaluator.rank_frame_candidates(frame,dce_plan,cands,False)
            best=ranked[0]; retried=False
            if retry and (best.scores.get('emotion_visibility',0)<eth or best.scores.get('colorfulness',0)<cth or best.scores.get('event_grounding',0)<egh or best.scores.get('evidence_visibility',0)<evh):
                retried=True; sp=self._strengthen_packet(packet,frame); rc=self.image_generator.generate_from_packet(sp, frame.frame_id, tdir, max(2,count//2)); rr=self.evaluator.rerank_ending_candidates(frame,dce_plan,rc) if last else self.evaluator.rank_frame_candidates(frame,dce_plan,rc,False)
                if rr and rr[0].scores.get('overall',0)>best.scores.get('overall',0): ranked=rr; best=rr[0]; packet=sp
            best.notes['retried_for_dcee_event_evidence_or_emotion']=retried
            if last: ending_candidates=ranked
            selected.append(best); memory.add(frame,best); prev=frame
            packet_log.append({'frame_id':frame.frame_id,'visual_control_packet':_safe_asdict(packet),'anchors':anchors,'retried':retried}); memory_log.append({'frame_id':frame.frame_id,'memory':mem,'selected_image':_safe_asdict(best),'all_candidates':[_safe_asdict(c) for c in ranked]})
        save_json(packet_log,out_dir/'visual_control_packets.json'); save_json(memory_log,out_dir/'memory_log.json'); save_json([_safe_asdict(x) for x in ending_candidates],out_dir/'ending_candidates.json')
        questions=self.evaluator.generate_questions(dce_plan,emotion_arc,storyboard); save_json(questions,out_dir/'eval_questions.json')
        evaluation=self.evaluator.evaluate_sequence(dce_plan,emotion_arc,storyboard,selected,questions,out_dir); save_json(evaluation,out_dir/'evaluation.json')
        md=self._build_markdown(abstract,dce_plan,emotion_arc,storyboard,selected,ending_candidates,evaluation); (out_dir/'final_story.md').write_text(md,encoding='utf-8')
        return PipelineResult(seed=seed, abstract=abstract, dce_plan=dce_plan, emotion_arc=emotion_arc, storyboard=storyboard, selected_images=selected, ending_candidates=ending_candidates, evaluation_questions=questions, evaluation=evaluation, final_story_markdown=md)
    @staticmethod
    def _build_markdown(abstract,dce_plan,emotion_arc,storyboard,images,ending_candidates,evaluation):
        lines=['# DCEE-CausalVerse Visual Story\n','## Abstract\n',abstract+'\n','## Selected DCEE Plan\n',f"- Desire: {dce_plan.desire}",f"- Conflict: {dce_plan.conflict}",f"- Planning Structure: {getattr(dce_plan,'planning_structure','DCEE')}",f"- Ending Emotion: {dce_plan.target_ending_emotion}\n",'## Frames\n']
        for f,img in zip(storyboard,images): lines += [f"### Frame {f.frame_id}: {f.narrative_function}",f"![Frame {f.frame_id}]({img.image_path})",f"- Event: {f.event}",f"- Event Grounding: {getattr(f,'event_grounding','')}",f"- Emotion: {f.emotion} ({f.emotion_intensity}/5)",f"- Evidence: {getattr(f,'emotion_evidence',[])}",f"- Scores: {img.scores}\n"]
        lines.append('## Evaluation\n'); [lines.append(f'- {k}: {v}') for k,v in evaluation.items()]
        return '\n'.join(lines)
