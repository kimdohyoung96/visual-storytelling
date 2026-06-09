import json
import os
from typing import Any, Dict, List

import requests

from .utils import image_to_data_url


class BaseLLM:
    def generate(self, system: str, user: str, temperature: float = 0.4, max_tokens: int = 2000) -> str:
        raise NotImplementedError


class BaseVLM:
    def generate_with_images(self, system: str, user: str, image_paths: List[str], temperature: float = 0.0, max_tokens: int = 1200) -> str:
        raise NotImplementedError


class OpenAICompatibleLLM(BaseLLM):
    def __init__(self, model: str, base_url: str, api_key_env: str = 'OPENAI_API_KEY'):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.api_key = os.environ.get(api_key_env)

    def generate(self, system: str, user: str, temperature: float = 0.4, max_tokens: int = 2000) -> str:
        if not self.api_key:
            return DummyLLM().generate(system, user, temperature, max_tokens)
        url = f'{self.base_url}/chat/completions'
        headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
        payload = {'model': self.model, 'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}], 'temperature': temperature, 'max_tokens': max_tokens}
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=180)
        if not resp.ok:
            print("STATUS:", resp.status_code)
            print("URL:", resp.url)
            print("RESPONSE:", resp.text)
        resp.raise_for_status()
        data = resp.json()
        return data['choices'][0]['message']['content']


class OpenAICompatibleVLM(BaseVLM):
    def __init__(self, model: str, base_url: str, api_key_env: str = 'OPENAI_API_KEY'):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.api_key = os.environ.get(api_key_env)

    def generate_with_images(self, system: str, user: str, image_paths: List[str], temperature: float = 0.0, max_tokens: int = 1200) -> str:
        if not self.api_key:
            return DummyVLM().generate_with_images(system, user, image_paths, temperature, max_tokens)
        content = [{'type': 'text', 'text': user}]
        for path in image_paths:
            content.append({'type': 'image_url', 'image_url': {'url': image_to_data_url(path)}})
        url = f'{self.base_url}/chat/completions'
        headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
        payload = {'model': self.model, 'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}], 'temperature': temperature, 'max_tokens': max_tokens}
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=180)
        resp.raise_for_status()
        data = resp.json()
        return data['choices'][0]['message']['content']


class DummyLLM(BaseLLM):
    def generate(self, system: str, user: str, temperature: float = 0.4, max_tokens: int = 2000) -> str:
        lower = user.lower()
        if 'create a dce plan' in lower or 'desire-conflict-ending' in lower or 'desire, conflict, ending' in lower:
            return json.dumps({'protagonist':'turtle','desire':'to prove that being slow does not mean being worthless','fear':'being laughed at and forgotten','misbelief':'only winners deserve respect','obstacle':'the rabbit is faster and openly mocks the turtle','conflict':'the turtle must continue despite humiliation, doubt, and the rabbit\'s natural advantage','event_spine':['the turtle accepts the race','the rabbit humiliates the turtle in public','the turtle struggles alone on the path','the turtle nearly gives up','the turtle keeps moving while the rabbit loses focus','the turtle reaches the finish line and earns recognition'],'turning_point':'the turtle decides that finishing with dignity matters more than winning quickly','target_ending_emotion':'happy','ending_state':'the turtle feels joyful, respected, and relieved','moral_or_theme':'steady persistence can defeat arrogance'}, indent=2)
        if 'create an emotion arc planner' in lower or 'emotion arc planner' in lower:
            return json.dumps({'states':['hope','humiliation','doubt','determination','relief','joy'],'intensities':[2,4,4,5,4,5],'rationale':'the protagonist begins with hope, falls into doubt under conflict, then rises toward relief and joy'}, indent=2)
        if 'create a 6-frame storyboard' in lower or 'storyboard' in lower:
            frames = [
                {'frame_id':1,'caption':'The turtle accepts the rabbit\'s challenge at the forest starting line.','narrative_function':'inciting incident','event':'the race begins','protagonist_state':'The turtle is hopeful but nervous.','desire_link':'The turtle wants to prove its worth.','conflict_level':1,'emotion':'hope','emotion_intensity':2,'visual_focus':'turtle, rabbit, starting line','key_objects':['starting line','forest path'],'facial_cue':'soft determined eyes','body_cue':'upright posture with cautious tension','event_cue':'accepting a public challenge','scene_cue':'forest clearing, race beginning','cinematic_cue':'clean composition, bright daylight, anticipatory framing'},
                {'frame_id':2,'caption':'The rabbit laughs as the crowd doubts the turtle.','narrative_function':'conflict introduction','event':'the turtle is mocked','protagonist_state':'The turtle feels humiliated and small.','desire_link':'Mockery deepens the turtle\'s need for recognition.','conflict_level':2,'emotion':'humiliation','emotion_intensity':4,'visual_focus':'laughing rabbit, small turtle, crowd','key_objects':['crowd','forest path'],'facial_cue':'lowered eyes and tight mouth','body_cue':'slightly hunched shoulders','event_cue':'public humiliation','scene_cue':'onlookers watching and laughing','cinematic_cue':'slightly off-center framing, mild visual pressure'},
                {'frame_id':3,'caption':'The turtle climbs a small hill while the rabbit speeds far ahead.','narrative_function':'rising tension','event':'the gap widens','protagonist_state':'The turtle is doubtful and physically strained.','desire_link':'The desire persists despite visible disadvantage.','conflict_level':3,'emotion':'doubt','emotion_intensity':4,'visual_focus':'lonely turtle on hill, distant rabbit','key_objects':['small hill','forest path'],'facial_cue':'tired eyes','body_cue':'leaning forward with effort','event_cue':'struggling against odds','scene_cue':'uphill path, distance between characters','cinematic_cue':'larger negative space, cooler tone, diagonal composition'},
                {'frame_id':4,'caption':'The turtle nearly stops, then chooses to keep moving.','narrative_function':'turning point','event':'the turtle refuses to quit','protagonist_state':'The turtle transforms doubt into determination.','desire_link':'The protagonist recommits to the original desire.','conflict_level':4,'emotion':'determination','emotion_intensity':5,'visual_focus':'close-up turtle resuming the journey','key_objects':['forest path','small hill'],'facial_cue':'focused eyes and firm expression','body_cue':'low but steady forward motion','event_cue':'choosing perseverance','scene_cue':'solitary path and resumed movement','cinematic_cue':'closer shot, stronger contrast, motivational framing'},
                {'frame_id':5,'caption':'The rabbit wakes in panic as the turtle approaches the finish line.','narrative_function':'climax','event':'the outcome reverses','protagonist_state':'The turtle feels relief mixed with urgency.','desire_link':'The desire is finally becoming achievable.','conflict_level':5,'emotion':'relief','emotion_intensity':4,'visual_focus':'finish line ahead, rabbit shocked','key_objects':['finish line','forest path'],'facial_cue':'calm, relieved concentration','body_cue':'steady confident posture','event_cue':'reversal of expectations','scene_cue':'dramatic approach to finish line','cinematic_cue':'dynamic framing, brighter light, rising momentum'},
                {'frame_id':6,'caption':'The turtle crosses the finish line as the forest cheers.','narrative_function':'resolution','event':'the turtle earns recognition','protagonist_state':'The turtle feels joyful, respected, and free.','desire_link':'The turtle finally proves its worth.','conflict_level':3,'emotion':'joy','emotion_intensity':5,'visual_focus':'turtle at finish line, cheering crowd','key_objects':['finish line','cheering animals'],'facial_cue':'broad smile and bright eyes','body_cue':'upright triumphant posture','event_cue':'successful completion and social recognition','scene_cue':'cheering forest crowd near finish line','cinematic_cue':'warm sunlight, vivid colors, centered triumphant composition'}
            ]
            return json.dumps(frames, indent=2)
        if 'generate evaluation questions' in lower or 'evaluation questions' in lower:
            return json.dumps({'global_questions':['Do the frames form a coherent story sequence?','Is the protagonist desire visible?','Is the ending emotion clearly expressed?'],'frame_questions':{'1':['Does the first frame introduce the protagonist and goal?'],'2':['Is conflict visible in this frame?'],'3':['Does the frame intensify the conflict?'],'4':['Does the protagonist face a turning point?'],'5':['Does the frame show climax?'],'6':['Does the frame clearly show the target ending emotion?']},'ending_questions':['Does the final frame express joy?','Does the final frame resolve the conflict?','Does the final frame show the protagonist achieving the ending state?']}, indent=2)
        if 'input image for multimodal story planning' in lower or 'describe the input image' in lower:
            return json.dumps({'caption':'A rabbit and a turtle stand near a forest path, preparing for a race.','characters':['rabbit','turtle'],'setting':'forest path','objects':['path','trees','finish line'],'mood':'playful but competitive','inferred_plot_hint':'a race or challenge is about to begin'}, indent=2)
        if 'write a story abstract' in lower or ('story abstract' in lower and 'return json' not in lower):
            return ('In a forest race, a turtle who is often underestimated accepts a challenge from a proud rabbit. What begins as a simple race becomes a test of dignity and perseverance. As the rabbit mocks the turtle and the crowd doubts its chances, the turtle struggles through self-doubt. The story ends with the turtle proving its worth and reaching a joyful ending through persistence.')
        return '{}'


class DummyVLM(BaseVLM):
    def generate_with_images(self, system: str, user: str, image_paths: List[str], temperature: float = 0.0, max_tokens: int = 1200) -> str:
        if 'ending candidates' in user.lower():
            return json.dumps({'candidate_scores':[{'candidate_id':0,'ending_emotion_accuracy':0.80,'visual_clarity':0.75,'overall':0.78,'reason':'good joyful cue'},{'candidate_id':1,'ending_emotion_accuracy':0.84,'visual_clarity':0.80,'overall':0.82,'reason':'clear ending emotion'},{'candidate_id':2,'ending_emotion_accuracy':0.74,'visual_clarity':0.76,'overall':0.75,'reason':'acceptable but weaker emotion'},{'candidate_id':3,'ending_emotion_accuracy':0.79,'visual_clarity':0.77,'overall':0.78,'reason':'good but not strongest'}]}, indent=2)
        return json.dumps({'text_image_alignment':0.72,'character_consistency':0.70,'desire_alignment':0.82,'conflict_visibility':0.75,'emotion_arc_accuracy':0.80,'ending_emotion_accuracy':0.84,'narrative_coherence':0.79,'interestingness':0.78,'rationale':'DummyVLM placeholder evaluation based on expected narrative progression.'}, indent=2)


def build_llm(cfg: Dict[str, Any]) -> BaseLLM:
    provider = cfg.get('provider', 'openai_compatible')
    if provider == 'openai_compatible':
        return OpenAICompatibleLLM(cfg.get('model', 'gpt-4o-mini'), cfg.get('base_url', 'https://api.openai.com/v1'), cfg.get('api_key_env', 'OPENAI_API_KEY'))
    return DummyLLM()


def build_vlm(cfg: Dict[str, Any]) -> BaseVLM:
    provider = cfg.get('provider', 'openai_compatible')
    if provider == 'openai_compatible':
        return OpenAICompatibleVLM(cfg.get('model', 'gpt-4o-mini'), cfg.get('base_url', 'https://api.openai.com/v1'), cfg.get('api_key_env', 'OPENAI_API_KEY'))
    return DummyVLM()
