#!/usr/bin/env python3
"""Working romance annotator with strict JSON enforcement."""

import json, hashlib, argparse, asyncio, httpx, sys, re
from pathlib import Path

CACHE_DIR = Path("/Users/alexokita/romance-corpus/annotation_cache")
CACHE_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """You are a romance fiction classifier. Analyze the text and respond with ONLY a JSON object.

Required JSON structure:
{
  "genre": ["classic"|"paranormal"|"billionaire"|"contemporary"|"gothic"|"bwwm"|"mm"|"ff"|"age_gap"|"forbidden"|"smut"|"romantasy"],
  "heat_level": "sweet"|"mild"|"moderate"|"steamy"|"explicit",
  "plot_types": ["enemies_to_lovers"|"friends_to_lovers"|"fake_relationship"|"second_chance"|"fake_pregnancy"|"marriage_of_convenience"|"secret_baby"|"billionaire"|"fated_mates"|"forbidden_love"|"love_triangle"|"redemption"|"protection"|"slow_burn"],
  "tropes": ["alpha_male"|"virgin"|"dominant"|"submissive"|"bdsm"|"werewolf"|"vampire"|"shifter"|"witch"|"amnesia"|"secret_baby"|"billionaire"|"single_dad"|"small_town"|"mountain_man"|"bad_boy"|"forced_proximity"|"marriage_of_convenience"|"instalove"],
  "cliffhanger": boolean,
  "cliffhanger_strength": 1-5,
  "plot_twist": boolean,
  "plot_twist_strength": 1-5,
  "pov": "first-person"|"third-limited"|"third-omniscient",
  "emotional_tone": "sweet"|"angsty"|"humorous"|"dark"|"inspirational"|"steamy"
}

Return ONLY the JSON, no other text."""

async def query_llm(text: str) -> dict:
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            'http://10.0.1.3:1234/v1/chat/completions',
            headers={'Authorization': 'Bearer sk-lm-0VwEHEMa:V2Bi4XGJXsANMAPfxeeI', 'Content-Type': 'application/json'},
            json={
                'model': 'qwen3.5-9b-uncensored-hauhaucs-aggressive',
                'messages': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': f'Analyze this romance excerpt:\n\n{text[:2000]}'}
                ],
                'max_tokens': 500,
                'temperature': 0.0,
                'reasoning': 'off'
            }
        )
        resp.raise_for_status()
        content = resp.json()['choices'][0]['message']['content']
        # Extract JSON more aggressively
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        # Try finding first { }
        s = content.find('{')
        e = content.rfind('}') + 1
        if s >= 0 and e > s:
            return json.loads(content[s:e])
        raise ValueError(f'No JSON found: {content[:200]}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    
    if not args.output:
        args.output = args.input.with_name(args.input.stem + '_annotated.jsonl')
    
    with open(args.input) as inf, open(args.output, 'w') as outf:
        count = cache_hits = 0
        for line in inf:
            if not line.strip(): continue
            if args.limit and count >= args.limit: break
            try: sample = json.loads(line)
            except: outf.write(line); count += 1; continue
            
            text = sample.get('text', '')
            if not text or len(text.split()) < 30:
                outf.write(line)
                count += 1
                continue
            
            ck = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
            cache_file = CACHE_DIR / f'{ck}.json'
            meta = sample.get('metadata', {})
            
            if cache_file.exists():
                with open(cache_file) as cf:
                    meta.update(json.load(cf))
                cache_hits += 1
            else:
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(query_llm(text))
                    meta.update(result)
                    with open(cache_file, 'w') as cf:
                        json.dump(result, cf, indent=2)
                    loop.close()
                except Exception as e:
                    print(f'Error {count}: {str(e)[:100]}')
            
            sample['metadata'] = meta
            outf.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            if (count + 1) % 2 == 0:
                print(f'  {count+1} done (cache hits: {cache_hits})')
            
            count += 1
    
    print(f'\n✅ {count} samples, {cache_hits} from cache')
    print(f'Output: {args.output}')

if __name__ == '__main__':
    main()
