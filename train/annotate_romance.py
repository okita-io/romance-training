#!/usr/bin/env python3
import json, hashlib, argparse, asyncio, httpx, re, sys
from pathlib import Path

CACHE_DIR = Path("/Users/alexokita/romance-corpus/annotation_cache")
CACHE_DIR.mkdir(exist_ok=True)

PROMPT = """Analyze this romance excerpt and return ONLY a JSON object.

Excerpt: "{text[:1500]}"

Required JSON format (strict):
{{
  "heat_level": "sweet|mild|moderate|steamy|explicit",
  "genre": ["classic"|"contemporary"|"paranormal"|"billionaire"|"gothic"|"bwwm"|"mm"|"ff"|"age_gap"|"forbidden"|"smut"|"romantasy"],
  "plot_types": ["enemies_to_lovers"|"friends_to_lovers"|"fake_relationship"|"second_chance"|"marriage_of_convenience"|"secret_baby"|"fated_mates"|"slow_burn"],
  "tropes": ["alpha_male"|"virgin"|"dominant"|"submissive"|"bdsm"|"werewolf"|"vampire"|"shifter"|"witch"|"amnesia"|"single_dad"|"small_town"|"forced_proximity"],
  "pov": "first-person|third-limited|third-omniscient",
  "emotional_tone": "sweet|angsty|humorous|dark|steamy",
  "cliffhanger": true/false,
  "cliffhanger_strength": 1-5,
  "plot_twist": true/false,
  "plot_twist_strength": 1-5
}}

Rules:
- Select ONE heat_level
- Select 1-3 genres
- Select 0-3 plot_types
- Select 0-5 tropes
- cliffhanger_strength = 0 if cliffhanger is false
- plot_twist_strength = 0 if plot_twist is false
- Return ONLY the JSON object, no other text

Output:"""

async def query_llm(text):
    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(
            'http://10.0.1.3:1234/v1/chat/completions',
            headers={'Authorization': 'Bearer sk-lm-0VwEHEMa:V2Bi4XGJXsANMAPfxeeI', 'Content-Type': 'application/json'},
            json={
                'model': 'qwen3.5-9b-uncensored-hauhaucs-aggressive',
                'messages': [{'role': 'user', 'content': PROMPT.format(text=text)}],
                'max_tokens': 1200,
                'temperature': 0.0
            }
        )
        resp.raise_for_status()
        msg = resp.json()['choices'][0]['message']
        full = (msg.get('content', '') or msg.get('reasoning_content', '')).strip()
        full = re.sub(r'```json\s*|\s*```|```', '', full)
        start = full.find('{')
        if start == -1:
            raise ValueError("No {")
        depth = 0
        for i in range(start, len(full)):
            if full[i] == '{': depth += 1
            elif full[i] == '}':
                depth -= 1
                if depth == 0:
                    return json.loads(full[start:i+1])
        raise ValueError("Incomplete JSON")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--limit', type=int, default=100)
    args = parser.parse_args()
    if not args.output:
        args.output = args.input.with_name(args.input.stem + '_annotated.jsonl')
    
    processed = cache_hits = errors = 0
    with open(args.input) as inf, open(args.output, 'w') as outf:
        for line in inf:
            if not line.strip(): continue
            if processed >= args.limit: break
            try: sample = json.loads(line)
            except: outf.write(line); processed += 1; continue
            
            text = sample.get('text', '')
            if not text or len(text.split()) < 20:
                outf.write(json.dumps(sample, ensure_ascii=False) + '\n')
                processed += 1; continue
            
            ck = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
            cache_file = CACHE_DIR / f'{ck}.json'
            meta = sample.get('metadata', {})
            
            if cache_file.exists():
                try: 
                    with open(cache_file) as cf: meta.update(json.load(cf))
                    cache_hits += 1
                except: cache_file.unlink(missing_ok=True)
            
            if not meta.get('heat_level'):
                try:
                    result = asyncio.run(query_llm(text))
                    meta.update(result)
                    with open(cache_file, 'w') as cf: json.dump(result, cf, indent=2)
                except Exception as e:
                    errors += 1
                    if errors <= 10: print(f'  Err {processed}: {str(e)[:60]}')
            
            sample['metadata'] = meta
            outf.write(json.dumps(sample, ensure_ascii=False) + '\n')
            processed += 1
            if processed % 10 == 0:
                print(f'  {processed} (cache:{cache_hits} err:{errors})')
    
    print(f'\n✅ Done: {processed} samples, {cache_hits} cached, {errors} errors')
    print(f'Output: {args.output}')

if __name__ == '__main__':
    main()
