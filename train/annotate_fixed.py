#!/usr/bin/env python3
"""Romance annotator that handles LM Studio reasoning mode."""

import json, hashlib, argparse, asyncio, httpx, sys, re
from pathlib import Path

CACHE_DIR = Path("/Users/alexokita/romance-corpus/annotation_cache")
CACHE_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """You are a romance fiction classifier. Analyze the text and respond with ONLY a JSON object.

Required fields:
- genre: array of strings from [classic, paranormal, billionaire, contemporary, gothic, bwwm, mm, ff, age_gap, forbidden, smut, romantasy]
- heat_level: string from [sweet, mild, moderate, steamy, explicit]
- plot_types: array from [enemies_to_lovers, friends_to_lovers, fake_relationship, second_chance, fake_pregnancy, marriage_of_convenience, secret_baby, billionaire, fated_mates, forbidden_love, love_triangle, redemption, protection, slow_burn]
- tropes: array from [alpha_male, virgin, dominant, submissive, bdsm, werewolf, vampire, shifter, witch, amnesia, secret_baby, billionaire, single_dad, small_town, mountain_man, bad_boy, forced_proximity, marriage_of_convenience, instalove]
- cliffhanger: boolean
- cliffhanger_strength: integer 1-5
- plot_twist: boolean
- plot_twist_strength: integer 1-5
- pov: string from [first-person, third-limited, third-omniscient]
- emotional_tone: string from [sweet, angsty, humorous, dark, inspirational, steamy]

Return ONLY the JSON object. No other text."""

async def query_llm(text: str) -> dict:
    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(
            'http://10.0.1.3:1234/v1/chat/completions',
            headers={'Authorization': 'Bearer sk-lm-0VwEHEMa:V2Bi4XGJXsANMAPfxeeI', 'Content-Type': 'application/json'},
            json={
                'model': 'qwen3.5-9b-uncensored-hauhaucs-aggressive',
                'messages': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': f'Analyze this romance excerpt:\n\n{text[:2000]}'}
                ],
                'max_tokens': 600,
                'temperature': 0.0,
                'reasoning': 'off'  # Try to disable reasoning
            }
        )
        resp.raise_for_status()
        msg = resp.json()['choices'][0]['message']
        content = msg.get('content', '')
        reasoning = msg.get('reasoning_content', '')
        
        # Prefer reasoning_content if content is empty
        full_text = content or reasoning
        
        # Extract JSON
        json_match = re.search(r'\{.*\}', full_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        s = full_text.find('{')
        e = full_text.rfind('}') + 1
        if s >= 0 and e > s:
            return json.loads(full_text[s:e])
        raise ValueError(f'No JSON found. Content: {full_text[:200]}')

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
