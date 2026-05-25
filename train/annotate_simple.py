#!/usr/bin/env python3
"""Minimal romance annotator - direct LM Studio calls, no classes."""

import json, hashlib, argparse, asyncio, httpx, sys
from pathlib import Path

CACHE_DIR = Path("/Users/alexokita/romance-corpus/annotation_cache")
CACHE_DIR.mkdir(exist_ok=True)

PROMPT_TEMPLATE = """Analyze this romance excerpt and return JSON only.

Text: {text}

Categories (use exact values):
Genre (list): classic, paranormal, billionaire, contemporary, gothic, bwwm, mm, ff, age_gap, forbidden, smut, romantasy
Heat: sweet, mild, moderate, steamy, explicit
Plot types (list): enemies_to_lovers, friends_to_lovers, fake_relationship, second_chance, fake_pregnancy, marriage_of_convenience, secret_baby, billionaire, fated_mates, forbidden_love, love_triangle, redemption, protection, slow_burn
Tropes (list): alpha_male, virgin, dominant, submissive, bdsm, werewolf, vampire, shifter, witch, amnesia, secret_baby, billionaire, single_dad, small_town, mountain_man, bad_boy, forced_proximity, marriage_of_convenience, instalove
POV: first-person, third-limited, third-omniscient
Emotional tone: sweet, angsty, humorous, dark, inspirational, steamy
Cliffhanger: true/false (1-5 strength)
Plot twist: true/false (1-5 strength)

JSON schema:
{
  "genre": ["genre"],
  "heat_level": "heat",
  "plot_types": ["type"],
  "tropes": ["trope"],
  "cliffhanger": true,
  "cliffhanger_strength": 1,
  "plot_twist": false,
  "plot_twist_strength": 0,
  "pov": "third-limited",
  "emotional_tone": "sweet"
}"""

async def query_llm(text: str, url: str = "http://10.0.1.3:1234", api_key: str = "sk-lm-0VwEHEMa:V2Bi4XGJXsANMAPfxeeI") -> dict:
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "qwen3.5-9b-uncensored-hauhaucs-aggressive",
                "messages": [
                    {"role": "system", "content": "You are a romance fiction classifier. Return only JSON."},
                    {"role": "user", "content": PROMPT_TEMPLATE.format(text=text[:1500])}
                ],
                "max_tokens": 400,
                "temperature": 0.1,
                "reasoning": "off"
            }
        )
        resp.raise_for_status()
        content = resp.json()['choices'][0]['message']['content']
        start = content.find('{')
        end = content.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(content[start:end])
        raise ValueError("No JSON in response")

def cache_key(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    
    if not args.output:
        args.output = args.input.with_name(args.input.stem + '_enriched.jsonl')
    
    with open(args.input) as inf, open(args.output, 'w') as outf:
        count = 0
        cache_hits = 0
        for line in inf:
            if not line.strip():
                continue
            if count >= args.limit if args.limit else False:
                break
            try:
                sample = json.loads(line)
            except:
                outf.write(line)
                count += 1
                continue
            
            text = sample.get('text', '')
            if not text or len(text.split()) < 30:
                outf.write(line)
                count += 1
                continue
            
            ck = cache_key(text)
            cache_file = CACHE_DIR / f"{ck}.json"
            meta = sample.get('metadata', {})
            
            if cache_file.exists():
                with open(cache_file) as cf:
                    meta.update(json.load(cf))
                cache_hits += 1
            else:
                # Query LLM
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(query_llm(text))
                    meta.update(result)
                    with open(cache_file, 'w') as cf:
                        json.dump(result, cf, indent=2)
                    loop.close()
                except Exception as e:
                    print(f"Error on sample {count}: {e}")
            
            sample['metadata'] = meta
            outf.write(json.dumps(sample, ensure_ascii=False) + '
')
            
            if (count + 1) % 5 == 0:
                print(f"  {count+1} processed, cache hits: {cache_hits}")
            
            count += 1
    
    print(f"✅ Completed: {count} samples, {cache_hits} from cache")
    print(f"Output: {args.output}")

if __name__ == '__main__':
    main()
