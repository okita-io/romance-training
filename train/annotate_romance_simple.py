#!/usr/bin/env python3
"""Simple romance metadata annotator - one file version."""

import json, hashlib, argparse, sys
from pathlib import Path
from typing import Dict, Any, List
import asyncio
import httpx

SYSTEM_PROMPT = r"""You are an expert romance fiction analyst with deep knowledge of romance genres, tropes, and narrative structures. You analyze text samples and provide structured metadata in JSON format.

Important: Analyze ONLY the provided text. Do not make up information not present in the text.

Categories (use exact values):
Genre: classic, paranormal, billionaire, contemporary, gothic, bwwm, mm, ff, age_gap, forbidden, smut, romantasy
Heat: sweet, mild, moderate, steamy, explicit
Plot types: enemies_to_lovers, friends_to_lovers, fake_relationship, second_chance, fake_pregnancy, marriage_of_convenience, secret_baby, billionaire, ordinary_to_celebrity, fated_mates, forbidden_love, love_triangle, redemption, protection, teacher_student, boss_employee, neighbor, instalove, slow_burn
Tropes: alpha_male, bro_code, virgin, first_time, grooming, dominant, submissive, orgasm_denial, bdsm, polygamy, menage, haute_couture, running_of_the_bulls, captive_romance, forced_proximity, amnesia, disguise, twins, mail_order, mafia, werewolf, vampire, shifter, witch, fae, angel, demon, god, time_travel, reincarnation, parallel_universe, dystopian, post_apocalyptic, space, royalty, sports, music, art, cooking, farm, mountain_man, lone_wolf, bad_boy, boy_next_door, single_dad, single_mom, widow, divorced, military, cowboy, rancher, pirate, viking, knight, highlander, regency, victorian, western, christmas, holiday, beach, small_town, island, castle, mansion, boardroom, school, college, hospital
POV: first-person, third-limited, third-omniscient
Emotional tone: sweet, angsty, humorous, dark, inspirational, steamy

Respond with ONLY JSON:
{
  "genre": ["genre1", "genre2"],
  "heat_level": "heat",
  "plot_types": ["plot_type1"],
  "tropes": ["trope1", "trope2"],
  "cliffhanger": true/false,
  "cliffhanger_type": "type or null",
  "cliffhanger_strength": 1-5,
  "plot_twist": true/false,
  "plot_twist_type": "type or null",
  "plot_twist_strength": 1-5,
  "pov": "pov",
  "emotional_tone": "tone"
}"""

CACHE_DIR = Path("/Users/alexokita/romance-corpus/annotation_cache")
CACHE_DIR.mkdir(exist_ok=True)

def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

def get_cache(text: str):
    h = text_hash(text)
    cache_file = CACHE_DIR / f"{h}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    return None

def save_cache(text: str, result: Dict):
    h = text_hash(text)
    with open(CACHE_FILE, 'w') as f:
        json.dump(result, f, indent=2)

async def query_llm(prompt: str, text: str, url: str = "http://10.0.1.3:1234", api_key: str = "sk-lm-0VwEHEMa:V2Bi4XGJXsANMAPfxeeI") -> Dict:
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "qwen3.5-9b-uncensored-hauhaucs-aggressive",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 800,
                "temperature": 0.1,
                "reasoning": "off"
            }
        )
        resp.raise_for_status()
        data = resp.json()
        content = data['choices'][0]['message']['content']
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end != 0:
            return json.loads(content[start:end])
        raise ValueError("No JSON found")

def build_prompt(text: str, title: str = "", max_len: int = 1200) -> str:
    if len(text) > max_len:
        half = max_len // 2
        text = text[:half] + "

[...]

" + text[-half:]
    return f"""Analyze this romance fiction excerpt{f' titled "{title}"' if title else ''}.

Text to analyze:
---
{text}
---

Provide a comprehensive analysis in the required JSON format."""

def enrich_sample(sample: Dict, use_cache: bool = True) -> Dict:
    text = sample.get('text', '')
    if not text or len(text.split()) < 50:
        return sample
    
    meta = sample.get('metadata', {})
    title = meta.get('title', '')
    
    if use_cache:
        cached = get_cache(text)
        if cached:
            meta.update(cached)
            sample['metadata'] = meta
            return sample
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        prompt = build_prompt(text, title)
        result = loop.run_until_complete(query_llm(prompt, text))
        meta.update(result)
        save_cache(text, result)
    except Exception as e:
        print(f"Warning: LLM annotation failed: {e}")
    finally:
        loop.close()
    
    sample['metadata'] = meta
    return sample

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    
    if not args.output:
        args.output = args.input.with_name(f"{args.input.stem}_enriched{args.input.suffix}")
    
    print(f"Enriching {args.input} → {args.output}")
    
    with open(args.input) as infile, open(args.output, 'w') as outfile:
        count = 0
        cache_hits = 0
        for line in infile:
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except:
                outfile.write(line)
                count += 1
                if args.limit and count >= args.limit:
                    break
                continue
            
            text = sample.get('text', '')
            h = text_hash(text) if text else None
            if h and (CACHE_DIR / f"{h}.json").exists():
                cache_hits += 1
            
            enriched = enrich_sample(sample, use_cache=True)
            outfile.write(json.dumps(enriched, ensure_ascii=False) + '
')
            
            if (count + 1) % 10 == 0:
                print(f"  {count+1} samples (cache hits: {cache_hits})")
            
            count += 1
            if args.limit and count >= args.limit:
                break
    
    print(f"✅ Done. {count} samples processed. Cache hits: {cache_hits}")

if __name__ == '__main__':
    main()
