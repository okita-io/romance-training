#!/usr/bin/env python3
import json, hashlib, argparse, asyncio, httpx, sys
from pathlib import Path

CACHE_DIR = Path('/Users/alexokita/romance-corpus/annotation_cache')
CACHE_DIR.mkdir(exist_ok=True)

PROMPT = '''Analyze this romance excerpt and return JSON only.'''.strip()

async def query_llm(text):
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            'http://10.0.1.3:1234/v1/chat/completions',
            headers={'Authorization': 'Bearer sk-lm-0VwEHEMa:V2Bi4XGJXsANMAPfxeeI', 'Content-Type': 'application/json'},
            json={
                'model': 'qwen3.5-9b-uncensored-hauhaucs-aggressive',
                'messages': [{
                    'role': 'system',
                    'content': 'Return JSON: {genre:[], heat_level:string, plot_types:[], tropes:[], cliffhanger:bool, cliffhanger_strength:int, plot_twist:bool, plot_twist_strength:int, pov:string, emotional_tone:string}'
                }, {
                    'role': 'user',
                    'content': f'Analyze: {text[:1500]}'
                }],
                'max_tokens': 400,
                'temperature': 0.1,
                'reasoning': 'off'
            })
        resp.raise_for_status()
        content = resp.json()['choices'][0]['message']['content']
        s = content.find('{')
        e = content.rfind('}') + 1
        if s >= 0 and e > s:
            return json.loads(content[s:e])
        raise ValueError('No JSON')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if not args.output:
        args.output = args.input.with_name(args.input.stem + '_enriched.jsonl')
    with open(args.input) as inf, open(args.output, 'w') as outf:
        count = cache_hits = 0
        for line in inf:
            if not line.strip(): continue
            if args.limit and count >= args.limit: break
            try: sample = json.loads(line)
            except: outf.write(line); count += 1; continue
            text = sample.get('text', '')
            if not text or len(text.split()) < 30: outf.write(line); count += 1; continue
            ck = hashlib.sha256(text.encode()).hexdigest()[:16]
            cache_file = CACHE_DIR / f'{ck}.json'
            meta = sample.get('metadata', {})
            if cache_file.exists():
                with open(cache_file) as cf: meta.update(json.load(cf))
                cache_hits += 1
            else:
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(query_llm(text))
                    meta.update(result)
                    with open(cache_file, 'w') as cf: json.dump(result, cf, indent=2)
                    loop.close()
                except Exception as e: print(f'Error {count}: {e}')
            sample['metadata'] = meta
            outf.write(json.dumps(sample, ensure_ascii=False) + '\n')
            if (count + 1) % 5 == 0: print(f'  {count+1} processed, cache hits: {cache_hits}')
            count += 1
    print(f'✅ {count} samples, {cache_hits} from cache. Output: {args.output}')

if __name__ == '__main__':
    main()