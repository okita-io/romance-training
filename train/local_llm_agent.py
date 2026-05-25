#!/usr/bin/env python3

"""
Local LLM Agent for scraping and curating romance fiction data sources.
Coordinates with an LMStudio-hosted OpenAI-compatible LLM (e.g., openai-gpt-oss-20b-heretic-uncensored-neo-imatrix)
to fetch, evaluate, and collect suitable romance text for training.

Usage:
    python local_llm_agent.py --seeds seeds.txt --output romance_corpus.jsonl
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

import requests
from bs4 import BeautifulSoup
import html2text
from openai import OpenAI

# ----------------------------
# Configuration
# ----------------------------
LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
# Use the provided token
LMSTUDIO_API_KEY = "sk-lm-N3auileW:s9AHi8r85ABFOv6Sc4XR"
MODEL_NAME = "openai-gpt-oss-20b-heretic-uncensored-neo-imatrix"  # adjust if different

# Initialize OpenAI client pointing to LM Studio
client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# HTML to text converter
html_converter = html2text.HTML2Text()
html_converter.ignore_links = False
html_converter.ignore_images = True
html_converter.body_width = 0  # No wrapping

@dataclass
class ScrapedItem:
    url: str
    title: str
    content: str
    word_count: int
    romance_score: float  # 0.0 to 1.0 from LLM judgment
    kept: bool
    notes: str = ""

def fetch_url(url: str, timeout: int = 15) -> Optional[str]:
    """Fetch raw HTML from a URL with a simple user-agent."""
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; RomanceFactoryScraper/1.0; +https://github.com/alexokita/romance-factory)"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None

def extract_text(html: str) -> Dict[str, str]:
    """Extract title and main text content from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    # Remove script/style elements
    for script in soup(["script", "style", "noscript"]):
        script.decompose()
    title = soup.title.string if soup.title else "No Title"
    # Try to get main content: prioritize <article>, then <main>, then body
    main = soup.find('article') or soup.find('main') or soup.body
    if main:
        text = html_converter.handle(str(main))
    else:
        text = html_converter.handle(str(soup))
    # Clean up excessive newlines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = chr(10).join(lines)  # Use chr(10) for newline to avoid quote issues
    return {"title": title.strip(), "text": cleaned}

def llm_judge_romance(title: str, excerpt: str) -> tuple[float, str]:
    """
    Ask the local LLM to judge whether the text is suitable romance fiction
    and provide a score plus brief reasoning.
    Returns (score, reasoning).
    """
    prompt = f"""You are an expert evaluator of romance fiction for training a language model.
Given the title and an excerpt of a text, evaluate:
1. Whether the text is primarily romance fiction (focus on romantic relationships, emotional development, love story).
2. The quality and suitability for training a modern romance LLM (coherent, engaging, appropriate length, not overly verbose or repetitive).
Provide a score from 0.0 (not romance/unsuitable) to 1.0 (excellent romance material).
Also give a brief one-sentence reasoning.

Title: {title}
Excerpt: {excerpt[:1500]}  # limit excerpt size

Respond in JSON format exactly as:
{"score": <float>, "reasoning": "<string>"}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=150,
        )
        content = response.choices[0].message.content.strip()
        # Attempt to parse JSON
        data = json.loads(content)
        score = float(data.get("score", 0.0))
        reasoning = data.get("reasoning", "No reasoning provided.")
        # Clamp score
        score = max(0.0, min(1.0, score))
        return score, reasoning
    except json.JSONDecodeError:
        logger.warning(f"LLM did not return valid JSON: {content}")
        # Fallback: look for a number in the response
        import re
        match = re.search(r"0?\.\d+|1\.0?", content)
        score = float(match.group()) if match else 0.0
        reasoning = "Parsed from free-text response."
        return score, reasoning
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return 0.0, f"LLM error: {e}"

def process_url(url: str) -> Optional[ScrapedItem]:
    """Fetch, extract, judge, and return a ScrapedItem if worth keeping."""
    logger.info(f"Processing {url}")
    html = fetch_url(url)
    if not html:
        return None
    extracted = extract_text(html)
    if not extracted["text"] or len(extracted["text"].split()) < 50:
        logger.info(f"Too little content at {url}")
        return None
    # Use first 1500 chars for judgment
    excerpt = extracted["text"][:1500]
    score, reasoning = llm_judge_romance(extracted["title"], excerpt)
    kept = score >= 0.6  # threshold; adjust as needed
    item = ScrapedItem(
        url=url,
        title=extracted["title"],
        content=extracted["text"],
        word_count=len(extracted["text"].split()),
        romance_score=score,
        kept=kept,
        notes=reasoning
    )
    logger.info(f"Score: {score:.2f} - {'KEPT' if kept else 'REJECTED'} - {reasoning}")
    return item

def main():
    parser = argparse.ArgumentParser(description="Local LLM Agent for romance data scraping")
    parser.add_argument("--seeds", required=True, help="File containing seed URLs, one per line")
    parser.add_argument("--output", default="romance_corpus.jsonl", help="Output JSONL file")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds)")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of URLs to process (0 = no limit)")
    args = parser.parse_args()

    # Read seeds
    if not os.path.exists(args.seeds):
        logger.error(f"Seed file not found: {args.seeds}")
        sys.exit(1)
    with open(args.seeds, 'r') as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    if args.limit > 0:
        urls = urls[:args.limit]

    logger.info(f"Starting with {len(urls)} seed URLs")
    kept_count = 0
    with open(args.output, 'a', encoding='utf-8') as out_f:
        for i, url in enumerate(urls, start=1):
            item = process_url(url)
            if item:
                out_f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
                out_f.flush()
                if item.kept:
                    kept_count += 1
            logger.info(f"Progress: {i}/{len(urls)} - Kept: {kept_count}")
            if i < len(urls):
                time.sleep(args.delay)
    logger.info(f"Finished. Kept {kept_count} items. Output saved to {args.output}")

if __name__ == "__main__":
    main()