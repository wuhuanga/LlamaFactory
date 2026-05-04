"""
Prepare D_KP dataset: Wikipedia paragraph prefixes as prompts for OOD knowledge preservation.

Usage:
  python prepare_dkp.py [--num_prompts 10000] [--min_words 30] [--max_words 50]

Output: data/dkp_wiki_10k.jsonl  (each line: {"prompt": "..."})
"""
import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LLAMA_FACTORY_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = LLAMA_FACTORY_ROOT / "data"


def extract_first_paragraph(text: str) -> str | None:
    """Extract the first non-empty paragraph from a Wikipedia article."""
    for para in text.split("\n\n"):
        para = para.strip()
        if len(para.split()) >= 20:
            return para
    return None


def truncate_to_word_range(text: str, min_words: int, max_words: int) -> str | None:
    """Truncate text to a word count within [min_words, max_words]."""
    words = text.split()
    if len(words) < min_words:
        return None
    target = min(len(words), max_words)
    return " ".join(words[:target])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_prompts", type=int, default=10000)
    parser.add_argument("--min_words", type=int, default=30)
    parser.add_argument("--max_words", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to tokenizer for verifying token counts (optional)")
    args = parser.parse_args()

    # Try multiple sources in order of preference
    ds = None
    sources = [
        ("wikimedia/wikipedia", {"name": "20231101.en"}),
        ("wikipedia", {"name": "20220301.en"}),
        ("wikitext", {"name": "wikitext-103-raw-v1"}),
    ]

    for ds_name, ds_kwargs in sources:
        try:
            logger.info(f"Trying dataset: {ds_name} ...")
            ds = load_dataset(ds_name, split="train", streaming=True, **ds_kwargs)
            # Test that we can actually iterate
            first = next(iter(ds))
            logger.info(f"Successfully loaded {ds_name}")
            break
        except Exception as e:
            logger.warning(f"Failed to load {ds_name}: {e}")
            ds = None

    if ds is None:
        raise RuntimeError("Could not load any Wikipedia/wikitext dataset. "
                           "Please manually create data/dkp_wiki_10k.jsonl")

    # Detect text column name
    text_key = "text" if "text" in first else "page" if "page" in first else None
    if text_key is None:
        text_key = list(first.keys())[0]
    logger.info(f"Using text column: '{text_key}'")

    prompts = []
    seen = 0
    for article in ds:
        seen += 1
        text = article.get(text_key, "")
        if not text or len(text.strip()) < 50:
            continue
        para = extract_first_paragraph(text)
        if para is None:
            continue
        prompt = truncate_to_word_range(para, args.min_words, args.max_words)
        if prompt is None:
            continue
        prompts.append(prompt)
        if len(prompts) >= args.num_prompts:
            break
        if len(prompts) % 2000 == 0:
            logger.info(f"  collected {len(prompts)}/{args.num_prompts} prompts (scanned {seen} articles)")

    logger.info(f"Collected {len(prompts)} prompts from {seen} articles")

    # Optional: verify token counts with the target tokenizer
    if args.model_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        token_counts = [len(tokenizer.encode(p)) for p in prompts]
        avg_tokens = sum(token_counts) / len(token_counts)
        logger.info(f"Token stats: avg={avg_tokens:.1f}, min={min(token_counts)}, max={max(token_counts)}")

    # Save
    out_path = DATA_DIR / "dkp_wiki_10k.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps({"prompt": p}, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(prompts)} prompts to {out_path}")
    logger.info(f"Sample: {prompts[0][:120]}...")


if __name__ == "__main__":
    main()
