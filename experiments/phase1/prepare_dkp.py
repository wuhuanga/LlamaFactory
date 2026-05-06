"""
Prepare D_KP dataset: Wikipedia (prompt, continuation) pairs for off-policy OOD knowledge preservation.

Each output line: {"prompt_text": "...", "continuation_text": "..."}

Includes PopQA + TriviaQA entity filtering to prevent data contamination.

Usage:
  python prepare_dkp.py [--num_prompts 10000] [--skip_entity_filter]
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


def extract_first_paragraph(text: str, min_words: int = 80) -> str | None:
    """Extract the first paragraph with at least `min_words` words."""
    for para in text.split("\n\n"):
        para = para.strip()
        if len(para.split()) >= min_words:
            return para
    return None


def split_prompt_continuation(
    para: str, min_prompt_words: int, max_prompt_words: int, cont_words: int = 40
) -> tuple[str, str] | None:
    """Split a paragraph into (prompt, continuation).

    prompt: first min_prompt_words..max_prompt_words words
    continuation: next ~cont_words words after prompt
    """
    words = para.split()
    prompt_len = min(len(words), max_prompt_words)
    if prompt_len < min_prompt_words:
        return None
    remaining = words[prompt_len:]
    if len(remaining) < 10:
        return None
    cont_len = min(len(remaining), cont_words)
    prompt_text = " ".join(words[:prompt_len])
    cont_text = " ".join(remaining[:cont_len])
    return prompt_text, cont_text


def load_eval_entities() -> set[str]:
    """Load entity strings from PopQA and TriviaQA test sets for contamination filtering."""
    entities = set()

    # PopQA
    try:
        logger.info("Loading PopQA for entity filtering...")
        ds = load_dataset("akariasai/PopQA", split="test", streaming=True)
        count = 0
        for row in ds:
            for key in ("question", "possible_answers", "obj"):
                val = row.get(key, "")
                if isinstance(val, str) and len(val) >= 3:
                    entities.add(val.lower())
                elif isinstance(val, list):
                    for v in val:
                        if isinstance(v, str) and len(v) >= 3:
                            entities.add(v.lower())
            count += 1
            if count >= 50000:
                break
        logger.info(f"  PopQA: collected {len(entities)} entity strings")
    except Exception as e:
        logger.warning(f"  Failed to load PopQA: {e}")

    # TriviaQA
    try:
        logger.info("Loading TriviaQA for entity filtering...")
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="test", streaming=True)
        count = 0
        prev_count = len(entities)
        for row in ds:
            q = row.get("question", "")
            if isinstance(q, str) and len(q) >= 3:
                entities.add(q.lower())
            answer = row.get("answer", {})
            if isinstance(answer, dict):
                for key in ("value", "aliases", "normalized_aliases"):
                    val = answer.get(key, "")
                    if isinstance(val, str) and len(val) >= 3:
                        entities.add(val.lower())
                    elif isinstance(val, list):
                        for v in val:
                            if isinstance(v, str) and len(v) >= 3:
                                entities.add(v.lower())
            count += 1
            if count >= 50000:
                break
        logger.info(f"  TriviaQA: added {len(entities) - prev_count} entity strings")
    except Exception as e:
        logger.warning(f"  Failed to load TriviaQA: {e}")

    logger.info(f"Total eval entities for filtering: {len(entities)}")
    return entities


def is_contaminated(text: str, entities: set[str]) -> bool:
    """Check if text contains any eval entity (case-insensitive substring match)."""
    text_lower = text.lower()
    for entity in entities:
        if entity in text_lower:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_prompts", type=int, default=10000)
    parser.add_argument("--min_prompt_words", type=int, default=30)
    parser.add_argument("--max_prompt_words", type=int, default=50)
    parser.add_argument("--cont_words", type=int, default=40)
    parser.add_argument("--min_paragraph_words", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to tokenizer for verifying token counts (optional)")
    parser.add_argument("--skip_entity_filter", action="store_true",
                        help="Skip PopQA/TriviaQA entity filtering (faster, for debugging)")
    args = parser.parse_args()

    # Step 1: Load eval entities for contamination filtering
    if not args.skip_entity_filter:
        entities = load_eval_entities()
    else:
        entities = set()
        logger.info("Skipping entity filtering.")

    # Step 2: Load Wikipedia
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
            first = next(iter(ds))
            logger.info(f"Successfully loaded {ds_name}")
            break
        except Exception as e:
            logger.warning(f"Failed to load {ds_name}: {e}")
            ds = None

    if ds is None:
        raise RuntimeError("Could not load any Wikipedia/wikitext dataset.")

    # Detect text column name
    text_key = "text" if "text" in first else "page" if "page" in first else list(first.keys())[0]
    logger.info(f"Using text column: '{text_key}'")

    # Step 3: Extract (prompt, continuation) pairs
    pairs = []
    seen = 0
    filtered_count = 0
    for article in ds:
        seen += 1
        text = article.get(text_key, "")
        if not text or len(text.strip()) < 100:
            continue
        para = extract_first_paragraph(text, min_words=args.min_paragraph_words)
        if para is None:
            continue
        result = split_prompt_continuation(
            para, args.min_prompt_words, args.max_prompt_words, args.cont_words
        )
        if result is None:
            continue
        prompt_text, cont_text = result

        # Entity contamination check
        if entities and is_contaminated(prompt_text + " " + cont_text, entities):
            filtered_count += 1
            continue

        pairs.append((prompt_text, cont_text))
        if len(pairs) >= args.num_prompts:
            break
        if len(pairs) % 2000 == 0:
            logger.info(f"  collected {len(pairs)}/{args.num_prompts} pairs "
                        f"(scanned {seen} articles, filtered {filtered_count})")

    logger.info(f"Collected {len(pairs)} pairs from {seen} articles "
                f"(entity-filtered: {filtered_count})")

    # Optional: verify token counts
    if args.model_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        prompt_tokens = [len(tokenizer.encode(p)) for p, _ in pairs]
        cont_tokens = [len(tokenizer.encode(c)) for _, c in pairs]
        logger.info(f"Prompt token stats: avg={sum(prompt_tokens)/len(prompt_tokens):.1f}, "
                    f"min={min(prompt_tokens)}, max={max(prompt_tokens)}")
        logger.info(f"Continuation token stats: avg={sum(cont_tokens)/len(cont_tokens):.1f}, "
                    f"min={min(cont_tokens)}, max={max(cont_tokens)}")

    # Save
    out_path = DATA_DIR / "dkp_wiki_10k.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for prompt_text, cont_text in pairs:
            f.write(json.dumps({
                "prompt_text": prompt_text,
                "continuation_text": cont_text,
            }, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(pairs)} (prompt, continuation) pairs to {out_path}")
    logger.info(f"Sample prompt: {pairs[0][0][:100]}...")
    logger.info(f"Sample continuation: {pairs[0][1][:100]}...")


if __name__ == "__main__":
    main()
