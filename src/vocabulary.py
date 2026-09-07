"""
Custom Vocabulary & Word-Replacement Engine for Wisperno.
Boosts transcription accuracy two ways:
  1. Whisper keyword biasing: canonical dictionary terms are appended to the
     `initial_prompt` so Whisper is more likely to transcribe them correctly.
  2. Phonetic/custom post-processing: raw transcripts are regex-corrected
     against a user dictionary (e.g. "pie torch" -> "PyTorch") before being
     handed to the LLM.

Backed by src/database.py (SQLite) rather than a flat file, so the Dictionary
tab's add/edit/remove is immediately durable and immediately reflected here.
"""

import re
from typing import TYPE_CHECKING, Dict, List, Optional
from loguru import logger

if TYPE_CHECKING:
    from src.database import WispernoDB


class Vocabulary:
    """Loads dictionary entries from the database and applies keyword biasing + replacements."""

    def __init__(self, db: Optional["WispernoDB"] = None):
        self.db = db
        self.entries: List[Dict] = []  # rows: id, word, replacement, case_sensitive, category
        self.reload()

    def reload(self) -> None:
        """(Re)load dictionary entries from the database."""
        if not self.db:
            self.entries = []
            return
        rows = self.db.list_dictionary()
        # Longest phrase first, so multi-word entries match before shorter overlapping ones.
        rows.sort(key=lambda r: len(r["word"]), reverse=True)
        self.entries = rows
        logger.info(f"Vocabulary: loaded {len(self.entries)} dictionary entries from the database.")

    def add_entry(self, word: str, replacement: str, case_sensitive: bool = False, category: str = "Jargon") -> None:
        word, replacement = word.strip(), replacement.strip()
        if not self.db or not word or not replacement:
            return
        self.db.add_dictionary_entry(word, replacement, case_sensitive, category)
        self.reload()

    def remove_entry(self, entry_id: int) -> None:
        if not self.db:
            return
        self.db.delete_dictionary_entry(entry_id)
        self.reload()

    def apply_replacements(self, text: str) -> str:
        """Whole-word replacement of dictionary phrases in `text` (case-insensitive unless flagged)."""
        if not text or not self.entries:
            return text
        for entry in self.entries:
            flags = 0 if entry["case_sensitive"] else re.IGNORECASE
            pattern = r"(?<!\w)" + re.escape(entry["word"]) + r"(?!\w)"
            text = re.sub(pattern, entry["replacement"], text, flags=flags)
        return text

    def build_initial_prompt(self, base_prompt: str) -> str:
        """Append canonical dictionary terms to Whisper's initial_prompt for keyword biasing."""
        if not self.entries:
            return base_prompt
        terms = []
        for entry in self.entries:
            if entry["replacement"] not in terms:
                terms.append(entry["replacement"])
        return f"{base_prompt} Technical terms: {', '.join(terms)}."


def _demo() -> None:
    """Self-check against a throwaway temp database - no JSON file involved."""
    import os
    import tempfile
    from src.database import WispernoDB

    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    try:
        db = WispernoDB(path)
        db.add_dictionary_entry("pie torch", "PyTorch")
        db.add_dictionary_entry("git hub", "GitHub")
        db.add_dictionary_entry("json", "JSON")

        vocab = Vocabulary(db=db)
        text = "I was using pie torch and git hub to parse some json data"
        result = vocab.apply_replacements(text)
        assert result == "I was using PyTorch and GitHub to parse some JSON data", result

        # Word-boundary safety: "json" inside "jsonify" must NOT be replaced.
        assert vocab.apply_replacements("jsonify this") == "jsonify this"

        prompt = vocab.build_initial_prompt("Clean transcription.")
        assert "PyTorch" in prompt and "GitHub" in prompt and "JSON" in prompt

        vocab.add_entry("cuda", "CUDA")
        assert any(e["word"] == "cuda" for e in vocab.entries)
        cuda_id = next(e["id"] for e in vocab.entries if e["word"] == "cuda")
        vocab.remove_entry(cuda_id)
        assert all(e["word"] != "cuda" for e in vocab.entries)

        db.close()
        print("PASS: vocabulary replacement, word-boundary safety, biasing prompt, and add/remove all verified.")
    finally:
        if os.path.exists(path):
            os.unlink(path)


if __name__ == "__main__":
    _demo()
