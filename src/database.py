"""
Local SQLite persistence for Wisperno: dictation history, custom vocabulary,
voice snippets, and a generic settings key-value store.

Design note: hotkeys/trigger-mode/model-paths/autostart are already persisted
robustly through config.yaml (src/config.py:save_config/load_config, in place
since an earlier round and covered by tests/test_pipeline.py). Duplicating that
into the `settings` table here would create two sources of truth for the same
data, so `settings` is kept as a generic KV store for anything the UI needs
that ISN'T already owned by config.yaml (e.g. last-selected History filter),
rather than re-storing config.yaml's fields.
"""

import datetime
import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from loguru import logger

RETENTION_POLICY_DAYS = {"3_months": 90, "6_months": 182, "1_year": 365, "never": None}

SCHEMA = """
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    duration_seconds REAL NOT NULL DEFAULT 0,
    raw_transcript TEXT NOT NULL DEFAULT '',
    polished_transcript TEXT NOT NULL DEFAULT '',
    mode_used TEXT NOT NULL DEFAULT 'polish',
    word_count INTEGER NOT NULL DEFAULT 0,
    latency_ms REAL NOT NULL DEFAULT 0,
    is_favorite INTEGER NOT NULL DEFAULT 0,
    is_pinned INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'dictation',
    entry_type TEXT NOT NULL DEFAULT 'dictation'
);

CREATE TABLE IF NOT EXISTS dictionary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word TEXT NOT NULL,
    replacement TEXT NOT NULL,
    case_sensitive INTEGER NOT NULL DEFAULT 0,
    category TEXT NOT NULL DEFAULT 'Jargon',
    UNIQUE(word, case_sensitive)
);

CREATE TABLE IF NOT EXISTS snippets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger_phrase TEXT NOT NULL UNIQUE,
    expansion_text TEXT NOT NULL DEFAULT '',
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transforms (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    shortcut TEXT NOT NULL DEFAULT '',
    system_prompt TEXT NOT NULL DEFAULT '',
    is_default INTEGER NOT NULL DEFAULT 0,
    is_active INTEGER NOT NULL DEFAULT 1,
    sort_order INTEGER NOT NULL DEFAULT 0
);
"""

# Seeded once, the first time the transforms table is empty - "Reset to
# defaults" restores exactly this set. Kept as data (not code) so a factory
# reset is a straight re-insert, no special-cased Python branching.
# Phonetic corrections added after config/dictionary.json's original seed list -
# see seed_additional_corrections() below for why these need their own migration.
ADDITIONAL_DEFAULT_CORRECTIONS = [
    ("cloud code", "Claude Code"),
    ("cloud board", "Claude Code"),
    ("claud code", "Claude Code"),
    ("clawed code", "Claude Code"),
    ("screenshot", "screenshot"),  # self-mapped: adds "screenshot" to Whisper's
                                    # own initial_prompt vocabulary bias (see
                                    # Vocabulary.build_initial_prompt) without a
                                    # risky regex correction pass.
]

# Round 3 of the same class of addition - see seed_additional_corrections_v3().
ADDITIONAL_DEFAULT_CORRECTIONS_V3 = [
    ("github", "GitHub"),
    ("pytorch", "PyTorch"),
    ("screen shot", "screenshot"),
    ("screen shit", "screenshot"),
    ("wisper no", "Wisperno"),
    ("whisper no", "Wisperno"),
]

DEFAULT_TRANSFORMS = [
    {
        "id": "polish", "title": "Polish", "shortcut": "alt+c", "sort_order": 0,
        "description": "Improve clarity and conciseness",
        "system_prompt": (
            "You are a precise, deterministic speech-to-text dictation post-processor.\n"
            "Your sole job is to clean, correct, and format the user's raw spoken transcript into natural, polished prose.\n\n"
            "MANDATORY RULES:\n"
            "1. NEVER reply to the content, NEVER answer questions, and NEVER continue the conversation. The input is dictation to be cleaned, not a message directed at you.\n"
            "2. If the user dictates a question, output it cleaned - DO NOT answer it.\n"
            "3. Remove verbal tics, hesitations, stutters, and filler words (\"um\", \"uh\", \"like\", \"you know\").\n"
            "4. Resolve mid-sentence self-corrections BEFORE applying any other rule, including punctuation: find the word/phrase immediately before \"wait no\"/\"actually\"/\"i mean\", DELETE it along with the correction marker itself, and splice in the word/phrase that comes after the marker in its place, producing ONE single clean sentence with no trace of the correction. The words \"wait\" and \"no\" must NEVER appear in your output - they are a silent edit instruction, not spoken content.\n"
            "5. Fix capitalization, punctuation, spelling, and grammar while preserving the speaker's exact tone, intent, and first-person voice.\n"
            "6. Output ONLY the polished text - no markdown, quotes, headers, or commentary.\n"
            "7. The English first-person singular pronoun \"I\" and its contractions (\"I'm\", \"I've\", \"I'll\", \"I'd\") must ALWAYS be capitalized, everywhere in the sentence. Never output a lowercase \"i\" as a standalone word.\n\n"
            "MANDATORY FIDELITY RULES:\n"
            "- NEVER summarize, condense, or omit any thoughts, details, or sentences spoken by the user.\n"
            "- Your output MUST match the length, detail, and semantic completeness of the input speech.\n"
            "- Only remove verbal tics and repair grammatical errors. Keep all distinct clauses intact.\n\n"
            "PUNCTUATION & CLAUSE MANDATES:\n"
            "- Actively insert commas at natural pauses, introductory clauses, compound sentence joins (\"and\", \"but\", \"so\"), and list transitions.\n"
            "- Break long rambling thoughts into clean, well-punctuated sentences using periods and semicolons where appropriate.\n"
            "- Maintain the exact vocabulary and tone, but fix run-on phrasing so the rhythm reads like clear, comma-punctuated written English."
        ),
    },
    {
        "id": "prompt_engineer", "title": "Prompt Engineer", "shortcut": "alt+x", "sort_order": 1,
        "description": "**Title** (1 concise line)...",
        "system_prompt": (
            "You are a deterministic prompt-engineering compiler, not a conversational assistant.\n"
            "The user will dictate a messy, unstructured, spoken idea or task. Convert it into a clean, optimized AI "
            "prompt using EXACTLY this template - keep every section header, and write a short placeholder guidance "
            "line (in parentheses, like the template shows) for any section the dictation gave you nothing to fill in:\n\n"
            "**Title**\n(1 concise line)\n\n"
            "**Role & stance**\n(who the model is and how it should behave)\n\n"
            "**Task**\n(what the model must do)\n\n"
            "**Context**\n(only what the model needs to know)\n\n"
            "**Inputs available**\n(explicit list)\n\n"
            "**Output requirements**\n(format, structure, tone, length - only if specified; otherwise placeholders)\n\n"
            "**Constraints / Do-nots**\n(bulleted)\n\n"
            "**Examples / References**\n(include all examples verbatim)\n\n"
            "**Execution checklist**\n(short, factual verification list)\n\n"
            "**Conflict resolution**\n(only if applicable)\n\n"
            "NEVER answer or fulfill the dictated request yourself - you are formatting a prompt FOR another AI to "
            "answer later. Output ONLY the filled template, no preamble, no markdown fences around the whole thing."
        ),
    },
    {
        "id": "code_fix", "title": "Detailed Code Fix Prompt", "shortcut": "alt+d", "sort_order": 2,
        "description": "Title Universal Dynamic Debugging Prompt for a described bug",
        "system_prompt": (
            "You are a deterministic bug-report compiler, not a conversational assistant.\n"
            "The user will dictate a description of a bug or broken behavior, informally. Convert it into a "
            "structured code-fix request another AI/engineer can act on directly, using this template:\n\n"
            "**Title**\n(1 concise line naming the bug)\n\n"
            "**Symptom**\n(what is observed, in the user's own terms)\n\n"
            "**Expected behavior**\n(what should happen instead)\n\n"
            "**Reproduction context**\n(file/module/feature mentioned, steps if described)\n\n"
            "**Suspected cause**\n(only if the user speculated on one - otherwise omit)\n\n"
            "**Constraints**\n(anything the fix must not break, if mentioned)\n\n"
            "NEVER attempt to diagnose or fix the bug yourself - you are structuring the report FOR someone else to "
            "act on. Output ONLY the filled template, no preamble."
        ),
    },
    {
        "id": "grammar_correct", "title": "Grammar Correct", "shortcut": "alt+g", "sort_order": 3,
        "description": (
            "Strictly repairs spelling, punctuation, capitalization, and grammatical errors without "
            "rewriting your personal tone or changing word choices."
        ),
        "system_prompt": (
            "You are a strict grammatical proofreader. Your only job is to correct spelling mistakes, "
            "punctuation errors, sentence-start capitalization, and grammatical flaws in the provided text.\n"
            "DO NOT rephrase sentences, DO NOT alter the author's tone, and DO NOT replace casual or "
            "colloquial words with formal alternatives.\n"
            "Return ONLY the corrected text. Never add preambles, notes, or quotation marks."
        ),
    },
    {
        "id": "ai_relay", "title": "AI Relay", "shortcut": "alt+r", "sort_order": 4,
        "description": (
            "Light-touch cleanup for dictation headed to another AI chatbot: removes filler words and "
            "fixes grammar, but does NOT restructure sentences or split up your rambling - unlike Polish, "
            "which rewrites for human readability, this keeps your original flow intact since a downstream "
            "AI benefits from more of your raw phrasing, not less."
        ),
        "system_prompt": (
            "You are a precise, deterministic speech-to-text dictation post-processor. Your output will be "
            "pasted directly into another AI chatbot as the user's message - it is NOT the final human-facing "
            "text, so preserving the user's raw train of thought matters more than polished prose.\n\n"
            "MANDATORY RULES:\n"
            "1. NEVER reply to the content, NEVER answer questions, and NEVER continue the conversation. The "
            "input is dictation to be cleaned, not a message directed at you.\n"
            "2. If the user dictates a question, output it cleaned - DO NOT answer it.\n"
            "3. Remove verbal tics, hesitations, stutters, and filler words (\"um\", \"uh\", \"like\", \"you "
            "know\", \"ah\", \"I mean\", \"sort of\", \"kind of\").\n"
            "4. Resolve mid-sentence self-corrections BEFORE applying any other rule: find the word/phrase "
            "immediately before \"wait no\"/\"actually\"/\"i mean\", DELETE it along with the correction "
            "marker itself, and splice in the word/phrase that comes after the marker in its place. The words "
            "\"wait\" and \"no\" must NEVER appear in your output as a standalone correction marker.\n"
            "5. Fix objectively wrong spelling and grammar (e.g. subject-verb agreement, wrong tense, missing "
            "words) while preserving the speaker's exact tone, vocabulary, and first-person voice.\n"
            "6. NEVER censor, bleep, or soften profanity or swear words the user actually said.\n"
            "7. The English first-person singular pronoun \"I\" and its contractions (\"I'm\", \"I've\", "
            "\"I'll\", \"I'd\") must ALWAYS be capitalized, everywhere in the sentence.\n"
            "8. Output ONLY the cleaned text - no markdown, quotes, headers, or commentary.\n\n"
            "MANDATORY FIDELITY RULES:\n"
            "- NEVER summarize, condense, or omit any thoughts, details, or sentences spoken by the user.\n"
            "- Your output MUST match the length, detail, and semantic completeness of the input speech.\n\n"
            "DO NOT RESTRUCTURE (this is what makes this mode different from Polish):\n"
            "- Do NOT split run-on or rambling sentences into multiple shorter sentences.\n"
            "- Do NOT reorder clauses or reorganize the sequence of ideas.\n"
            "- Do NOT insert extra commas or semicolons beyond what's needed to fix an objective grammar "
            "error - a long, comma-light, rambling structure is fine and should be left as spoken.\n"
            "- Only add the minimum punctuation needed for the sentence to be grammatically valid (e.g. a "
            "final period, a question mark on a real question) - do not add polish-level punctuation."
        ),
    },
]


# Seeded once, the first time the snippets table is empty - realistic voice
# macros so a fresh install demonstrates the feature instead of showing an
# empty grid. "insert email" deliberately stays a generic placeholder (not
# this dev machine's real address): these ship to every install, not just this one.
DEFAULT_SNIPPETS = [
    {"trigger_phrase": "insert email", "expansion_text": "yourname@example.com"},
    {"trigger_phrase": "meeting link", "expansion_text": "https://meet.google.com/abc-defg-hij"},
    {
        "trigger_phrase": "daily standup",
        "expansion_text": "Today: Working on core features.\nBlockers: None.\nNext: Verification & tests.",
    },
]


class WispernoDB:
    """Thread-safe SQLite access (one lock guards all writes - dictation volume never justifies more)."""

    def __init__(self, db_path: Union[str, Path]):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL;")  # crash-safe, concurrent-read-friendly for a background daemon
        self._migrate()

    def _migrate(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(SCHEMA)
            # CREATE TABLE IF NOT EXISTS is a no-op on an already-existing history
            # table, so a column added after a table shipped needs its own
            # explicit ALTER TABLE - guarded since SQLite has no "ADD COLUMN IF
            # NOT EXISTS" and re-running this must stay a no-op on later launches.
            existing_cols = {row[1] for row in self._conn.execute("PRAGMA table_info(history)")}
            if "is_pinned" not in existing_cols:
                self._conn.execute("ALTER TABLE history ADD COLUMN is_pinned INTEGER NOT NULL DEFAULT 0")
            if "source" not in existing_cols:
                self._conn.execute("ALTER TABLE history ADD COLUMN source TEXT NOT NULL DEFAULT 'dictation'")
            # entry_type separates WHAT KIND of entry this is ('dictation' vs
            # 'live_transcript', src/live_transcriber.py) - orthogonal to
            # `source`, which separates HOW a 'dictation'-shaped entry
            # happened (voice / transform / selection-polish). A live session
            # is always entry_type='live_transcript', source='dictation' (it
            # came from voice, not a text-selection polish or a Transforms-hub
            # run - see add_live_transcript()).
            if "entry_type" not in existing_cols:
                self._conn.execute("ALTER TABLE history ADD COLUMN entry_type TEXT NOT NULL DEFAULT 'dictation'")

            transform_count = self._conn.execute("SELECT COUNT(*) FROM transforms").fetchone()[0]
            if transform_count == 0:
                self._insert_default_transforms(self._conn)

            snippet_count = self._conn.execute("SELECT COUNT(*) FROM snippets").fetchone()[0]
            if snippet_count == 0:
                self._insert_default_snippets(self._conn)

            # One-time migration for a DB created by an earlier round, when the
            # 3 default snippets were seeded active (is_active=1) - they were
            # live voice macros hijacking real speech containing their trigger
            # phrases (e.g. actually saying "meeting link" mid-sentence). Only
            # runs once (guarded via the settings KV store) so a user who
            # deliberately re-enables one of these later doesn't get silently
            # overridden on the next launch.
            if not self._conn.execute(
                "SELECT 1 FROM settings WHERE key = 'default_snippets_deactivated'"
            ).fetchone():
                self._conn.execute(
                    "UPDATE snippets SET is_active = 0 WHERE trigger_phrase IN "
                    "('insert email', 'meeting link', 'daily standup')"
                )
                self._conn.execute(
                    "INSERT OR REPLACE INTO settings (key, value) VALUES ('default_snippets_deactivated', 'true')"
                )
        logger.info(f"Database ready at '{self.db_path}'.")

    @staticmethod
    def _insert_default_transforms(conn: sqlite3.Connection) -> None:
        for t in DEFAULT_TRANSFORMS:
            conn.execute(
                "INSERT OR REPLACE INTO transforms "
                "(id, title, description, shortcut, system_prompt, is_default, is_active, sort_order) "
                "VALUES (?,?,?,?,?,1,1,?)",
                (t["id"], t["title"], t["description"], t["shortcut"], t["system_prompt"], t["sort_order"]),
            )

    @staticmethod
    def _insert_default_snippets(conn: sqlite3.Connection) -> None:
        """Seeded inactive (is_active=0): these are example templates to
        customize, not live voice macros - active by default meant literally
        saying "meeting link" or "daily standup" mid-sentence got silently
        replaced by the canned expansion text instead of being dictated."""
        for s in DEFAULT_SNIPPETS:
            conn.execute(
                "INSERT OR REPLACE INTO snippets (trigger_phrase, expansion_text, is_active) VALUES (?,?,0)",
                (s["trigger_phrase"], s["expansion_text"]),
            )

    @contextmanager
    def _cursor(self):
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

    # --- History -------------------------------------------------------

    def add_history(
        self, timestamp: str, duration_seconds: float, raw_transcript: str,
        polished_transcript: str, mode_used: str, latency_ms: float,
        source: str = "dictation", entry_type: str = "dictation",
    ) -> int:
        """`source` drives History's badge: 'dictation' (voice, no transform),
        'transform' (a Transforms-hub hotkey drove fresh dictation), or
        'text_polish' (an existing selection was polished in place).
        `entry_type` is the orthogonal "kind of entry" axis used by History's
        top-level filter toggles: 'dictation' (a normal quick utterance) or
        'live_transcript' (a Live Transcription session - see
        add_live_transcript(), the usual way those get inserted)."""
        word_count = len(polished_transcript.split())
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO history (timestamp, duration_seconds, raw_transcript, "
                "polished_transcript, mode_used, word_count, latency_ms, source, entry_type) VALUES (?,?,?,?,?,?,?,?,?)",
                (timestamp, duration_seconds, raw_transcript, polished_transcript, mode_used, word_count, latency_ms, source, entry_type),
            )
            return cur.lastrowid

    def add_live_transcript(
        self, timestamp: str, duration_seconds: float, raw_transcript: str, polished_transcript: str,
    ) -> int:
        """Post-session insert for a completed Live Transcription session
        (src/live_transcriber.py) - both the raw and the lightly-cleaned
        transcript are stored (see LiveTranscriptionWorker.clean_live_transcript()),
        `mode_used='live'` names the mode for History's badge/label lookup,
        latency_ms=0.0 since there is no single per-utterance latency figure
        for a multi-minute streamed session."""
        return self.add_history(
            timestamp=timestamp, duration_seconds=duration_seconds, raw_transcript=raw_transcript,
            polished_transcript=polished_transcript, mode_used="live", latency_ms=0.0,
            source="dictation", entry_type="live_transcript",
        )

    def list_history(
        self, search: Optional[str] = None, limit: int = 200, pinned_only: bool = False,
        entry_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """`entry_type`: None returns both kinds ("All"), or filter to exactly
        'dictation' / 'live_transcript' (History's segmented filter bar)."""
        with self._cursor() as cur:
            clauses, params = [], []
            if search:
                clauses.append("(raw_transcript LIKE ? OR polished_transcript LIKE ?)")
                params += [f"%{search}%", f"%{search}%"]
            if pinned_only:
                clauses.append("is_pinned = 1")
            if entry_type:
                clauses.append("entry_type = ?")
                params.append(entry_type)
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            cur.execute(
                f"SELECT * FROM history {where} ORDER BY is_pinned DESC, timestamp DESC, id DESC LIMIT ?",
                (*params, limit),
            )
            return [dict(row) for row in cur.fetchall()]

    def toggle_favorite(self, history_id: int) -> None:
        with self._cursor() as cur:
            cur.execute("UPDATE history SET is_favorite = 1 - is_favorite WHERE id = ?", (history_id,))

    def toggle_pin(self, history_id: int) -> None:
        with self._cursor() as cur:
            cur.execute("UPDATE history SET is_pinned = 1 - is_pinned WHERE id = ?", (history_id,))

    def delete_history(self, history_id: int) -> None:
        with self._cursor() as cur:
            cur.execute("DELETE FROM history WHERE id = ?", (history_id,))

    def prune_history(self, retention_policy: str) -> int:
        """Delete unpinned rows older than the policy window. Returns rows deleted."""
        days = RETENTION_POLICY_DAYS.get(retention_policy)
        if days is None:  # "never" - retention disabled
            return 0
        cutoff = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
        with self._cursor() as cur:
            cur.execute("DELETE FROM history WHERE is_pinned = 0 AND timestamp < ?", (cutoff,))
            return cur.rowcount

    def clear_oldest_transcripts(self, fraction: float = 0.25) -> int:
        """Delete the oldest `fraction` of unpinned rows (e.g. 0.25 = oldest 25%). Returns rows deleted."""
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM history WHERE is_pinned = 0")
            unpinned_count = cur.fetchone()[0]
            n_to_delete = int(unpinned_count * fraction) if fraction < 1.0 else unpinned_count
            if n_to_delete <= 0:
                return 0
            cur.execute(
                "DELETE FROM history WHERE id IN ("
                "  SELECT id FROM history WHERE is_pinned = 0 ORDER BY timestamp ASC, id ASC LIMIT ?"
                ")",
                (n_to_delete,),
            )
            return cur.rowcount

    def get_database_storage_stats(self) -> Dict[str, Any]:
        """File size (formatted) and record count, for the Settings/Advanced disk-usage display."""
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM history")
            record_count = cur.fetchone()[0]
        size_bytes = Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0
        # WAL mode keeps recent writes in a sidecar file until checkpointed - include it in the reported size.
        for suffix in ("-wal", "-shm"):
            sidecar = Path(self.db_path + suffix)
            if sidecar.exists():
                size_bytes += sidecar.stat().st_size
        return {
            "size_mb_formatted": f"{size_bytes / (1024 * 1024):.1f} MB",
            "size_bytes": size_bytes,
            "record_count": record_count,
        }

    def get_stats(self) -> Dict[str, Any]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) as n, COALESCE(SUM(word_count),0) as words, "
                "COALESCE(SUM(duration_seconds),0) as secs FROM history"
            )
            row = cur.fetchone()
        n, words, secs = row["n"], row["words"], row["secs"]
        avg_wpm = (words / (secs / 60.0)) if secs > 0 else 0.0
        return {
            "total_dictations": n,
            "total_words": words,
            "total_duration_sec": secs,
            "hours_saved": round(words / 40.0 / 60.0, 2),  # ~40 wpm typing baseline
            "avg_wpm": round(avg_wpm, 1),
        }

    # --- Dictionary ------------------------------------------------------

    def list_dictionary(self) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM dictionary ORDER BY word COLLATE NOCASE")
            return [dict(row) for row in cur.fetchall()]

    def add_dictionary_entry(self, word: str, replacement: str, case_sensitive: bool = False, category: str = "Jargon") -> int:
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO dictionary (word, replacement, case_sensitive, category) VALUES (?,?,?,?)",
                (word.strip(), replacement.strip(), int(case_sensitive), category),
            )
            return cur.lastrowid

    def delete_dictionary_entry(self, entry_id: int) -> None:
        with self._cursor() as cur:
            cur.execute("DELETE FROM dictionary WHERE id = ?", (entry_id,))

    def free_up_alt_v_shortcut(self) -> bool:
        """One-time migration: 'alt+v' was the default 'code_fix' transform's
        shortcut, but it's also the new global Writing Styles hotkey - two
        different actions bound to the same physical chord would race. Moves
        code_fix to 'alt+d' ONLY if it's still sitting on the untouched
        default (a user who already rebound it themselves is left alone -
        their custom binding no longer conflicts with alt+v anyway)."""
        flag = "alt_v_freed_for_writing_styles"
        if self.get_setting(flag):
            return False
        row = self.get_transform("code_fix")
        if row and row["shortcut"] == "alt+v":
            self.update_transform("code_fix", shortcut="alt+d")
            logger.info("Reassigned 'Detailed Code Fix Prompt' from Alt+V to Alt+D (Alt+V is now Writing Styles).")
        self.set_setting(flag, "true")
        return True

    def seed_additional_corrections(self) -> int:
        """One-time seed of phonetic corrections added after config/dictionary.json's
        original list - existing installs already have dictionary_json_imported=true
        (migrate_dictionary_json above is one-time-only) and would otherwise never
        receive these. Its own flag, gated separately, so it reaches existing users
        without re-running the full legacy import."""
        flag = "dictionary_v2_seeded"
        if self.get_setting(flag):
            return 0
        count = 0
        for word, replacement in ADDITIONAL_DEFAULT_CORRECTIONS:
            self.add_dictionary_entry(word, replacement)
            count += 1
        self.set_setting(flag, "true")
        logger.info(f"Seeded {count} additional default dictionary corrections.")
        return count

    def seed_grammar_correct_transform(self) -> bool:
        """DEFAULT_TRANSFORMS is only inserted for a brand-new database (see
        _migrate()'s transform_count == 0 guard) - an existing install's
        transforms table already has rows and never re-seeds from that list,
        so a transform added there after the fact (this one) needs its own
        one-time insert to reach existing users. Gated on the row's own
        existence rather than a settings flag, since a user who deletes this
        transform on purpose has an equally clear signal not to re-add it -
        checking existence covers both "never had it" and "explicitly removed
        it" as the one case that's actually ambiguous is not this one."""
        if self.get_transform("grammar_correct") is not None:
            return False
        entry = next(t for t in DEFAULT_TRANSFORMS if t["id"] == "grammar_correct")
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO transforms "
                "(id, title, description, shortcut, system_prompt, is_default, is_active, sort_order) "
                "VALUES (?,?,?,?,?,1,1,?)",
                (entry["id"], entry["title"], entry["description"], entry["shortcut"],
                 entry["system_prompt"], entry["sort_order"]),
            )
        logger.info("Seeded the 'Grammar Correct' transform (Alt+G) for an existing install.")
        return True

    def seed_ai_relay_transform(self) -> bool:
        """Same one-time-insert-for-existing-installs pattern as
        seed_grammar_correct_transform() above - see that method's docstring
        for why existence-check is the right gate here too."""
        if self.get_transform("ai_relay") is not None:
            return False
        entry = next(t for t in DEFAULT_TRANSFORMS if t["id"] == "ai_relay")
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO transforms "
                "(id, title, description, shortcut, system_prompt, is_default, is_active, sort_order) "
                "VALUES (?,?,?,?,?,1,1,?)",
                (entry["id"], entry["title"], entry["description"], entry["shortcut"],
                 entry["system_prompt"], entry["sort_order"]),
            )
        logger.info("Seeded the 'AI Relay' transform (Alt+R) for an existing install.")
        return True

    def seed_additional_corrections_v3(self) -> int:
        """Same reasoning as seed_additional_corrections() above, own flag -
        installs that already got the v2 seed still need these new ones."""
        flag = "dictionary_v3_seeded"
        if self.get_setting(flag):
            return 0
        count = 0
        for word, replacement in ADDITIONAL_DEFAULT_CORRECTIONS_V3:
            self.add_dictionary_entry(word, replacement)
            count += 1
        self.set_setting(flag, "true")
        logger.info(f"Seeded {count} additional (v3) default dictionary corrections.")
        return count

    def migrate_dictionary_json(self, json_path: Union[str, Path]) -> int:
        """One-time import of the legacy config/dictionary.json into the DB. Idempotent."""
        path = Path(json_path)
        if not path.exists() or self.get_setting("dictionary_json_imported"):
            return 0
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Could not read legacy dictionary '{path}': {e}")
            return 0
        count = 0
        for item in entries:
            find, replace = item.get("find"), item.get("replace")
            if find and replace:
                self.add_dictionary_entry(find, replace)
                count += 1
        self.set_setting("dictionary_json_imported", "true")
        logger.info(f"Migrated {count} dictionary entries from '{path}' into the database.")
        return count

    # --- Snippets ----------------------------------------------------------

    def list_snippets(self, active_only: bool = False) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            if active_only:
                cur.execute("SELECT * FROM snippets WHERE is_active = 1")
            else:
                cur.execute("SELECT * FROM snippets ORDER BY trigger_phrase COLLATE NOCASE")
            return [dict(row) for row in cur.fetchall()]

    def add_snippet(self, trigger_phrase: str, expansion_text: str, is_active: bool = True) -> int:
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO snippets (trigger_phrase, expansion_text, is_active) VALUES (?,?,?)",
                (trigger_phrase.strip().lower(), expansion_text, int(is_active)),
            )
            return cur.lastrowid

    def delete_snippet(self, snippet_id: int) -> None:
        with self._cursor() as cur:
            cur.execute("DELETE FROM snippets WHERE id = ?", (snippet_id,))

    def set_snippet_active(self, snippet_id: int, is_active: bool) -> None:
        with self._cursor() as cur:
            cur.execute("UPDATE snippets SET is_active = ? WHERE id = ?", (int(is_active), snippet_id))

    # --- Generic settings KV -------------------------------------------------

    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._cursor() as cur:
            cur.execute("SELECT value FROM settings WHERE key = ?", (key,))
            row = cur.fetchone()
            return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self._cursor() as cur:
            cur.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?,?)", (key, value))

    # --- Transforms (Wispr Flow "Transforms" hub) -----------------------------

    def list_transforms(self, active_only: bool = False) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            if active_only:
                cur.execute("SELECT * FROM transforms WHERE is_active = 1 ORDER BY sort_order, title COLLATE NOCASE")
            else:
                cur.execute("SELECT * FROM transforms ORDER BY sort_order, title COLLATE NOCASE")
            return [dict(row) for row in cur.fetchall()]

    def get_transform(self, transform_id: str) -> Optional[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM transforms WHERE id = ?", (transform_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def add_transform(self, title: str, description: str, shortcut: str, system_prompt: str) -> str:
        transform_id = title.strip().lower().replace(" ", "_") or "custom"
        base_id, n = transform_id, 1
        while self.get_transform(transform_id) is not None:
            n += 1
            transform_id = f"{base_id}_{n}"
        with self._cursor() as cur:
            cur.execute(
                "SELECT COALESCE(MAX(sort_order), -1) + 1 FROM transforms"
            )
            next_order = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO transforms (id, title, description, shortcut, system_prompt, is_default, is_active, sort_order) "
                "VALUES (?,?,?,?,?,0,1,?)",
                (transform_id, title.strip(), description.strip(), shortcut.strip().lower(), system_prompt, next_order),
            )
        return transform_id

    def update_transform(
        self, transform_id: str, title: Optional[str] = None, description: Optional[str] = None,
        shortcut: Optional[str] = None, system_prompt: Optional[str] = None, is_active: Optional[bool] = None,
    ) -> None:
        fields, params = [], []
        for col, val in (("title", title), ("description", description), ("shortcut", shortcut),
                          ("system_prompt", system_prompt)):
            if val is not None:
                fields.append(f"{col} = ?")
                params.append(val)
        if is_active is not None:
            fields.append("is_active = ?")
            params.append(int(is_active))
        if not fields:
            return
        params.append(transform_id)
        with self._cursor() as cur:
            cur.execute(f"UPDATE transforms SET {', '.join(fields)} WHERE id = ?", params)

    def delete_transform(self, transform_id: str) -> None:
        with self._cursor() as cur:
            cur.execute("DELETE FROM transforms WHERE id = ? AND is_default = 0", (transform_id,))

    def reset_transforms_to_defaults(self) -> None:
        with self._cursor() as cur:
            cur.execute("DELETE FROM transforms")
            self._insert_default_transforms(self._conn)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _demo() -> None:
    """Self-check against a throwaway in-memory-like temp DB file."""
    import tempfile
    import os

    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    try:
        db = WispernoDB(path)

        hid = db.add_history("2026-01-01T00:00:00", 3.5, "um hello", "Hello.", "polish", 420.0)
        assert hid > 0
        rows = db.list_history()
        assert len(rows) == 1 and rows[0]["polished_transcript"] == "Hello."
        assert rows[0]["source"] == "dictation", f"Expected default source 'dictation', got {rows[0]['source']!r}"
        assert rows[0]["entry_type"] == "dictation", f"Expected default entry_type 'dictation', got {rows[0]['entry_type']!r}"
        db.toggle_favorite(hid)
        assert db.list_history()[0]["is_favorite"] == 1

        live_id = db.add_live_transcript("2026-01-01T00:02:00", 245.0, "raw live text", "Cleaned live text.")
        assert live_id > 0
        live_row = next(r for r in db.list_history() if r["id"] == live_id)
        assert live_row["entry_type"] == "live_transcript" and live_row["source"] == "dictation"
        assert live_row["mode_used"] == "live"
        assert [r["id"] for r in db.list_history(entry_type="live_transcript")] == [live_id]
        assert hid not in [r["id"] for r in db.list_history(entry_type="live_transcript")]
        assert live_id not in [r["id"] for r in db.list_history(entry_type="dictation")]
        assert len(db.list_history()) == 2, "entry_type=None ('All') must still return every row"
        db.delete_history(live_id)

        transform_hid = db.add_history(
            "2026-01-01T00:01:00", 2.0, "raw", "polished", "prompt_engineer", 300.0, source="transform"
        )
        assert db.list_history(limit=1)[0]["source"] == "transform"
        db.delete_history(transform_hid)

        stats = db.get_stats()
        assert stats["total_dictations"] == 1 and stats["total_words"] == 1

        did = db.add_dictionary_entry("pie torch", "PyTorch")
        assert did > 0
        assert any(d["replacement"] == "PyTorch" for d in db.list_dictionary())
        db.delete_dictionary_entry(did)
        assert not db.list_dictionary()

        seeded_count = db.seed_additional_corrections()
        assert seeded_count == len(ADDITIONAL_DEFAULT_CORRECTIONS)
        assert any(d["word"] == "cloud code" and d["replacement"] == "Claude Code" for d in db.list_dictionary())
        assert db.seed_additional_corrections() == 0, "Must not re-seed once the flag is set"

        seeded_v3 = db.seed_additional_corrections_v3()
        assert seeded_v3 == len(ADDITIONAL_DEFAULT_CORRECTIONS_V3)
        assert any(d["word"] == "pytorch" and d["replacement"] == "PyTorch" for d in db.list_dictionary())
        assert db.seed_additional_corrections_v3() == 0, "Must not re-seed once the v3 flag is set"

        seeded = db.list_snippets()
        assert {s["trigger_phrase"] for s in seeded} == {"insert email", "meeting link", "daily standup"}, (
            f"Expected the 3 default snippets seeded on first boot, got {[s['trigger_phrase'] for s in seeded]}"
        )
        assert all(s["is_active"] == 0 for s in seeded), (
            "Default snippet examples must be seeded INACTIVE - active by default means live speech "
            "containing their trigger phrase (e.g. actually saying 'meeting link') gets silently hijacked."
        )
        assert db.list_snippets(active_only=True) == [], "No snippet should be active on a fresh database."

        sid = db.add_snippet("my email", "dev@example.com")
        assert any(s["trigger_phrase"] == "my email" for s in db.list_snippets(active_only=True))
        db.delete_snippet(sid)

        db.set_setting("last_filter", "favorites")
        assert db.get_setting("last_filter") == "favorites"
        assert db.get_setting("missing_key", "fallback") == "fallback"

        defaults = db.list_transforms()
        assert {t["id"] for t in defaults} == {"polish", "prompt_engineer", "code_fix"}
        assert all(t["is_default"] == 1 for t in defaults)

        cid = db.add_transform("My Custom", "test transform", "alt+z", "You are a test.")
        assert db.get_transform(cid)["title"] == "My Custom"
        db.update_transform(cid, description="updated")
        assert db.get_transform(cid)["description"] == "updated"
        db.delete_transform(cid)
        assert db.get_transform(cid) is None
        db.delete_transform("polish")  # is_default guard: must NOT delete
        assert db.get_transform("polish") is not None

        db.update_transform("polish", system_prompt="mutated")
        db.reset_transforms_to_defaults()
        assert db.get_transform("polish")["system_prompt"] != "mutated"

        db.close()
        print("PASS: database schema, history/dictionary/snippets/settings/transforms CRUD, and stats all verified.")
    finally:
        if os.path.exists(path):
            os.unlink(path)


if __name__ == "__main__":
    _demo()
