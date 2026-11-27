"""Build extra Shakespeare training text from the Project Gutenberg Complete Works.

Source: Project Gutenberg ebook 100, "The Complete Works of William Shakespeare"
(https://www.gutenberg.org/ebooks/100, plain text UTF-8, public domain in the USA).
It is downloaded to data/gutenberg/pg100.txt from an official mirror when missing; the
SHA-256 below pins the exact file used.

The plays are converted to the layout of data/shakespeare.txt (speaker label with a colon,
speech lines, blank line) and restricted to its 65-character vocabulary. Left out:
  * the nine plays already in data/shakespeare.txt, so no validation text enters training
    (the validation split is the end of The Taming of the Shrew and the start of The Tempest);
  * Hamlet, so the soliloquy used as the generation prompt stays unseen;
  * the sonnets and narrative poems, which are not dialogue.

Usage: python experiments/prepare_extended_data.py          writes data/shakespeare_extra.txt
"""
import hashlib
import re
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # core modules are in the repository root
SOURCE = ROOT / "data/gutenberg/pg100.txt"
SOURCE_SHA256 = "a023115c2d4e2ee12221bdd780fdf2ac5a864fe225948656f51f8be462c7fffb"
OUTPUT = ROOT / "data/shakespeare_extra.txt"
ORIGINAL = ROOT / "data/shakespeare.txt"
MIRRORS = ("https://aleph.pglaf.org/1/0/100/100-0.txt", "https://gutenberg.pglaf.org/1/0/100/100-0.txt")

ALREADY_IN_ORIGINAL = {
    "THE TRAGEDY OF CORIOLANUS", "THE THIRD PART OF KING HENRY THE SIXTH", "MEASURE FOR MEASURE",
    "KING RICHARD THE SECOND", "KING RICHARD THE THIRD", "THE TRAGEDY OF ROMEO AND JULIET",
    "THE TAMING OF THE SHREW", "THE TEMPEST", "THE WINTER’S TALE",
}
PROMPT_PLAY = {"THE TRAGEDY OF HAMLET, PRINCE OF DENMARK"}
NOT_DIALOGUE = {"THE SONNETS", "A LOVER’S COMPLAINT", "THE PASSIONATE PILGRIM",
                "THE PHOENIX AND THE TURTLE", "THE RAPE OF LUCRECE", "VENUS AND ADONIS"}

LABEL = re.compile(r"^[A-Z][A-Z’ \-]*\.$")
DIRECTION = re.compile(r"^(Enter|Re-enter|Exit|Exeunt|Flourish|Alarum|Alarums|Sennet|Trumpets?|Music|Thunder|"
                       r"A cry|Drum|Retreat|Excursions|Tucket|Cornets|Hautboys|Dies|Sound|Shout|Noise)\b")
HEADING = re.compile(r"^(ACT [IVX]+|SCENE [IVX]+\.|SCENE\.|PROLOGUE|EPILOGUE|INDUCTION|THE END)")
ORDINALS = {"FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH", "SIXTH", "ALL", "BOTH"}
GENERIC = {"SERVANT", "MESSENGER", "GENTLEMAN", "LORD", "SOLDIER", "CAPTAIN", "OFFICER", "CLOWN", "NURSE",
           "PAGE", "PORTER", "HERALD", "BOY", "PROVOST", "SHEPHERD", "GAOLER", "CITIZEN", "WATCH", "PRIEST",
           "DOCTOR", "SAILOR", "MUSICIAN", "CHORUS", "ATTENDANT", "HOSTESS", "HOST", "LADY", "KNIGHT"}
REPLACE = {"’": "'", "‘": "'", "“": "", "”": "", "—": "--", "…": "...", "æ": "ae", "Æ": "AE", "œ": "oe",
           "À": "A", "Ç": "C", "É": "E", "à": "a", "â": "a", "ç": "c", "è": "e", "é": "e", "ê": "e",
           "ë": "e", "î": "i", "_": ""}


def clean(text, allowed):
    """Map typography to the original character set and drop anything outside it."""
    for old, new in REPLACE.items():
        text = text.replace(old, new)
    text = "".join(c for c in text if c in allowed)
    return re.sub(r" {2,}", " ", text).strip()


def label_text(raw):
    """HAMLET. -> HAMLET:   FIRST CLOWN. -> First Clown:   (as in the original file)."""
    name = raw[:-1].replace("’", "'")
    words = name.split()
    if words[0] in ORDINALS or name in GENERIC:
        name = name.title()
    return name + ":"


def convert_play(body, allowed):
    """Turn one play into 'LABEL:\\nspeech lines\\n\\n' blocks; drop headings and stage directions."""
    start = re.search(r"(?m)^SCENE I\.", body)          # the contents list uses ' Scene I.' instead
    if not start:
        return []
    text = re.sub(r"\[[^\]]*\]", "", body[start.start():])   # bracketed stage directions, also multi-line
    speeches, current = [], None
    for block in re.split(r"\n\s*\n", text):
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if not lines or HEADING.match(lines[0]) or DIRECTION.match(lines[0]):
            continue
        if LABEL.match(lines[0]):
            current = [label_text(lines[0])]
            speeches.append(current)
            lines = lines[1:]
        if current is None:
            continue
        current.extend(cleaned for cleaned in (clean(line, allowed) for line in lines) if cleaned)
    return ["\n".join(s) for s in speeches if len(s) > 1]


def split_works(body):
    """Yield (title, text) using the titles of the contents list as work boundaries."""
    listing = body[:body.index("THE SONNETS", body.index("THE SONNETS") + 20)]
    titles = [line.strip() for line in listing.split("\n") if line.startswith("    ") and line.strip().isupper()]
    positions = [(body.index("\n" + title + "\n", len(listing) - 20) + 1, title) for title in titles]
    for (start, title), (end, _) in zip(positions, positions[1:] + [(len(body), None)]):
        yield title, body[start:end]


def download_source():
    """Fetch ebook 100 when it is missing; only a copy matching the pinned SHA-256 is kept."""
    if SOURCE.exists():
        return
    SOURCE.parent.mkdir(parents=True, exist_ok=True)
    for url in MIRRORS:
        try:
            with urlopen(Request(url, headers={"User-Agent": "python-urllib"}), timeout=900) as response:
                raw = response.read()
        except OSError:
            continue
        if hashlib.sha256(raw).hexdigest() == SOURCE_SHA256:
            SOURCE.write_bytes(raw)
            return
    raise SystemExit(f"Could not download the pinned Project Gutenberg file; place it at {SOURCE}")


def main():
    from data import download_shakespeare
    download_shakespeare(ORIGINAL)
    download_source()
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise SystemExit(f"{SOURCE} differs from the pinned Project Gutenberg file")
    text = raw.decode("utf-8").replace("\r\n", "\n")
    body = text[text.index("*** START OF"):text.index("*** END OF")]
    original = ORIGINAL.read_text(encoding="utf-8")
    allowed = set(original)
    skip = ALREADY_IN_ORIGINAL | PROMPT_PLAY | NOT_DIALOGUE
    works = list(split_works(body))
    missing = skip - {title for title, _ in works}
    if missing:
        raise SystemExit(f"titles not found in the source: {sorted(missing)}")
    parts, report = [], []
    for title, work in works:
        if title in skip:
            continue
        speeches = convert_play(work, allowed)
        parts.extend(speeches)
        report.append((title, len(speeches), sum(len(s) + 2 for s in speeches)))
    extra = "\n\n".join(parts) + "\n\n"
    assert set(extra) <= allowed, sorted(set(extra) - allowed)
    for phrase in ("To be, or not to be", "Good morrow, neighbour Baptista"):   # prompt play, validation play
        assert phrase not in extra, phrase
    if "--check" in sys.argv:
        # Quality check: convert a play that IS in the original file and compare line by line.
        play = dict(works)["THE TRAGEDY OF CORIOLANUS"]
        lines = [l for s in convert_play(play, allowed) for l in s.split("\n")]
        original_lines = set(original.split("\n"))
        same = sum(l in original_lines for l in lines)
        print(f"Coriolanus converted: {len(lines)} lines, {100 * same / len(lines):.1f}% identical to lines of {ORIGINAL}")
    for title, speeches, chars in report:
        print(f"{title:45s} {speeches:5d} speeches {chars:8,d} characters")
    print(f"{len(report)} plays, {len(parts):,} speeches, {len(extra):,} characters "
          f"({len(extra) / int(0.9 * len(original)):.1f}x the original training split)")
    if "--check" not in sys.argv:
        OUTPUT.write_text(extra, encoding="utf-8")
        print("written", OUTPUT, "sha256", hashlib.sha256(extra.encode()).hexdigest())
    return extra


if __name__ == "__main__":
    main()
