import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency checks — explicit errors, no silent fallback
# ---------------------------------------------------------------------------

def check_deps():
    missing = []
    for pkg, pip_name in [
        ("pandas",          "pandas"),
        ("pyarrow",         "pyarrow"),
        ("jinja2",          "jinja2"),
        ("gguf",            "gguf"),
        ("huggingface_hub", "huggingface_hub"),
    ]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print("\n[ERROR] Missing required packages:")
        for m in missing:
            print(f"  pip install {m}")
        sys.exit(1)

check_deps()

import pandas as pd
from jinja2 import Template
from gguf import GGUFReader
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------

HF_REPO = "eaddario/imatrix-calibration"
CACHE_DIR = Path.home() / ".cache" / "dataset-builder"
MAX_GGUF_BYTES = 4 * 1024 * 1024 * 1024 * 1024

CATALOGUE = {
    # ── text — single languages ───────────────────────────────────────────
    "text_ar": {"files": {"micro":1625,"tiny":3250,"small":6500,"medium":13000,"large":26000}, "lang":"ar", "cat":"text"},
    "text_cn": {"files": {"micro":1625,"tiny":3250,"small":6500,"medium":13000,"large":26000}, "lang":"zh", "cat":"text"},
    "text_de": {"files": {"micro":1625,"tiny":3250,"small":6500,"medium":13000,"large":26000}, "lang":"de", "cat":"text"},
    "text_en": {"files": {"micro":1625,"tiny":3250,"small":6500,"medium":13000,"large":26000}, "lang":"en", "cat":"text"},
    "text_es": {"files": {"micro":1625,"tiny":3250,"small":6500,"medium":13000,"large":26000}, "lang":"es", "cat":"text"},
    "text_fr": {"files": {"micro":1625,"tiny":3250,"small":6500,"medium":13000,"large":26000}, "lang":"fr", "cat":"text"},
    "text_hi": {"files": {"micro":1625,"tiny":3250,"small":6500,"medium":13000,"large":26000}, "lang":"hi", "cat":"text"},
    "text_it": {"files": {"micro":1625,"tiny":3250,"small":6500,"medium":13000,"large":26000}, "lang":"it", "cat":"text"},
    "text_jp": {"files": {"micro":1625,"tiny":3250,"small":6500,"medium":13000,"large":26000}, "lang":"ja", "cat":"text"},
    "text_nl": {"files": {"micro":1625,"tiny":3250,"small":6500,"medium":13000,"large":26000}, "lang":"nl", "cat":"text"},
    "text_pl": {"files": {"micro":1625,"tiny":3250,"small":6500,"medium":13000,"large":26000}, "lang":"pl", "cat":"text"},
    "text_pt": {"files": {"micro":1625,"tiny":3250,"small":6500,"medium":13000,"large":26000}, "lang":"pt", "cat":"text"},
    "text_ru": {"files": {"micro":1625,"tiny":3250,"small":6500,"medium":13000,"large":26000}, "lang":"ru", "cat":"text"},
    # ── text — language groups ────────────────────────────────────────────
    "text_all": {"files": {"micro":1625,"tiny":3250,"small":6500,"medium":13000,"large":26000}, "lang":"all", "cat":"text_group"},
    "text_eur": {"files": {"micro":1620,"tiny":3246,"small":6498,"medium":12996,"large":25998}, "lang":"eur", "cat":"text_group"},
    "text_gem": {"files": {"micro":1623,"tiny":3250,"small":6498,"medium":12999,"large":25998}, "lang":"gem", "cat":"text_group"},
    "text_roa": {"files": {"micro":1622,"tiny":3252,"small":6500,"medium":13000,"large":26000}, "lang":"roa", "cat":"text_group"},
    "text_sla": {"files": {"micro":1624,"tiny":3250,"small":6500,"medium":13000,"large":26000}, "lang":"sla", "cat":"text_group"},
    # ── math ─────────────────────────────────────────────────────────────
    "math":  {"files": {"micro":13653,"tiny":27297,"small":54587,"medium":109168,"large":218332,"huge":436661}, "lang":"en", "cat":"math"},
    # ── code ─────────────────────────────────────────────────────────────
    "code":  {"files": {"micro":6324,"tiny":12645,"small":25288,"medium":50576,"large":101148,"huge":202295},   "lang":"en", "cat":"code"},
    # ── tools ────────────────────────────────────────────────────────────
    "tools": {"files": {"micro":3638,"tiny":7275,"small":14550,"medium":29100,"large":58199,"huge":116398},     "lang":"en", "cat":"tools"},
    # ── combined ─────────────────────────────────────────────────────────
    "combined_en":  {"files": {"micro":4159,"tiny":8319,"small":16638,"medium":33275,"large":66550}, "lang":"en",  "cat":"combined"},
    "combined_fr":  {"files": {"micro":4159,"tiny":8319,"small":16638,"medium":33275,"large":66550}, "lang":"fr",  "cat":"combined"},
    "combined_all": {"files": {"micro":4158,"tiny":8318,"small":16637,"medium":33274,"large":66548}, "lang":"all", "cat":"combined"},
    "combined_eur": {"files": {"micro":4151,"tiny":8312,"small":16635,"medium":33268,"large":66547}, "lang":"eur", "cat":"combined"},
}

AVG_CHARS = {"text": 200, "text_group": 200, "math": 215, "code": 750, "tools": 432, "combined": 300}
CONTEXT   = 512
SIZES     = ["micro", "tiny", "small", "medium", "large", "huge"]

# Single language display list
SINGLE_LANGS = [
    ("ar", "Arabic"),
    ("cn", "Chinese"),
    ("hi", "Hindi"),
    ("jp", "Japanese"),
    ("de", "German"),
    ("en", "English"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("it", "Italian"),
    ("nl", "Dutch"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("ru", "Russian"),
]

# Language groups including Asian
LANG_GROUPS = [
    ("all", "All languages"),
    ("eur", "European  (en / fr / de / it / pt / es)"),
    ("gem", "Germanic  (nl / en / de)"),
    ("roa", "Romance   (fr / it / pt / es)"),
    ("sla", "Slavic    (pl / ru)"),
    # Asian group: individual files, not a combined parquet — handled specially
    ("_asian", "Asian     (ar / cn / hi / jp)"),
]
ASIAN_LANGS = ["ar", "cn", "hi", "jp"]

# ---------------------------------------------------------------------------
# Chunk estimation
# ---------------------------------------------------------------------------

def estimate_chunks(key: str, size: str) -> int:
    entry = CATALOGUE[key]
    rows  = entry["files"].get(size, 0)
    avg   = AVG_CHARS.get(entry["cat"], 200)
    return int(rows * avg / 4 / CONTEXT)

# ---------------------------------------------------------------------------
# Parquet fetch + read
# ---------------------------------------------------------------------------

def fetch_parquet(fname: str) -> Path:
    if ".." in Path(fname).parts:
        print(f"\n[ERROR] Invalid filename (path traversal): {fname}")
        sys.exit(1)
    cached = CACHE_DIR / fname
    if cached.exists():
        print(f"  [cache] {fname}")
        return cached
    print(f"  [download] {fname} ...", end=" ", flush=True)
    try:
        path = hf_hub_download(
            repo_id=HF_REPO,
            filename=fname,
            repo_type="dataset",
            local_dir=str(CACHE_DIR),
        )
        print("done")
        return Path(path)
    except Exception as e:
        print(f"FAILED\n\n[ERROR] Could not download {fname}: {e}")
        sys.exit(1)


def read_rows(fname: str) -> list:
    parquet_path = fetch_parquet(fname)
    df = pd.read_parquet(parquet_path)
    if df.empty:
        print(f" [WARN] {fname}: parquet is empty")
        return []
    col = df.columns[0]
    raw = str(df[col].iloc[0])
    return [line.strip() for line in raw.split("\n") if line.strip()]

# ---------------------------------------------------------------------------
# Chat template extraction
# ---------------------------------------------------------------------------

def extract_chat_template(gguf_path: Path) -> str:
    print(f"\n  Reading chat template from {gguf_path.name} ...")
    try:
        reader = GGUFReader(str(gguf_path), mode="r")
    except Exception as e:
        print(f"\n[ERROR] Could not open GGUF file: {e}")
        sys.exit(1)

    for field in reader.fields.values():
        if field.name == "tokenizer.chat_template":
            tmpl = bytes(field.parts[-1]).decode("utf-8")
            print(f"  Chat template found ({len(tmpl)} chars)")
            return tmpl

    print("\n[ERROR] No 'tokenizer.chat_template' key found in this GGUF.")
    print("  This model may not have a chat template embedded.")
    print("  Use a different model file or an instruct variant.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Jinja2 wrapping
# ---------------------------------------------------------------------------

def wrap_text(text: str, tmpl: Template) -> str:
    return tmpl.render(
        messages=[{"role": "user", "content": text}],
        add_generation_prompt=True,
        bos_token="",
        eos_token="",
        raise_exception=lambda msg: "",
    )

# ---------------------------------------------------------------------------
# GGUF path picker — native file dialog with manual fallback
# ---------------------------------------------------------------------------

def pick_gguf_path() -> Path:
    path = None
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        print("\n  A file picker dialog will open — select your GGUF model file.")
        raw = filedialog.askopenfilename(
            title="Select GGUF model",
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")],
        )
        root.destroy()
        if raw:
            path = Path(raw)
    except Exception:
        pass  # headless — fall through

    if not path:
        print("  (File picker unavailable — enter path manually)")
        raw = input("  Path to GGUF\n  > ").strip()
        path = Path(raw)

    if not path.exists():
        print(f"\n[ERROR] File not found: {path}")
        sys.exit(1)

    if path.stat().st_size > MAX_GGUF_BYTES:
        print(f"\n[ERROR] File too large: {path} ({path.stat().st_size / 1e12:.1f} TB, limit {MAX_GGUF_BYTES / 1e12:.0f} TB)")
        sys.exit(1)

    # Validate GGUF magic bytes — first 4 bytes must be b'GGUF'
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
        if magic != b"GGUF":
            print(f"\n[ERROR] Not a valid GGUF file: {path}")
            print(f"  Expected magic bytes 'GGUF', got: {magic!r}")
            sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Could not read file: {e}")
        sys.exit(1)

    print(f"  Selected: {path}")
    return path

# ---------------------------------------------------------------------------
# Interactive helpers
# ---------------------------------------------------------------------------

def ask(prompt: str, default: str = None) -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"{prompt}{suffix}\n  > ").strip()
    return val if val else (default or "")

def ask_int(prompt: str, valid: list, default: int = None) -> int:
    while True:
        val = ask(prompt, str(default) if default is not None else None)
        try:
            n = int(val)
            if n in valid:
                return n
        except ValueError:
            pass
        print(f"  Please enter one of: {valid}")

def ask_multi(prompt: str, valid: list) -> list:
    while True:
        val = ask(prompt)
        try:
            chosen = [int(x) for x in val.split()]
            if chosen and all(c in valid for c in chosen):
                return chosen
        except ValueError:
            pass
        print(f"  Please enter space-separated numbers from: {valid}")

# ---------------------------------------------------------------------------
# Interactive flow
# ---------------------------------------------------------------------------

def run_interactive():
    print("\n" + "=" * 60)
    print("  dataset-builder — interactive mode")
    print("=" * 60)

    # ── Step 1 — Purpose ─────────────────────────────────────────────────
    print("\nStep 1/5  Purpose")
    print("  1. KLD evaluation")
    print("  2. imatrix calibration")
    is_kld = ask_int("  Enter 1 or 2", [1, 2]) == 1

    # ── Step 2 — Chat wrapping (both purposes) ───────────────────────────
    print("\nStep 2/5  Chat wrapping")
    print("  1. With chat template  [recommended]")
    print("  2. Raw text")
    wrap = ask_int("  Enter 1 or 2", [1, 2]) == 1

    # ── Step 3 — Categories ──────────────────────────────────────────────
    print("\nStep 3/5  Categories  (space-separated, e.g.: 1 3  or  5 for all)")
    print("  1. text    Natural language")
    print("  2. math    Math problems")
    print("  3. code    Code prompts")
    print("  4. tools   Tool-calling prompts")
    print("  5. all     All of the above")
    cats_raw = ask_multi("  Enter numbers", [1, 2, 3, 4, 5])
    if 5 in cats_raw:
        selected_cats = ["text", "math", "code", "tools"]
    else:
        cat_map = {1: "text", 2: "math", 3: "code", 4: "tools"}
        selected_cats = [cat_map[c] for c in cats_raw]

    # ── Step 4 — Language (skipped if "all" or no text) ──────────────────
    selected_sources = []

    if "text" in selected_cats:
        print("\nStep 4/5  Language for text")
        print("  1. Single language  (pick one or more)")
        print("  2. Language group")
        lang_type = ask_int("  Enter 1 or 2", [1, 2])

        if lang_type == 1:
            print("\n  Available languages:")
            for i, (code, name) in enumerate(SINGLE_LANGS, 1):
                print(f"    {i:2}. {name} ({code})")
            print("  You can select multiple (e.g.: 1 4 6)")
            idxs = ask_multi("  Enter numbers", list(range(1, len(SINGLE_LANGS) + 1)))
            for idx in idxs:
                code = SINGLE_LANGS[idx - 1][0]
                selected_sources.append(f"text_{code}")
        else:
            print("\n  Available groups:")
            for i, (code, name) in enumerate(LANG_GROUPS, 1):
                print(f"    {i}. {name}")
            idx = ask_int("  Enter number", list(range(1, len(LANG_GROUPS) + 1)))
            code = LANG_GROUPS[idx - 1][0]
            if code == "_asian":
                for lang in ASIAN_LANGS:
                    selected_sources.append(f"text_{lang}")
            else:
                selected_sources.append(f"text_{code}")
    else:
        print("\nStep 4/5  Language — skipped (no text category selected)")

    # Add non-text categories
    for cat in selected_cats:
        if cat != "text":
            selected_sources.append(cat)

    # ── Step 5 — Target chunks ────────────────────────────────────────────
    print("\nStep 5/5  Target chunks")
    if is_kld:
        print("  1.  50 chunks  (quick check)")
        print("  2. 100 chunks  [recommended]")
        preset = ask_int("  Enter 1 or 2", [1, 2])
        target_chunks = 50 if preset == 1 else 100
    else:
        print("  1.  100 chunks")
        print("  2.  250 chunks")
        print("  3.  500 chunks")
        print("  4.  750 chunks  [recommended]")
        print("  5. 1000 chunks")
        preset = ask_int("  Enter 1-5", [1, 2, 3, 4, 5])
        target_chunks = [100, 250, 500, 750, 1000][preset - 1]

    # ── Size selection ────────────────────────────────────────────────────
    print(f"\n  Target: {target_chunks} chunks at -c {CONTEXT}")
    print("  Checking availability...\n")

    source_plans = []
    any_insufficient = False
    n_sources = max(1, len(selected_sources))
    chunks_per_src = max(1, target_chunks // n_sources)

    for key in selected_sources:
        entry = CATALOGUE.get(key)
        if not entry:
            print(f" [WARN] Unknown source key: {key} — skipped")
            continue

        chosen_size = None
        chosen_est = 0
        for sz in SIZES:
            if sz not in entry["files"]:
                continue
            est = estimate_chunks(key, sz)
            if est >= chunks_per_src:
                chosen_size = sz
                chosen_est = est
                break

        if chosen_size is None:
            for sz in reversed(SIZES):
                if sz in entry["files"]:
                    chosen_size = sz
                    chosen_est = estimate_chunks(key, sz)
                    break
            print(f" [WARN] {key}: max ~{chosen_est} chunks (target {chunks_per_src} unreachable)")
            any_insufficient = True
        else:
            print(f" {key}: '{chosen_size}' -> ~{chosen_est} chunks")

        source_plans.append((key, chosen_size, f"{key}_{chosen_size}.parquet", entry["cat"], chosen_est))

    if any_insufficient:
        if ask("\n  Some sources cannot reach target. Continue anyway? [y/N]", "n").lower() != "y":
            print("  Aborted.")
            sys.exit(0)

    # ── Output filename ───────────────────────────────────────────────────
    ts           = datetime.now().strftime("%y%m%d-%H%M")
    purpose_tag  = "eval" if is_kld else "imatrix"
    default_name = f"{purpose_tag}_dataset_{ts}.txt"
    print(f"\n  Output file  [default: {default_name}]")
    raw_out  = input("  Press Enter to accept or type a custom path\n  > ").strip()
    out_path = Path(raw_out) if raw_out else Path(default_name)

    # ── GGUF (only if wrapping) ───────────────────────────────────────────
    gguf_path = None
    if wrap:
        gguf_path = pick_gguf_path()

    return source_plans, target_chunks, is_kld, wrap, gguf_path, out_path

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def resolve_output_path(raw_path: Path, purpose_tag: str) -> Path:
    ts = datetime.now().strftime("%y%m%d-%H%M")
    default_name = f"{purpose_tag}_dataset_{ts}.txt"
    if raw_path.is_dir() or (not raw_path.suffix and not raw_path.exists()):
        return raw_path / default_name
    return raw_path


def build(source_plans, target_chunks, is_kld, wrap, gguf_path, out_path, seed=42):
    rng = random.Random(seed)

    jinja_tmpl = None
    if wrap:
        jinja_tmpl = Template(extract_chat_template(gguf_path))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []

    # Divide target evenly across sources
    n_sources       = len(source_plans)
    chunks_per_src  = max(1, target_chunks // n_sources)

    for key, size, fname, cat, est_chunks in source_plans:
        print(f"\n  {fname}")
        rows = read_rows(fname)
        print(f"  {len(rows):,} rows loaded")

        avg            = AVG_CHARS.get(cat, 200)
        rows_per_chunk = CONTEXT / (avg / 4)
        n_sample       = min(len(rows), int(chunks_per_src * rows_per_chunk))
        sampled        = rng.sample(rows, n_sample)
        print(f"  Sampled {len(sampled):,} rows (~{int(len(sampled)/rows_per_chunk)} chunks)")

        all_rows.extend(
            [wrap_text(r, jinja_tmpl) for r in sampled] if jinja_tmpl else sampled
        )

    purpose_tag = "eval" if is_kld else "imatrix"
    out_path = resolve_output_path(out_path, purpose_tag)

    rng.shuffle(all_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(all_rows), encoding="utf-8")

    size_kb      = out_path.stat().st_size // 1024
    total_tokens  = sum(len(r) // 4 for r in all_rows)
    est_chunks    = total_tokens // CONTEXT
    print(f"\nDone.")
    print(f"  {len(all_rows):,} rows written -> {out_path}  ({size_kb:,} KB)")
    print(f"  Estimated llama-perplexity chunks at -c {CONTEXT}: ~{est_chunks}")
    if is_kld:
        print(f"\n  Use with kld_sweep.py:")
        print(f"    python kld_sweep.py --dataset \"{out_path}\" [other args]")
    else:
        print(f"\n  Use as imatrix calibration data:")
        print(f"    llama-imatrix -m <model.gguf> -f \"{out_path}\" -o imatrix.dat [other args]")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Build a KLD eval or imatrix calibration dataset.")
    p.add_argument("--purpose",  choices=["kld", "imatrix"])
    p.add_argument("--sources",  nargs="+", help="e.g. text_fr math code")
    p.add_argument("--chunks",   type=int)
    p.add_argument("--wrap",     action="store_true", help="Wrap with chat template")
    p.add_argument("--gguf",     help="Path to GGUF (required if --wrap)")
    p.add_argument("--output",   help="Output .txt path")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--list",     action="store_true", help="List sources and exit")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list:
        print("\nAvailable source keys:\n")
        for key, entry in CATALOGUE.items():
            sizes = ", ".join(entry["files"].keys())
            print(f"  {key:<20} lang={entry['lang']:<6} cat={entry['cat']:<12} sizes=[{sizes}]")
        sys.exit(0)

    if args.purpose and args.sources and args.chunks and args.output:
        if args.chunks < 1:
            print("\n[ERROR] --chunks must be >= 1")
            sys.exit(1)
        # CLI mode
        if args.wrap and not args.gguf:
            print("\n[ERROR] --gguf is required when --wrap is set")
            sys.exit(1)

        gguf_path = Path(args.gguf) if args.gguf else None
        source_plans = []
        n_sources = len(args.sources)
        chunks_per_src = max(1, args.chunks // n_sources)

        for key in args.sources:
            if key not in CATALOGUE:
                print(f"\n[ERROR] Unknown source: {key!r}. Run --list to see options.")
                sys.exit(1)
            entry = CATALOGUE[key]
            chosen_size = None
            for sz in SIZES:
                if sz in entry["files"] and estimate_chunks(key, sz) >= chunks_per_src:
                    chosen_size = sz
                    break
            if not chosen_size:
                for sz in reversed(SIZES):
                    if sz in entry["files"]:
                        chosen_size = sz
                        break
                print(f" [WARN] {key}: cannot reach {chunks_per_src} chunks, using '{chosen_size}'")
            source_plans.append((key, chosen_size, f"{key}_{chosen_size}.parquet",
                                 entry["cat"], estimate_chunks(key, chosen_size)))

        is_kld = (args.purpose == "kld")
        build(source_plans, args.chunks, is_kld, args.wrap, gguf_path, Path(args.output), args.seed)
    else:
        # Interactive mode
        source_plans, target_chunks, is_kld, wrap, gguf_path, out_path = run_interactive()
        build(source_plans, target_chunks, is_kld, wrap, gguf_path, out_path, args.seed)


if __name__ == "__main__":
    main()
