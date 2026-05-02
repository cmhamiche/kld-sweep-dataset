# dataset-builder

A CLI tool to build KLD evaluation and imatrix calibration datasets for GGUF models, sourced from [eaddario/imatrix-calibration](https://huggingface.co/datasets/eaddario/imatrix-calibration).

Files are downloaded on demand and cached in `~/.cache/dataset-builder/`. No repo clone required.

## Requirements

```
pip install pandas pyarrow jinja2 gguf huggingface_hub
```

## Usage

**Interactive (recommended):**
```
python build_dataset.py
```

**CLI:**
```
python build_dataset.py \
--purpose kld \
--sources text_fr math code \
--chunks 100 \
--wrap \
--gguf C:\models\MyModel.gguf \
--output eval_dataset.txt
```

Each run produces a unique dataset — rows are randomly sampled and shuffled by default. Pass `--seed <int>` for reproducible output.

**List available sources:**
```
python build_dataset.py --list
```

## Interactive flow

```
Step 1/5  Purpose          KLD evaluation  /  imatrix calibration
Step 2/5  Chat wrapping    With chat template (from GGUF)  /  Raw text
Step 3/5  Categories       text / math / code / tools / all
Step 4/5  Language         Single language, language group, or Asian group
Step 5/5  Target chunks    50–100 for eval, 100–1000 for calibration
```

The script selects the smallest parquet file that covers your target chunk count,
shows an estimate before downloading, and warns if the target is unreachable.

## Purposes

| Purpose | Wrapping | Suggested chunks |
|---------|----------|-----------------|
| KLD evaluation | Chat template (recommended) or raw | 50 / 100 |
| imatrix calibration | Chat template (recommended) or raw | 100 / 250 / 500 / 750 / 1000 |

## Sources

| Key | Category | Languages |
|-----|----------|-----------|
| `text_fr`, `text_en`, ... | Natural language | ar cn de en es fr hi it jp nl pl pt ru |
| `text_eur`, `text_roa`, `text_sla`, `text_gem`, `text_all` | Language groups | European / Romance / Slavic / Germanic / All |
| `math` | Math problems | en |
| `code` | Code prompts | en |
| `tools` | Tool-calling prompts | en |
| `combined_fr`, `combined_en`, `combined_all`, `combined_eur` | Mixed (text+math+code+tools) | fr / en / all / eur |

**Asian group** (available via language group selection): Arabic / Chinese / Hindi / Japanese

## Chat template

When wrapping is enabled, the Jinja2 chat template is extracted directly from the selected GGUF's metadata (`tokenizer.chat_template`). No separate `tokenizer_config.json` needed. If the key is absent the script exits with a clear error.

## Output

- Named `eval_dataset_YYMMDD-HHMM.txt` or `imatrix_dataset_YYMMDD-HHMM.txt` by default
- `--output` accepts a file path or a directory (auto-generates a timestamped filename inside)
- Plain UTF-8, rows separated by double newlines — compatible with `llama-perplexity` and `llama-imatrix`
- Final summary shows row count and estimated llama-perplexity chunk count at `-c 512`

## Notes

- Chunk estimation uses hardcoded average chars/row per category (derived from sampling). Actual chunk count from llama-perplexity may vary ±10%.
- The `_small` files in the eaddario repo contain ~2x the rows listed in the README.
- The script reads actual row counts at runtime.
- 
- Tools files are stored as a single blob per file and are split by newline internally.
- Datasets are non-reproducible by default (random seed). Use `--seed` if you need deterministic output.
