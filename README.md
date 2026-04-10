# 🔬 BERTopic Agentic Thematic Analysis

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Gradio-6.11-orange?logo=gradio&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-0.2%2B-green?logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/Mistral-Large%20Latest-purple?logo=mistral&logoColor=white" />
  <img src="https://img.shields.io/badge/BERTopic-0.16%2B-red" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

<p align="center">
  A production-grade <strong>agentic AI application</strong> that automates the full
  <a href="https://doi.org/10.1191/1478088706qp063oa">Braun &amp; Clarke (2006)</a>
  six-phase thematic analysis framework on academic literature corpora exported from Scopus.
  <br /><br />
  Built with <strong>BERTopic · LangGraph · Mistral LLM · Gradio</strong>
</p>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [The Six Phases](#the-six-phases)
- [PAJAIS Taxonomy Mapping](#pajais-taxonomy-mapping)
- [Performance Notes](#performance-notes)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)
- [Technical Design Decisions](#technical-design-decisions)
- [Requirements](#requirements)
- [License](#license)

---

## Overview

This application implements a **fully agentic** thematic analysis pipeline for academic literature. Given a Scopus CSV export, it:

1. **Discovers** latent topic clusters using BERTopic with sentence embeddings
2. **Labels** each cluster using Mistral LLM via batch prompting
3. **Consolidates** clusters into overarching themes using Braun & Clarke methodology
4. **Maps** themes against the PAJAIS 25-category IS taxonomy
5. **Generates** a 500-word Section 7 narrative ready for academic publication
6. **Compares** abstract-based vs title-based analyses side by side

The agent is **interactive** — it pauses at 4 mandatory STOP gates for researcher review and approval, ensuring human oversight at every critical decision point.

**Corpus used in development:** Journal of Enterprise Information Management (JEIM), 1,085 papers, 2004–2025.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Gradio 6.11 UI                           │
│  ┌──────────────┐  ┌────────────────────┐  ┌─────────────────┐ │
│  │ ① Data Input │  │ ② Agent Chatbot     │  │ ③ Results       │ │
│  │  CSV Upload  │  │  LangGraph ReAct   │  │  Review Table   │ │
│  │  Auto Phase1 │  │  4 STOP Gates      │  │  Charts (Plotly)│ │
│  └──────────────┘  └────────────────────┘  │  Downloads      │ │
│                                            └─────────────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           │ agent.invoke()
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LangGraph ReAct Agent                         │
│   ChatMistralAI (mistral-large-latest)  +  MemorySaver          │
│   System Prompt: ~500-line Braun & Clarke instruction set       │
└──────────────────────────┬──────────────────────────────────────┘
                           │ tool calls
          ┌────────────────┼────────────────────────┐
          ▼                ▼                        ▼
┌──────────────┐  ┌─────────────────┐  ┌──────────────────────┐
│ load_scopus  │  │run_bertopic_    │  │ label_topics_with_   │
│    _csv      │  │  discovery      │  │      llm             │
│              │  │                 │  │ (BATCH — 1 API call) │
│ utf-8-sig    │  │ all-MiniLM-L6v2 │  └──────────────────────┘
│ quoting=0    │  │ AgglomCluster   │
│ BOM-safe     │  │ cosine, 0.7     │  ┌──────────────────────┐
└──────────────┘  │ NO UMAP         │  │consolidate_into_     │
                  │ 3000-sent cap   │  │     themes           │
                  │ 4 Plotly charts │  └──────────────────────┘
                  └─────────────────┘
                                       ┌──────────────────────┐
                                       │ compare_with_        │
                                       │    taxonomy          │
                                       │ (PAJAIS 25 cats)     │
                                       └──────────────────────┘
                                       ┌──────────────────────┐
                                       │generate_comparison   │
                                       │      _csv            │
                                       └──────────────────────┘
                                       ┌──────────────────────┐
                                       │  export_narrative    │
                                       │  (500-word Sec. 7)   │
                                       └──────────────────────┘
```

---

## Features

| Feature | Detail |
|---|---|
| **7 Specialised Tools** | Each decorated with `@tool`, pure functional, zero loops |
| **Batch LLM Labelling** | All topics in 1 API call vs 100 sequential calls (33× faster) |
| **3000-Sentence Cap** | Prevents 730MB distance matrix; clustering completes in ~30s |
| **No UMAP** | Clusters directly in 384-dim space; avoids curse of low dimensionality |
| **4 STOP Gates** | Researcher reviews after Phase 2, 3, 4, and 5.5 |
| **Session Rotation** | Auto-detects `INVALID_CHAT_HISTORY` and rotates LangGraph thread |
| **Rate-Limit Hardening** | 30/60/90s back-off; `max_retries=0` in tools prevents double-retry |
| **UTF-8 Everywhere** | BOM-safe CSV reading; emoji-safe Windows console |
| **4 Interactive Charts** | Intertopic map, frequency bars, treemap, cosine heatmap |
| **Dual Run** | Abstract analysis + Title analysis → side-by-side `comparison.csv` |
| **PAJAIS Mapping** | Maps themes to 25 IS taxonomy categories; flags NOVEL themes |
| **Phase Persistence** | Checkpoint files restore progress bar on app restart |

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/bertopic-thematic-analysis.git
cd bertopic-thematic-analysis

# 2. Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# 5. Set your Mistral API key
# Windows:
set MISTRAL_API_KEY=your_key_here
# Mac/Linux:
export MISTRAL_API_KEY=your_key_here

# 6. Launch
python app.py
# Open http://localhost:7860
```

Get a free Mistral API key at [console.mistral.ai](https://console.mistral.ai).

---

## Installation

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10 or higher |
| RAM | 4 GB minimum, 8 GB recommended |
| Storage | 2 GB (for model weights + embeddings) |
| Mistral API key | Free tier works (rate limits apply) |

### HuggingFace Spaces Deployment

1. Go to [huggingface.co](https://huggingface.co) → **New Space**
2. Name: `topic_modelling` → SDK: **Gradio** → Create
3. **Settings → Repository Secrets → Add:**
   - Name: `MISTRAL_API_KEY`
   - Value: your key from [console.mistral.ai](https://console.mistral.ai)
4. Upload `app.py`, `agent.py`, `tools.py`, `requirements.txt`
5. Wait 3–5 minutes for build → your app is live

> **Note:** HuggingFace free tier has 16 GB RAM. `all-MiniLM-L6-v2` uses ~200 MB — well within limits.

---

## Usage

### Step-by-Step Workflow

```
1. Upload CSV  →  Phase 1: Stats appear automatically
2. Type "run abstract"  →  Phase 2: BERTopic clusters + labels (3–8 min)
3. Review table  →  Approve/Rename topics  →  Submit Review
4. Phase 3 auto  →  Type "Continue"
5. Phase 4 auto  →  Type "Continue"
6. Phase 5 auto  →  Type "Continue"
7. Phase 5.5  →  Review PAJAIS mapping  →  Type "Continue"
8. Phase 6  →  Download narrative.txt + all files
9. Type "run title"  →  Repeat steps 2–8 for title analysis
```

### Chat Commands

| Command | Effect |
|---|---|
| `run abstract` | Start Phase 2 using the Abstract column |
| `run title` | Start Phase 2 using the Title column |
| `Continue` | Proceed past a STOP gate |
| `run title with threshold 0.65` | Use lower clustering threshold for more granular topics |

### CSV Format

Your Scopus CSV must have at minimum these columns:

| Column | Required | Used for |
|---|---|---|
| `Title` | Yes | Title run clustering |
| `Abstract` | Yes | Abstract run clustering |
| `Year` | No | Displayed in stats |
| `Authors` | No | Metadata only |

Export from Scopus: **Export → CSV → Select all fields**.

---

## Project Structure

```
bertopic-thematic-analysis/
│
├── app.py              # Gradio UI — all user interaction, session management
├── agent.py            # LangGraph ReAct agent + ~500-line system prompt
├── tools.py            # 7 @tool functions — all business logic
├── requirements.txt    # 13 pip packages
│
├── README.md           # This file
│
# Generated at runtime (git-ignored):
├── loaded_data.csv             # Cleaned CSV copy
├── summaries_abstract.json     # Raw cluster summaries
├── summaries_title.json
├── emb_abstract.npy            # 384-dim sentence embeddings
├── emb_title.npy
├── labels_abstract.json        # LLM-labelled topics
├── labels_title.json
├── themes_abstract.json        # Consolidated themes
├── themes_title.json
├── themes.json                 # Active themes (canonical)
├── taxonomy_map.json           # PAJAIS mapping
├── comparison.csv              # Abstract vs title comparison
├── narrative.txt               # 500-word Section 7 draft
├── chart_abstract_*.html       # 4 Plotly charts (abstract run)
├── chart_title_*.html          # 4 Plotly charts (title run)
└── error.txt                   # Error log
```

---

## The Six Phases

### Phase 1 — Familiarisation
Loads the CSV, reports statistics (paper count, sentence count, year range, column coverage), asks researcher which run to start with.

### Phase 2 — Generating Initial Codes ⛔ STOP GATE 1
Runs BERTopic pipeline:
- Splits abstracts/titles to sentences via NLTK `sent_tokenize`
- Removes 22 boilerplate patterns (publisher noise, copyright strings)
- Embeds up to 3,000 sentences with `all-MiniLM-L6-v2` (384-dim, normalised)
- Clusters with `AgglomerativeClustering(metric='cosine', linkage='average', distance_threshold=0.7)`
- Finds 5 nearest sentences per cluster centroid
- Sends all clusters to Mistral in **one batch call** for labelling
- Researcher reviews the populated table, approves/renames topics

### Phase 3 — Searching for Themes ⛔ STOP GATE 2
Consolidates approved topic clusters into 4–8 overarching themes via Mistral LLM (Braun & Clarke guided). Researcher reviews and confirms.

### Phase 4 — Reviewing Themes ⛔ STOP GATE 3
Checks corpus coverage. Flags if themes cover < 80% of sentences. Researcher confirms saturation.

### Phase 5 — Defining and Naming Themes
Agent writes academic definitions and core narratives for each theme. Researcher can rename via the table.

### Phase 5.5 — PAJAIS Taxonomy Mapping ⛔ STOP GATE 4
Maps each theme to the 25-category PAJAIS IS taxonomy. Themes not matching any category are flagged as **NOVEL** — potential new contributions to the field.

### Phase 6 — Producing the Report
- Generates `comparison.csv` (abstract vs title side-by-side)
- Generates `narrative.txt` (500-word Section 7 for academic paper)
- All files available in the Download tab

---

## PAJAIS Taxonomy Mapping

The system maps discovered themes against the 25-category PAJAIS Information Systems taxonomy:

| # | Category | # | Category |
|---|---|---|---|
| 1 | Artificial Intelligence Methods | 14 | Text Mining & Analytics |
| 2 | Natural Language Processing | 15 | Sentiment Analysis |
| 3 | Machine Learning | 16 | Social Media Analysis |
| 4 | Deep Learning | 17 | Business Intelligence |
| 5 | Knowledge Representation | 18 | Process Automation & RPA |
| 6 | Ontologies & Semantic Web | 19 | Computer Vision |
| 7 | Information Retrieval | 20 | Speech & Audio Processing |
| 8 | Recommender Systems | 21 | Multi-Agent Systems |
| 9 | Decision Support Systems | 22 | Robotics & Autonomous Systems |
| 10 | Human-Computer Interaction | 23 | Healthcare & Biomedical AI |
| 11 | Explainability & Transparency | 24 | Finance & Risk Analytics |
| 12 | Fairness, Accountability & Ethics | 25 | Education & E-Learning |
| 13 | Data Management & Integration | | |

Themes that do not match any category are classified as **🌟 NOVEL** — these represent emerging research areas not yet represented in the standard taxonomy, and are the most publishable findings of the analysis.

---

## Performance Notes

### Why no UMAP?

Standard BERTopic uses UMAP to reduce 384-dim embeddings to 5-dim before clustering. In testing on JEIM corpus:
- UMAP 384d → 5d caused **"curse of low dimensionality"**: HDBSCAN collapsed 11,000 sentences into only 2 topics
- AgglomerativeClustering in **full 384-dim space** with cosine metric correctly produced 60–100 granular topics

### Why the 3,000-sentence cap?

AgglomerativeClustering builds a full pairwise distance matrix. Without a cap:

| Sentences | Distance matrix | RAM | Time |
|---|---|---|---|
| 13,829 (full JEIM corpus) | 730 MB | Swap to disk | 900s+ timeout |
| 3,000 (capped) | 34 MB | In-memory | ~30s |

3,000 sentences is a representative sample — the theme structure is identical.

### Why batch LLM labelling?

| Method | API calls | Time |
|---|---|---|
| Sequential (old) | 100 calls × 5s each | ~500s |
| Batch (current) | 1 call | ~15s |

All 60 topic clusters are sent in a single prompt. Mistral returns a JSON array, which is merged back by `topic_id`.

---

## Output Files

| File | Description | Used for |
|---|---|---|
| `narrative.txt` | 500-word Section 7 draft | Copy into your paper |
| `comparison.csv` | Abstract vs title theme comparison | Convergence analysis |
| `themes.json` | Consolidated theme objects with sentences | Reference data |
| `taxonomy_map.json` | PAJAIS mapping with NOVEL flags | Gap analysis |
| `labels_abstract.json` | All labelled topic codes (abstract) | Appendix / audit |
| `labels_title.json` | All labelled topic codes (title) | Appendix / audit |
| `chart_*_intertopic.html` | PCA 2D cluster distance map | Paper figure |
| `chart_*_bars.html` | Topic frequency bar chart | Paper figure |
| `chart_*_hierarchy.html` | Topic treemap by sentence count | Paper figure |
| `chart_*_heatmap.html` | Inter-topic cosine similarity | Paper figure |

---

## Troubleshooting

### `ReadTimeout: The read operation timed out`
Mistral is slow. The app retries automatically with 30/60/90s back-off. Wait and do not click anything.

### `INVALID_CHAT_HISTORY` error
A tool call failed mid-execution leaving a dangling state in LangGraph's MemorySaver. The app **automatically rotates the session ID** and recovers. You should see a message like:
```
⚠️ Corrupt history detected — rotating session abc12345 → def67890
```

### Phase 2 takes more than 10 minutes
You may be on the broken `loaded_data.csv` from a previous run with wrong quoting. Delete `loaded_data.csv` and re-upload the original CSV.

### Review table is empty after Phase 2
1. Check `labels_abstract.json` exists in your working directory
2. Press F5 to refresh the browser — the table loads from disk on startup

### `UnicodeEncodeError` on Windows console
The app reconfigures `stdout` to UTF-8 at startup. If you still see this, run:
```bash
set PYTHONIOENCODING=utf-8
python app.py
```

### Mistral 429 Rate Limit
Free tier allows ~1 request/second. The 3,000-sentence cap and batch labelling are specifically designed to minimise API calls. If 429s persist, wait 60 seconds before retrying.

### `col_count` deprecation warning
Harmless. Gradio renamed this parameter. The code uses the new `column_count` parameter.

---

## Technical Design Decisions

### Functional Programming in `tools.py`
All 7 tools use `map()`, `filter()`, and list comprehensions exclusively. Zero `if/else` statements, zero `for/while` loops, zero `try/except` blocks. This is intentional — it mirrors the constraint from the course assignment and demonstrates functional Python style.

### LangGraph MemorySaver
Each user session gets a UUID `thread_id`. LangGraph's `MemorySaver` maintains conversation history across all tool calls within a session. The `on_clear()` handler rotates to a new UUID and deletes all checkpoint files for a clean restart.

### Gradio 6.11 Compatibility
This Gradio build is `hf-gradio` — a stripped-down HuggingFace variant that:
- Has no `type=` parameter on `gr.Chatbot`
- Has no `show_copy_button=` parameter
- Requires `{"role": "user"|"assistant", "content": str}` dict format for all messages
- Accepts `column_count=` (not the deprecated `col_count=`)

The code handles this via `inspect.signature` probing and always uses the dict message format.

---

## Requirements

```
gradio>=4.44.0
langchain-core>=0.2.0
langchain-mistralai>=0.1.0
langgraph>=0.2.0
sentence-transformers>=2.7.0
scikit-learn>=1.4.0
bertopic>=0.16.0
plotly>=5.22.0
numpy>=1.26.0
pandas>=2.2.0
hdbscan>=0.8.33
umap-learn>=0.5.6
pynndescent>=0.5.12
nltk>=3.8.1
```

> **Note:** `hdbscan`, `umap-learn`, and `pynndescent` are required by BERTopic internally even though UMAP/HDBSCAN are not used for clustering in this implementation.

---

## Acknowledgements

- Braun, V. & Clarke, V. (2006). Using thematic analysis in psychology. *Qualitative Research in Psychology*, 3(2), 77–101.
- [BERTopic](https://maartengr.github.io/BERTopic/) by Maarten Grootendorst
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) by Microsoft
- [Mistral AI](https://mistral.ai) — LLM backbone
- [LangGraph](https://langchain-ai.github.io/langgraph/) — agent orchestration
- [Gradio](https://gradio.app) — UI framework

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with Claude Sonnet 4.5 · Braun &amp; Clarke (2006) · PAJAIS Taxonomy
</p>
