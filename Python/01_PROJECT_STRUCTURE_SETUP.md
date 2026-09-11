# Step 1: Project Structure & Environment Setup Guide

## 📁 Complete Project Directory Layout

```
insurance-claims-rag/
│
├── README.md                          # Project overview & quick start
├── .gitignore                         # Git ignore patterns
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment template (DO NOT commit .env)
│
├── config/
│   ├── __init__.py
│   ├── settings.py                    # Centralized config (API keys, model selection, paths)
│   └── logging_config.py              # Structured logging setup
│
├── data/
│   ├── raw/                           # Raw input data (UNTRACKED in git)
│   │   ├── policies/                  # Policy documents (PDFs, TXT)
│   │   ├── claims.csv                 # Claims CSV file
│   │   └── .gitkeep
│   ├── processed/                     # After preprocessing
│   │   ├── chunked_policies.json
│   │   ├── processed_claims.csv
│   │   └── .gitkeep
│   └── synthetic/                     # Generated synthetic data
│       ├── synthetic_claims.csv
│       ├── synthetic_policies/
│       └── .gitkeep
│
├── src/
│   ├── __init__.py
│   ├── data_generation/               # Step 1
│   │   ├── __init__.py
│   │   ├── synthetic_data_generator.py
│   │   └── fraud_patterns.py
│   ├── data_ingestion/                # Step 2
│   │   ├── __init__.py
│   │   ├── document_loader.py
│   │   └── claims_loader.py
│   ├── embeddings/                    # Step 3
│   │   ├── __init__.py
│   │   └── embedding_service.py
│   ├── vector_store/                  # Step 3
│   │   ├── __init__.py
│   │   └── chroma_manager.py
│   ├── rag/                           # Step 4
│   │   ├── __init__.py
│   │   └── query_engine.py
│   ├── agents/                        # Step 5
│   │   ├── __init__.py
│   │   ├── fraud_investigator.py
│   │   └── tools.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
│
├── tests/                             # Unit & integration tests
│   ├── __init__.py
│   ├── test_data_generation.py
│   └── test_ingestion.py
│
├── notebooks/                         # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   └── 02_rag_testing.ipynb
│
├── ui/                                # Streamlit app (Step 6)
│   ├── __init__.py
│   ├── app.py
│   └── components/
│
└── logs/                              # Application logs
    └── .gitkeep
```

---

## 🔧 Step 1A: Initial Setup (Run Once)

### 1. Create Git Repository
```bash
cd ~/Desktop/Insurance\ Claims\ RAG\ Intelligence
git init
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# On Windows: venv\Scripts\activate
```

### 3. Create Project Structure
Copy the directory layout above, then run:
```bash
mkdir -p data/{raw,processed,synthetic}
mkdir -p data/raw/{policies,synthetic}
mkdir -p src/{data_generation,data_ingestion,embeddings,vector_store,rag,agents,utils}
mkdir -p tests notebooks ui logs config
touch .gitkeep logs/.gitkeep data/raw/.gitkeep data/processed/.gitkeep data/synthetic/.gitkeep
```

### 4. Install Dependencies
See **Step 1B** below.

---

## 📦 Step 1B: Dependencies (requirements.txt)

Create `requirements.txt` in your project root:

```txt
# Core Data Processing
pandas==2.1.3
numpy==1.26.2
python-dotenv==1.0.0

# LLM & RAG Stack
llama-index==0.9.36
llama-index-embeddings-huggingface==0.1.2
chromadb==0.4.13
sentence-transformers==2.2.2

# LLM Providers
anthropic==0.7.6

# Data Generation & Validation
faker==20.1.0
pydantic==2.5.0

# Utilities
python-dateutil==2.8.2
pytz==2023.3

# UI (Optional for Step 6)
streamlit==1.28.1
plotly==5.17.0

# Testing
pytest==7.4.3
pytest-cov==4.1.0

# Development
black==23.11.0
flake8==6.1.0
mypy==1.7.1
```

**Install:**
```bash
pip install -r requirements.txt
```

---

## 🔐 Step 1C: Environment Configuration

Create `.env` file in your project root (DO NOT COMMIT TO GIT):

```env
# LLM Configuration
LLM_PROVIDER=anthropic  # Options: anthropic, ollama
ANTHROPIC_API_KEY=your_api_key_here  # Get from https://console.anthropic.com
OLLAMA_BASE_URL=http://localhost:11434  # For local Ollama

# Embeddings
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# ChromaDB
CHROMA_DB_PATH=./data/chroma_db
CHROMA_COLLECTION_NAME=insurance_claims

# Data Paths
POLICIES_DIR=./data/raw/policies
CLAIMS_CSV_PATH=./data/raw/claims.csv
SYNTHETIC_DATA_DIR=./data/synthetic

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log

# Synthetic Data Generation
RANDOM_SEED=42
NUM_SYNTHETIC_CLAIMS=1000
NUM_SYNTHETIC_POLICIES=8
FRAUD_RATE=0.06  # 6% fraud prevalence
```

Add `.env` to `.gitignore`:
```
.env
venv/
__pycache__/
*.pyc
.DS_Store
logs/
data/raw/
data/processed/
.pytest_cache/
.mypy_cache/
```

---

## ✅ Step 1D: Verify Setup

Run this quick verification script:

```python
# verify_setup.py
import sys
from pathlib import Path

def verify_setup():
    """Verify project structure is correct."""
    required_dirs = [
        "config", "data/raw", "data/processed", "data/synthetic",
        "src/data_generation", "src/data_ingestion", "src/embeddings",
        "src/vector_store", "src/rag", "src/agents", "src/utils",
        "tests", "notebooks", "ui", "logs"
    ]
    
    missing = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing.append(dir_path)
    
    if missing:
        print(f"❌ Missing directories: {missing}")
        return False
    
    print("✅ Project structure verified!")
    
    # Check dependencies
    try:
        import pandas
        import llama_index
        import chromadb
        import sentence_transformers
        import anthropic
        print("✅ All core dependencies installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

if __name__ == "__main__":
    success = verify_setup()
    sys.exit(0 if success else 1)
```

Run it:
```bash
python verify_setup.py
```

---

## 📝 Next Step

Once you've completed the setup above and run verification successfully, I'll provide **the Synthetic Data Generator Script** (Step 1 Main Deliverable).

**Checkpoint:** When ready, reply with:
> "Step 1A-1D Setup Complete"

Then we'll proceed to the production-grade synthetic data generator that creates 1,000 realistic claims with seeded fraud patterns.
