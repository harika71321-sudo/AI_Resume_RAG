# AI Resume Assistant — RAG

A Streamlit-based AI Resume Assistant that accepts multiple PDF resumes,
creates a local vector knowledge base, retrieves relevant resume evidence,
and uses an NVIDIA-hosted LLM to answer HR questions and rank candidates
against a supplied job description.

## Features

- Multiple PDF resume upload
- Explicit "Create Knowledge Base" button
- PDF text extraction
- Chunking with overlap
- Sentence-Transformer embeddings
- FAISS vector search
- NVIDIA NIM/OpenAI-compatible LLM API
- Resume Q&A
- Candidate ranking against a job description
- Source filename + page citations
- Clear Chat
- Reset Knowledge Base
- Hallucination-reduction prompt
- Job-relevant screening guidance

## Setup

### 1. Create environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add NVIDIA API key

Copy:

```text
.env.example
```

to:

```text
.env
```

Then set:

```text
NVIDIA_API_KEY=your_key
```

### 4. Run

```bash
streamlit run app.py
```

## Workflow

1. Paste the Job Description.
2. Upload multiple PDF resumes.
3. Click "Create Knowledge Base".
4. Click "Rank Candidates" to compare resumes.
5. Ask questions in the chat.
6. Use "Clear Chat" to clear conversation history.
7. Use "Reset Knowledge Base" to remove the current vector index.

## RAG pipeline

PDF resumes
→ page extraction
→ text chunking
→ embeddings
→ FAISS vector database
→ semantic retrieval
→ NVIDIA LLM
→ grounded answer + sources

## Important limitation

This version handles text-based PDFs. Scanned/image-only resumes need OCR,
which can be added as a future module.

For real recruitment use, keep screening criteria job-related and audit the
system for bias. Do not use protected/sensitive personal attributes for
ranking or hiring decisions.
