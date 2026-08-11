import json
from pathlib import Path
from collections import defaultdict

from config import (
    VECTOR_DIR,
    EMBEDDING_MODEL,
    TOP_K,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)
from pdf_processor import extract_pdf_pages, extract_candidate_name
from chunker import chunk_text
from vector_store import VectorStore
from nvidia_llm import NvidiaLLM
from prompts import SYSTEM_PROMPT, RANKING_PROMPT


class ResumeRAG:
    def __init__(self):
        self.vector_store = VectorStore(EMBEDDING_MODEL)
        self.llm = NvidiaLLM()
        self.resume_records = []

    def build_knowledge_base(self, uploaded_files):
        documents = []
        self.resume_records = []

        for uploaded_file in uploaded_files:
            temp_path = VECTOR_DIR / uploaded_file.name
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.write_bytes(uploaded_file.getbuffer())

            pages = extract_pdf_pages(temp_path)
            candidate_name = extract_candidate_name(
                pages,
                uploaded_file.name
            )

            self.resume_records.append({
                "candidate_name": candidate_name,
                "file": uploaded_file.name
            })

            for page in pages:
                chunks = chunk_text(
                    page["text"],
                    chunk_size=CHUNK_SIZE,
                    overlap=CHUNK_OVERLAP
                )

                for chunk_id, chunk in enumerate(chunks):
                    documents.append({
                        "text": chunk,
                        "candidate": candidate_name,
                        "file": uploaded_file.name,
                        "page": page["page"],
                        "chunk_id": chunk_id
                    })

        if not documents:
            raise ValueError(
                "No readable text was found in the uploaded PDFs. "
                "Scanned/image-only PDFs need OCR."
            )

        self.vector_store.build(documents)
        self.vector_store.save(VECTOR_DIR)

        return {
            "documents": len(self.resume_records),
            "chunks": len(documents)
        }

    def retrieve(self, query: str, top_k: int = TOP_K):
        return self.vector_store.search(query, top_k=top_k)

    def answer(self, question, job_description="", chat_history=None):
        query = question

        if job_description:
            query += f"\nJob Description:\n{job_description}"

        results = self.retrieve(query)

        context_parts = []
        sources = []

        for item in results:
            context_parts.append(
                f"[Candidate: {item['candidate']} | "
                f"File: {item['file']} | Page: {item['page']}]\n"
                f"{item['text']}"
            )

            sources.append({
                "candidate": item["candidate"],
                "file": item["file"],
                "page": item["page"]
            })

        context = "\n\n".join(context_parts)

        history_text = ""
        if chat_history:
            recent = chat_history[-6:]
            history_text = "\n".join(
                f"{m['role']}: {m['content']}" for m in recent
            )

        user_prompt = f"""
Job Description:
{job_description or "Not provided"}

Previous conversation:
{history_text or "None"}

Retrieved Resume Context:
{context}

Question:
{question}

Answer only from the retrieved context.
"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        answer = self.llm.generate(messages)
        return answer, self._unique_sources(sources)

    def rank_candidates(self, job_description: str):
        if not job_description.strip():
            raise ValueError("Please provide a job description.")

        # Retrieve several relevant chunks for the complete JD.
        results = self.retrieve(
            f"Job requirements and candidate qualifications:\n{job_description}",
            top_k=max(TOP_K, 20)
        )

        by_candidate = defaultdict(list)
        for item in results:
            by_candidate[item["candidate"]].append(item)

        # If retrieval misses a candidate entirely, give the LLM the
        # available candidate list but do not invent evidence.
        candidate_blocks = []

        for candidate, items in by_candidate.items():
            block = [f"Candidate: {candidate}"]
            for item in items:
                block.append(
                    f"Source: {item['file']}, page {item['page']}\n"
                    f"{item['text']}"
                )
            candidate_blocks.append("\n".join(block))

        context = "\n\n---\n\n".join(candidate_blocks)

        user_prompt = f"""
JOB DESCRIPTION:
{job_description}

RESUME EVIDENCE:
{context}

Rank the candidates based on job-relevant evidence only.
Return valid JSON and nothing else.
"""

        response = self.llm.generate(
            [
                {"role": "system", "content": SYSTEM_PROMPT + "\n" + RANKING_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=2500
        )

        data = self._parse_json(response)

        candidates = data.get("candidates", [])
        candidates.sort(
            key=lambda x: float(x.get("score", 0)),
            reverse=True
        )

        return candidates

    @staticmethod
    def _parse_json(text):
        text = text.strip()

        if text.startswith("```"):
            text = text.replace("```json", "", 1)
            text = text.replace("```", "")
            text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start:end + 1])

            raise ValueError(
                "The LLM did not return valid JSON for ranking."
            )

    @staticmethod
    def _unique_sources(sources):
        seen = set()
        unique = []

        for source in sources:
            key = (
                source["candidate"],
                source["file"],
                source["page"]
            )
            if key not in seen:
                seen.add(key)
                unique.append(source)

        return unique

    def reset(self):
        self.vector_store.clear()
        self.resume_records = []

        for filename in ["resume.index", "metadata.json"]:
            path = VECTOR_DIR / filename
            if path.exists():
                path.unlink()
