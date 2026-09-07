import os
import json
import uuid
import tempfile
from pydantic import BaseModel, Field
from typing import List, Optional
import fitz  # PyMuPDF
from google import genai
from google.genai import types

class FactEvidence(BaseModel):
    document_name: str
    page: Optional[int]
    excerpt: str

class Fact(BaseModel):
    id: str
    text: str
    value: Optional[str]
    evidence: List[FactEvidence]

class FactOutput(BaseModel):
    text: str
    value: Optional[str]
    excerpt: str
    page: Optional[int]

class FactList(BaseModel):
    facts: list[FactOutput]

class Corroboration(BaseModel):
    type: str = "corroboration"
    fact_1: str
    fact_2: str
    explanation: str

class Contradiction(BaseModel):
    type: str
    fact_1: str
    fact_2: str
    explanation: str

class Failure(BaseModel):
    type: str
    description: str

class ReasoningOutput(BaseModel):
    corroborations: list[Corroboration]
    contradictions: list[Contradiction]
    failures: list[Failure]

class FactLayer:
    def __init__(self):
        self.facts: List[Fact] = []
        self.client = genai.Client()

    def extract_text_from_pdf(self, filepath: str) -> List[dict]:
        pages_text = []
        doc = fitz.open(filepath)
        for i in range(len(doc)):
            page = doc.load_page(i)
            text = page.get_text()
            if text.strip():
                pages_text.append({"page": i + 1, "text": text})
        return pages_text

    def process_document(self, filepath: str, filename: str):
        if filepath.endswith('.pdf') or filename.endswith('.pdf'):
            pages = self.extract_text_from_pdf(filepath)
        else:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                pages = [{"page": 1, "text": f.read()}]

        combined_text = ""
        for page_info in pages:
            combined_text += f"\n--- Page {page_info['page']} ---\n{page_info['text']}\n"

        prompt = f"""
        Extract the key financial, operational, and business facts from the following text.
        For each fact, extract:
        - "text": A concise description of the fact.
        - "value": The numerical or categorical value associated with the fact (if applicable).
        - "excerpt": A direct quote from the text supporting the fact.
        - "page": The page number where the fact was found (look at the --- Page X --- markers).
        
        Text:
        {combined_text}
        """
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-pro',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=FactList,
                )
            )
            
            extracted_data = json.loads(response.text)
            for item in extracted_data.get("facts", []):
                fact_id = str(uuid.uuid4())
                fact = Fact(
                    id=fact_id,
                    text=item.get("text", ""),
                    value=item.get("value"),
                    evidence=[FactEvidence(
                        document_name=filename,
                        page=item.get("page"),
                        excerpt=item.get("excerpt", "")
                    )]
                )
                self.facts.append(fact)
        except Exception as e:
            print(f"Error extracting facts: {e}")

        return {"status": "success", "message": f"Processed {filename}"}

    def get_facts(self) -> List[Fact]:
        return self.facts

    def run_reasoning(self):
        if not self.facts:
            return {"corroborations": [], "contradictions": [], "failures": []}

        facts_json = json.dumps([f.model_dump() for f in self.facts], indent=2)
        
        prompt = f"""
        Analyze the following list of facts extracted from documents.
        Identify relationships between these facts:
        1. "corroboration": Facts from different sources or parts of documents that support the same information.
        2. "genuine_contradiction": Facts that directly conflict with each other.
        3. "explained_by_context": Facts that seem to contradict but can be explained by context (e.g., different currencies, different time periods).
        
        Facts:
        {facts_json}
        """

        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-pro',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ReasoningOutput,
                )
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"Error reasoning: {e}")
            return {"corroborations": [], "contradictions": [], "failures": [{"type": "reasoning_failure", "description": str(e)}]}

fact_layer = FactLayer()
