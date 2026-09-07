import os
from pydantic import BaseModel
from typing import Optional, List
from google import genai
from google.genai import types

class FactOutput(BaseModel):
    text: str
    value: Optional[str]
    excerpt: str
    page: Optional[int]

class FactList(BaseModel):
    facts: list[FactOutput]

print("Test genai ready.")
