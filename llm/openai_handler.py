import os
import json
from openai import OpenAI
from typing import Dict, Any

class LLMHandler:
    def __init__(self):
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.model = "gpt-4o"
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def analyze_content(self, text: str) -> Dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """Analyze the following text and provide:
                        1. A summary (max 3 sentences)
                        2. Main topics (max 5)
                        3. Sentiment (positive/negative/neutral)
                        4. Key insights (max 3)
                        Format the response as JSON."""
                    },
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": str(e)}

    def categorize_content(self, text: str) -> Dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """Categorize the content into:
                        1. Primary category
                        2. Subcategories (max 3)
                        3. Content type (article/blog/news/other)
                        Format the response as JSON."""
                    },
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": str(e)}
