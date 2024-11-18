import os
import json
from openai import OpenAI, OpenAIError
from typing import Dict, Any, Optional

class LLMHandler:
    def __init__(self):
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.model = "gpt-4o"
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        try:
            self.client = OpenAI(api_key=self.api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {str(e)}")

    def test_connection(self) -> Dict[str, Any]:
        """Test the OpenAI connection with a simple prompt"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": "Return a simple JSON with key 'status' and value 'ok'"}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except OpenAIError as e:
            return {"error": f"OpenAI API Error: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

    def analyze_content(self, text: str) -> Dict[str, Any]:
        if not text or not isinstance(text, str):
            return {"error": "Invalid input: text must be a non-empty string"}
            
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
            
            result = json.loads(response.choices[0].message.content)
            if not isinstance(result, dict):
                raise ValueError("Invalid response format from OpenAI")
            return result
            
        except OpenAIError as e:
            return {"error": f"OpenAI API Error: {str(e)}"}
        except json.JSONDecodeError:
            return {"error": "Failed to parse OpenAI response"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

    def categorize_content(self, text: str) -> Dict[str, Any]:
        if not text or not isinstance(text, str):
            return {"error": "Invalid input: text must be a non-empty string"}
            
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
            
            result = json.loads(response.choices[0].message.content)
            if not isinstance(result, dict):
                raise ValueError("Invalid response format from OpenAI")
            return result
            
        except OpenAIError as e:
            return {"error": f"OpenAI API Error: {str(e)}"}
        except json.JSONDecodeError:
            return {"error": "Failed to parse OpenAI response"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}
