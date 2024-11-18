import os
import json
from openai import OpenAI, OpenAIError
from typing import Dict, Any, Optional
from datetime import datetime

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

    def validate_response(self, response_json: Dict) -> bool:
        """Validate the structure of LLM response"""
        try:
            if not isinstance(response_json, dict):
                return False
            # Add basic validation checks
            if "error" in response_json:
                return False
            # Check for required fields based on the response type
            if "metadata" in response_json:
                if not all(k in response_json["metadata"] for k in ["timestamp", "model_version"]):
                    return False
            return True
        except Exception:
            return False

    def test_connection(self) -> Dict[str, Any]:
        """Test the OpenAI connection with a simple prompt"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a testing system. Return this in JSON format with a 'status' field containing 'ok'"
                    },
                    {
                        "role": "user",
                        "content": "Return status"
                    }
                ],
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            return result if self.validate_response(result) else {"error": "Invalid response format"}
        except OpenAIError as e:
            return {"error": f"OpenAI API Error: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

    def analyze_content(self, text: str, url: Optional[str] = None) -> Dict[str, Any]:
        """Analyze content with enhanced structure and metadata"""
        if not text or not isinstance(text, str):
            return {"error": "Invalid input: text must be a non-empty string"}
            
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a content analysis expert. Return this analysis in JSON format with the exact structure specified below:
                        {
                            "summary": {
                                "short": "2-3 sentence summary",
                                "key_points": ["point1", "point2", "point3"]
                            },
                            "sentiment": {
                                "label": "positive/negative/neutral",
                                "confidence": 0.0 to 1.0,
                                "analysis": "Brief explanation"
                            },
                            "topics": {
                                "main_topic": "string",
                                "subtopics": ["topic1", "topic2", "topic3"],
                                "confidence": 0.0 to 1.0
                            },
                            "content_quality": {
                                "readability_score": 0.0 to 1.0,
                                "technical_level": "basic/intermediate/advanced",
                                "audience": "general/technical/academic"
                            }
                        }"""
                    },
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            if not self.validate_response(result):
                return {"error": "Invalid response format from OpenAI"}
            
            # Add metadata
            result["metadata"] = {
                "timestamp": datetime.utcnow().isoformat(),
                "source_url": url,
                "model_version": self.model,
                "processing_status": "completed"
            }
            
            return result
            
        except OpenAIError as e:
            return {"error": f"OpenAI API Error: {str(e)}"}
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse OpenAI response: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

    def categorize_content(self, text: str) -> Dict[str, Any]:
        """Categorize content with enhanced structure"""
        if not text or not isinstance(text, str):
            return {"error": "Invalid input: text must be a non-empty string"}
            
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a content categorization expert. Return this analysis in JSON format with the exact structure specified below:
                        {
                            "categories": {
                                "primary": "string",
                                "secondary": ["category1", "category2", "category3"],
                                "confidence": 0.0 to 1.0
                            },
                            "content_type": {
                                "type": "article/blog/news/research/other",
                                "confidence": 0.0 to 1.0,
                                "attributes": ["attribute1", "attribute2"]
                            },
                            "domain_specific": {
                                "field": "string",
                                "relevance_score": 0.0 to 1.0,
                                "key_terms": ["term1", "term2", "term3"]
                            }
                        }"""
                    },
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            if not self.validate_response(result):
                return {"error": "Invalid response format from OpenAI"}
            return result
            
        except OpenAIError as e:
            return {"error": f"OpenAI API Error: {str(e)}"}
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse OpenAI response: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

    def extract_training_data(self, text: str, url: Optional[str] = None) -> Dict[str, Any]:
        """Extract and format data specifically for model training"""
        if not text or not isinstance(text, str):
            return {"error": "Invalid input: text must be a non-empty string"}
        
        try:
            # Get both analysis and categorization
            analysis = self.analyze_content(text, url)
            if "error" in analysis:
                return {"error": f"Analysis failed: {analysis['error']}"}
                
            categories = self.categorize_content(text)
            if "error" in categories:
                return {"error": f"Categorization failed: {categories['error']}"}
            
            # Format data for training
            training_data = {
                "raw_content": text,
                "processed_content": {
                    "summary": analysis.get("summary", {}),
                    "topics": analysis.get("topics", {}),
                    "sentiment": analysis.get("sentiment", {})
                },
                "classifications": {
                    "categories": categories.get("categories", {}),
                    "content_type": categories.get("content_type", {}),
                    "domain": categories.get("domain_specific", {})
                },
                "metadata": {
                    "source_url": url,
                    "timestamp": datetime.utcnow().isoformat(),
                    "model_version": self.model,
                    "quality_metrics": analysis.get("content_quality", {}),
                    "confidence_scores": {
                        "sentiment": analysis.get("sentiment", {}).get("confidence", 0.0),
                        "topics": analysis.get("topics", {}).get("confidence", 0.0),
                        "categories": categories.get("categories", {}).get("confidence", 0.0)
                    }
                }
            }
            
            return training_data
            
        except Exception as e:
            return {"error": f"Failed to extract training data: {str(e)}"}
