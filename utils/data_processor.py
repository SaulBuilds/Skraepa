import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Union, Optional
import json
from datetime import datetime
import re
from urllib.parse import urlparse

class DataProcessor:
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text

    @staticmethod
    def validate_llm_output(data: Dict) -> bool:
        """Validate LLM output format"""
        required_fields = {
            "content_analysis": {
                "summary": ["short", "key_points"],
                "sentiment": ["label", "confidence", "analysis"],
                "topics": ["main_topic", "subtopics", "confidence"],
                "content_quality": ["readability_score", "technical_level", "audience"]
            },
            "categorization": {
                "categories": ["primary", "secondary", "confidence"],
                "content_type": ["type", "confidence", "attributes"],
                "domain_specific": ["field", "relevance_score", "key_terms"]
            }
        }
        
        try:
            # Check main sections
            for section, fields in required_fields.items():
                if section not in data:
                    return False
                
                # Check nested fields
                for field, subfields in fields.items():
                    if field not in data[section]:
                        return False
                    
                    # Check required subfields
                    for subfield in subfields:
                        if subfield not in data[section][field]:
                            return False
            
            return True
        except Exception:
            return False

    @staticmethod
    def validate_analysis_data(data: Dict) -> Dict:
        """Validate and normalize analysis data"""
        required_fields = {
            "summary": {
                "short": str,
                "key_points": list
            },
            "sentiment": {
                "label": str,
                "confidence": float,
                "analysis": str
            },
            "topics": {
                "main_topic": str,
                "subtopics": list,
                "confidence": float
            },
            "content_quality": {
                "readability_score": float,
                "technical_level": str,
                "audience": str
            }
        }
        
        validated = {}
        for field, subfields in required_fields.items():
            field_data = data.get(field, {})
            validated_field = {}
            
            for subfield, expected_type in subfields.items():
                value = field_data.get(subfield)
                
                # Type conversion and validation
                if expected_type == float:
                    try:
                        value = float(value) if value is not None else 0.0
                        value = max(0.0, min(1.0, value))  # Ensure value is between 0 and 1
                    except (TypeError, ValueError):
                        value = 0.0
                elif expected_type == list:
                    value = value if isinstance(value, list) else []
                elif expected_type == str:
                    value = str(value) if value is not None else ""
                
                validated_field[subfield] = value
            
            validated[field] = validated_field
        
        return validated

    @staticmethod
    def create_summary_visualization(data: List[tuple]) -> go.Figure:
        """Create summary visualization from database records"""
        # Update column names to match database schema
        df = pd.DataFrame(data, columns=['id', 'url', 'content', 'raw_content', 'analysis', 'processing_metadata', 'created_at'])
        
        def parse_analysis(x):
            if isinstance(x, str):
                try:
                    return json.loads(x)
                except json.JSONDecodeError:
                    return {}
            elif isinstance(x, dict):
                return x
            return {}
        
        df['analysis'] = df['analysis'].apply(parse_analysis)
        
        def extract_sentiment(x):
            try:
                return x.get('content_analysis', {}).get('sentiment', {}).get('label', 'neutral').lower()
            except:
                return 'neutral'
        
        sentiments = df['analysis'].apply(extract_sentiment).value_counts()
        
        fig = go.Figure()
        
        fig.add_trace(go.Pie(
            labels=sentiments.index,
            values=sentiments.values,
            name="Sentiment Distribution",
            hole=0.3
        ))
        
        fig.update_layout(
            title="Content Analysis Summary",
            template="plotly_dark",
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return fig
    
    @staticmethod
    def create_timeline_visualization(data: List[tuple]) -> go.Figure:
        """Create timeline visualization from database records"""
        df = pd.DataFrame(data, columns=['id', 'url', 'content', 'analysis', 'created_at'])
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        daily_counts = df.groupby(df['created_at'].dt.date).size().reset_index()
        daily_counts.columns = ['date', 'count']
        
        fig = px.line(
            daily_counts,
            x='date',
            y='count',
            title="Data Collection Timeline"
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Entries",
            template="plotly_dark",
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return fig

    @staticmethod
    def format_data_for_export(data: List[tuple], start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict]:
        """Format data for export with optional date filtering"""
        df = pd.DataFrame(data, columns=['id', 'url', 'content', 'analysis', 'created_at'])
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Apply date filters if provided
        if start_date:
            df = df[df['created_at'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['created_at'] <= pd.to_datetime(end_date)]
        
        export_data = []
        for _, row in df.iterrows():
            # Parse analysis JSON
            try:
                analysis = json.loads(row['analysis']) if isinstance(row['analysis'], str) else row['analysis']
            except (json.JSONDecodeError, TypeError):
                analysis = {}
            
            # Validate and clean data
            export_record = {
                "id": int(row['id']),
                "url": str(row['url']),
                "content": DataProcessor.clean_text(str(row['content'])),
                "analysis": DataProcessor.validate_analysis_data(analysis),
                "metadata": {
                    "created_at": row['created_at'].isoformat(),
                    "processing_status": "completed",
                    "last_updated": datetime.utcnow().isoformat()
                }
            }
            export_data.append(export_record)
        
        return export_data

    @staticmethod
    def format_training_data(data: List[Dict]) -> List[Dict]:
        """Format data specifically for model training"""
        training_data = []
        for record in data:
            try:
                analysis = record.get('analysis', {})
                if isinstance(analysis, str):
                    analysis = json.loads(analysis)
                
                # Extract and validate features
                content = record.get('content', '')
                cleaned_content = DataProcessor.clean_text(content)
                
                training_record = {
                    "features": {
                        "text": cleaned_content,
                        "url": record.get('url', ''),
                        "length": len(cleaned_content),
                        "has_technical_terms": bool(re.search(
                            r'\b(api|function|class|method|algorithm)\b',
                            cleaned_content.lower()
                        )),
                    },
                    "labels": {
                        "sentiment": analysis.get('content_analysis', {}).get('sentiment', {}).get('label', 'neutral'),
                        "category": analysis.get('categorization', {}).get('categories', {}).get('primary', 'unknown'),
                        "content_type": analysis.get('categorization', {}).get('content_type', {}).get('type', 'unknown')
                    },
                    "metadata": {
                        "confidence_scores": {
                            "sentiment": float(analysis.get('content_analysis', {}).get('sentiment', {}).get('confidence', 0.0)),
                            "category": float(analysis.get('categorization', {}).get('categories', {}).get('confidence', 0.0))
                        },
                        "timestamp": record.get('metadata', {}).get('created_at', datetime.utcnow().isoformat())
                    }
                }
                training_data.append(training_record)
            except Exception as e:
                print(f"Error processing record: {str(e)}")
                continue
        
        return training_data
    
    @staticmethod
    def process_batch_results(results: List[Dict]) -> Dict[str, Union[int, float]]:
        """Process batch processing results"""
        total = len(results)
        successful = sum(1 for r in results if r.get('success', False))
        failed = total - successful
        
        return {
            'total': total,
            'successful': successful,
            'failed': failed,
            'success_rate': round((successful / total * 100) if total > 0 else 0, 2)
        }