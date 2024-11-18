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
    def validate_analysis_data(data: Dict) -> Dict:
        """Validate and normalize analysis data"""
        required_fields = {
            "summary": dict,
            "sentiment": dict,
            "topics": dict,
            "content_quality": dict,
            "metadata": dict
        }
        
        validated = {}
        for field, field_type in required_fields.items():
            value = data.get(field, {})
            if not isinstance(value, field_type):
                value = field_type()
            validated[field] = value
        
        return validated

    @staticmethod
    def create_summary_visualization(data: List[tuple]) -> go.Figure:
        """Create summary visualization from database records"""
        # Convert tuple data to DataFrame
        df = pd.DataFrame.from_records(data, columns=['id', 'url', 'content', 'analysis', 'created_at'])
        
        # Parse the JSON string in analysis column
        def parse_analysis(x):
            if isinstance(x, str):
                return json.loads(x)
            elif isinstance(x, dict):
                return x
            return {}
        
        df['analysis'] = df['analysis'].apply(parse_analysis)
        
        # Extract sentiment from nested structure
        def extract_sentiment(x):
            try:
                return x.get('sentiment', {}).get('label', 'neutral').lower()
            except:
                return 'neutral'
        
        sentiments = df['analysis'].apply(extract_sentiment).value_counts()
        
        fig = go.Figure()
        
        # Add sentiment distribution
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
        # Convert tuple data to DataFrame
        df = pd.DataFrame.from_records(data, columns=['id', 'url', 'content', 'analysis', 'created_at'])
        
        # Ensure created_at is datetime
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Group by date and count
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
        df = pd.DataFrame.from_records(data, columns=['id', 'url', 'content', 'analysis', 'created_at'])
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Apply date filters if provided
        if start_date:
            df = df[df['created_at'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['created_at'] <= pd.to_datetime(end_date)]
        
        # Format data for export
        export_data = []
        for _, row in df.iterrows():
            analysis = row['analysis']
            if isinstance(analysis, str):
                try:
                    analysis = json.loads(analysis)
                except:
                    analysis = {}
            
            export_record = {
                "id": row['id'],
                "url": row['url'],
                "content": DataProcessor.clean_text(row['content']),
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
            analysis = record.get('analysis', {})
            if isinstance(analysis, str):
                try:
                    analysis = json.loads(analysis)
                except:
                    analysis = {}
            
            # Extract features and labels
            training_record = {
                "features": {
                    "text": DataProcessor.clean_text(record.get('content', '')),
                    "url": record.get('url', ''),
                    "length": len(record.get('content', '')),
                    "has_technical_terms": bool(re.search(r'\b(api|function|class|method|algorithm)\b', 
                                                        record.get('content', '').lower())),
                },
                "labels": {
                    "sentiment": analysis.get('sentiment', {}).get('label', 'neutral'),
                    "category": analysis.get('categories', {}).get('primary', 'unknown'),
                    "content_type": analysis.get('content_type', {}).get('type', 'unknown')
                },
                "metadata": {
                    "confidence_scores": {
                        "sentiment": analysis.get('sentiment', {}).get('confidence', 0.0),
                        "category": analysis.get('categories', {}).get('confidence', 0.0)
                    },
                    "timestamp": record.get('metadata', {}).get('created_at', datetime.utcnow().isoformat())
                }
            }
            training_data.append(training_record)
        
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
