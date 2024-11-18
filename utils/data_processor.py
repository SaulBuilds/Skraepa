import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Union
import json

class DataProcessor:
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
                return x.get('content_analysis', {}).get('sentiment', 'neutral').lower()
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
