import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict

class DataProcessor:
    @staticmethod
    def create_summary_visualization(data: List[Dict]) -> go.Figure:
        df = pd.DataFrame(data)
        fig = go.Figure()
        
        # Add sentiment distribution
        sentiments = df['analysis'].apply(lambda x: x.get('sentiment', 'unknown')).value_counts()
        fig.add_trace(go.Pie(
            labels=sentiments.index,
            values=sentiments.values,
            name="Sentiment Distribution"
        ))
        
        fig.update_layout(
            title="Content Analysis Summary",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return fig

    @staticmethod
    def create_timeline_visualization(data: List[Dict]) -> go.Figure:
        df = pd.DataFrame(data)
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        fig = px.line(
            df.groupby(df['created_at'].dt.date).size().reset_index(),
            x='created_at',
            y=0,
            title="Data Collection Timeline"
        )
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return fig

    @staticmethod
    def process_batch_results(results: List[Dict]) -> Dict:
        total = len(results)
        successful = sum(1 for r in results if r.get('success', False))
        failed = total - successful
        
        return {
            'total': total,
            'successful': successful,
            'failed': failed,
            'success_rate': (successful / total * 100) if total > 0 else 0
        }
