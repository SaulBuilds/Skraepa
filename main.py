import streamlit as st
import pandas as pd
from scraper.web_scraper import WebScraper
from llm.openai_handler import LLMHandler
from db.database import Database
from utils.data_processor import DataProcessor
import json
from typing import List, Dict, Any
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Data Harvesting Platform",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
scraper = WebScraper()

# Test LLM integration
try:
    llm = LLMHandler()
    llm_test = llm.test_connection()
    if "error" in llm_test:
        st.error(f"LLM Integration Error: {llm_test['error']}")
        st.stop()
except Exception as e:
    st.error(f"Failed to initialize LLM: {str(e)}")
    st.stop()

db = Database()
processor = DataProcessor()

# Load custom CSS
with open("styles/custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def main():
    st.title("Data Harvesting & Analysis Platform")
    
    # Sidebar
    st.sidebar.header("Control Panel")
    mode = st.sidebar.radio("Select Mode", ["Single URL", "Batch Processing", "Dashboard"])
    
    if mode == "Single URL":
        single_url_mode()
    elif mode == "Batch Processing":
        batch_processing_mode()
    else:
        dashboard_mode()

def single_url_mode():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    url = st.text_input("Enter URL to analyze")
    
    if st.button("Analyze"):
        with st.spinner("Processing URL..."):
            # Scrape content
            result = scraper.scrape_single_url(url)
            
            if result["success"]:
                # Analyze with LLM
                analysis = llm.analyze_content(result["content"])
                if "error" in analysis:
                    st.error(f"Analysis failed: {analysis['error']}")
                    return
                    
                categorization = llm.categorize_content(result["content"])
                if "error" in categorization:
                    st.error(f"Categorization failed: {categorization['error']}")
                    return
                
                # Save to database
                combined_analysis = {
                    "content_analysis": analysis,
                    "categorization": categorization
                }
                db.save_data(url, result["content"], combined_analysis)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Content Analysis")
                    st.json(analysis)
                
                with col2:
                    st.markdown("### Content Categorization")
                    st.json(categorization)
            else:
                st.error(f"Failed to process URL: {result.get('error', 'Unknown error')}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def batch_processing_mode():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload URLs (one per line)", type=["txt"])
    
    if uploaded_file is not None:
        urls = uploaded_file.getvalue().decode().splitlines()
        urls = [url.strip() for url in urls if url.strip()]
        
        if st.button("Process Batch"):
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            results = []
            for idx, url in enumerate(urls):
                progress = (idx + 1) / len(urls)
                progress_bar.progress(progress)
                progress_text.text(f"Processing URL {idx + 1} of {len(urls)}")
                
                result = scraper.scrape_single_url(url)
                if result["success"]:
                    analysis = llm.analyze_content(result["content"])
                    if "error" in analysis:
                        st.warning(f"Analysis failed for {url}: {analysis['error']}")
                        continue
                        
                    categorization = llm.categorize_content(result["content"])
                    if "error" in categorization:
                        st.warning(f"Categorization failed for {url}: {categorization['error']}")
                        continue
                        
                    combined_analysis = {
                        "content_analysis": analysis,
                        "categorization": categorization
                    }
                    db.save_data(url, result["content"], combined_analysis)
                results.append(result)
            
            # Display batch results
            batch_stats = processor.process_batch_results(results)
            st.success("Batch processing completed!")
            st.json(batch_stats)
    
    st.markdown('</div>', unsafe_allow_html=True)

def dashboard_mode():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Data Analysis Dashboard")
    
    # Fetch all data
    data = db.get_all_data()
    
    if data:
        try:
            # Create visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                summary_fig = processor.create_summary_visualization(data)
                st.plotly_chart(summary_fig, use_container_width=True)
            
            with col2:
                timeline_fig = processor.create_timeline_visualization(data)
                st.plotly_chart(timeline_fig, use_container_width=True)
            
            # Display recent entries
            st.markdown("### Recent Entries")
            df = pd.DataFrame.from_records(data, columns=['id', 'url', 'content', 'analysis', 'created_at'])
            display_df = df[['url', 'created_at']].copy()
            display_df['created_at'] = pd.to_datetime(display_df['created_at'])
            st.dataframe(display_df.sort_values('created_at', ascending=False).head(10))
        except Exception as e:
            st.error(f"Error processing dashboard data: {str(e)}")
    else:
        st.info("No data available yet. Start by analyzing some URLs!")
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
