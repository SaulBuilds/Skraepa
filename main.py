import streamlit as st
import pandas as pd
from scraper.web_scraper import WebScraper
from llm.openai_handler import LLMHandler
from db.database import Database
from utils.data_processor import DataProcessor
import json
from typing import List, Dict, Any
from datetime import datetime, timedelta
import io
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load custom CSS
def load_css():
    with open("styles/custom.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title="Data Harvesting Platform",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom styling
try:
    load_css()
except Exception as e:
    logger.error(f"Failed to load custom CSS: {str(e)}")
    st.warning("Custom styling could not be loaded.")

# Initialize components with error handling
try:
    scraper = WebScraper()
    logger.info("WebScraper initialized successfully")
except Exception as e:
    st.error(f"Failed to initialize WebScraper: {str(e)}")
    st.stop()

try:
    llm = LLMHandler()
    llm_test = llm.test_connection()
    if "error" in llm_test:
        st.error(f"LLM Integration Error: {llm_test['error']}")
        st.stop()
    logger.info("LLM Handler initialized successfully")
except Exception as e:
    st.error(f"Failed to initialize LLM: {str(e)}")
    st.stop()

try:
    db = Database()
    # Test database connection
    with db.conn.cursor() as cur:
        cur.execute("SELECT 1")
    logger.info("Database initialized and connected successfully")
except Exception as e:
    st.error(f"Failed to initialize Database: {str(e)}")
    st.stop()

processor = DataProcessor()

def process_url_with_progress(url: str, progress_text=None, progress_bar=None, current_idx=None, total=None) -> Dict:
    """Process a single URL with progress indicators"""
    try:
        if progress_text:
            progress_text.text("Step 1/4: Scraping content...")
        
        # Scrape content
        result = scraper.scrape_url(url)
        if not result["success"]:
            return {"success": False, "error": result.get("error", "Unknown error")}
        
        if progress_text:
            progress_text.text("Step 2/4: Cleaning content...")
        
        # Clean content
        cleaned_content = processor.clean_text(result["content"])
        raw_content = result["content"]  # Store raw content
        
        if progress_text:
            progress_text.text("Step 3/4: Analyzing content...")
        
        # Analyze with LLM
        analysis = llm.analyze_content(cleaned_content, url)
        if "error" in analysis:
            return {"success": False, "error": f"Analysis failed: {analysis['error']}"}
        
        categorization = llm.categorize_content(cleaned_content)
        if "error" in categorization:
            return {"success": False, "error": f"Categorization failed: {categorization['error']}"}
        
        if progress_text:
            progress_text.text("Step 4/4: Saving data...")
        
        # Prepare processing metadata
        processing_metadata = {
            "processing_timestamp": datetime.utcnow().isoformat(),
            "processing_status": "completed",
            "processing_steps": [
                "content_scraped",
                "content_cleaned",
                "content_analyzed",
                "content_categorized"
            ]
        }
        
        # Save to database with raw content
        try:
            db.save_data(url, cleaned_content, raw_content, analysis, processing_metadata)
        except Exception as e:
            logger.error(f"Failed to save data: {str(e)}")
            return {"success": False, "error": f"Failed to save data: {str(e)}"}
        
        # Update progress
        if progress_bar is not None and total is not None and current_idx is not None:
            progress = (current_idx + 1) / total
            progress_bar.progress(progress)
        
        return {
            "success": True,
            "url": url,
            "analysis": {
                "content_analysis": analysis,
                "categorization": categorization,
                "raw_content": raw_content
            }
        }
    except Exception as e:
        logger.error(f"Error processing URL {url}: {str(e)}")
        return {"success": False, "error": str(e)}

def main():
    st.title("Data Harvesting & Analysis Platform")
    
    # Sidebar
    st.sidebar.header("Control Panel")
    mode = st.sidebar.radio("Select Mode", ["Single URL", "Batch Processing", "Dashboard"])
    
    if mode == "Single URL":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        url = st.text_input("Enter URL to analyze")
        
        if st.button("Analyze"):
            if not processor.validate_url(url):
                st.error("Invalid URL format. Please enter a valid URL.")
                return
            
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            with st.spinner("Processing URL..."):
                result = process_url_with_progress(url, progress_text, progress_bar, 0, 1)
                
                if result["success"]:
                    progress_text.text("Processing completed!")
                    progress_bar.progress(1.0)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Content Analysis")
                        st.json(result["analysis"]["content_analysis"])
                    
                    with col2:
                        st.markdown("### Content Categorization")
                        st.json(result["analysis"]["categorization"])
                    
                    # Download button for the analysis
                    export_dict = {
                        "url": url,
                        "content": result["analysis"]["raw_content"],
                        "analysis": result["analysis"],
                        "metadata": {
                            "timestamp": datetime.utcnow().isoformat(),
                            "version": "1.0"
                        }
                    }
                    
                    json_str = json.dumps(export_dict, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="Download Analysis",
                        data=json_str,
                        file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    progress_text.text("Processing failed!")
                    st.error(f"Failed to process URL: {result.get('error', 'Unknown error')}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif mode == "Batch Processing":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload URLs (one per line)", type=["txt"])
        
        if uploaded_file is not None:
            urls = uploaded_file.getvalue().decode().splitlines()
            urls = [url.strip() for url in urls if url.strip()]
            
            # Validate URLs
            invalid_urls = [url for url in urls if not processor.validate_url(url)]
            if invalid_urls:
                st.warning(f"Found {len(invalid_urls)} invalid URLs. They will be skipped.")
            
            valid_urls = [url for url in urls if processor.validate_url(url)]
            
            if st.button("Process Batch"):
                overall_progress = st.progress(0)
                current_url_progress = st.progress(0)
                status_text = st.empty()
                detail_text = st.empty()
                
                results = []
                for idx, url in enumerate(valid_urls):
                    overall_progress.progress((idx) / len(valid_urls))
                    status_text.text(f"Processing URL {idx + 1} of {len(valid_urls)}: {url}")
                    
                    result = process_url_with_progress(
                        url,
                        detail_text,
                        current_url_progress,
                        idx,
                        len(valid_urls)
                    )
                    results.append(result)
                    
                    current_url_progress.progress(0)
                
                overall_progress.progress(1.0)
                status_text.text("Batch processing completed!")
                detail_text.text("")
                
                # Display batch results
                successful = sum(1 for r in results if r.get("success", False))
                failed = len(results) - successful
                
                st.success(f"Processed {len(results)} URLs: {successful} successful, {failed} failed")
                
                if failed > 0:
                    st.error("Failed URLs:")
                    for result in results:
                        if not result.get("success", False):
                            st.write(f"- {result.get('url', 'Unknown URL')}: {result.get('error', 'Unknown error')}")
                
                if successful > 0:
                    try:
                        export_data = processor.format_data_for_export(
                            db.get_all_data(),
                            start_date=(datetime.utcnow() - timedelta(hours=1)).isoformat()
                        )
                        
                        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
                        st.download_button(
                            label="Download Batch Results",
                            data=json_str,
                            file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    except Exception as e:
                        st.error(f"Failed to prepare download: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:  # Dashboard mode
        try:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Data Analysis Dashboard")
            
            # Date range filter
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
            with col2:
                end_date = st.date_input("End Date", datetime.now())
            
            # Fetch all data
            data = db.get_all_data()
            
            if data:
                try:
                    # Create visualizations
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        summary_fig = processor.create_summary_visualization(data)
                        st.plotly_chart(summary_fig, use_container_width=True)
                    
                    with viz_col2:
                        timeline_fig = processor.create_timeline_visualization(data)
                        st.plotly_chart(timeline_fig, use_container_width=True)
                    
                    # Format data for display and export
                    start_date_str = start_date.strftime('%Y-%m-%d')
                    end_date_str = end_date.strftime('%Y-%m-%d')
                    
                    export_data = processor.format_data_for_export(
                        data,
                        start_date=start_date_str,
                        end_date=end_date_str
                    )
                    
                    # Display recent entries
                    st.markdown("### Recent Entries")
                    if export_data:
                        df = pd.DataFrame(export_data)
                        display_df = pd.DataFrame({
                            'URL': df['url'],
                            'Created At': pd.to_datetime(df['metadata'].apply(lambda x: x['created_at']))
                        })
                        st.dataframe(display_df.sort_values('Created At', ascending=False).head(10))
                        
                        # Download options
                        st.markdown("### Download Options")
                        download_format = st.selectbox(
                            "Select Format",
                            ["Full Dataset", "Training Data Only"]
                        )
                        
                        if st.button("Download Data"):
                            if download_format == "Training Data Only":
                                download_data = processor.format_training_data(export_data)
                            else:
                                download_data = export_data
                            
                            json_str = json.dumps(download_data, indent=2, ensure_ascii=False)
                            st.download_button(
                                label="Download JSON",
                                data=json_str,
                                file_name=f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                
                except Exception as e:
                    logger.error(f"Error in dashboard visualization: {str(e)}")
                    st.error(f"Error processing dashboard data: {str(e)}")
            else:
                st.info("No data available yet. Start by analyzing some URLs!")
            
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            logger.error(f"Error in dashboard mode: {str(e)}")
            st.error(f"Failed to load dashboard: {str(e)}")

if __name__ == "__main__":
    main()
