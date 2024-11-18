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

# Custom CSS
st.markdown("""
<style>
.stApp {
    background-image: linear-gradient(
        45deg,
        rgba(14, 17, 23, 0.98),
        rgba(14, 17, 23, 0.96)
    );
    background-size: cover;
}

.card {
    background: linear-gradient(
        145deg,
        rgba(30, 35, 41, 0.95),
        rgba(30, 35, 41, 0.85)
    );
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(5px);
}

.progress-bar {
    height: 6px;
    background: linear-gradient(90deg, #7792E3, #4FD1C5);
    border-radius: 3px;
}

.visualization {
    border: 1px solid rgba(119, 146, 227, 0.2);
    border-radius: 8px;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

def process_url_with_progress(url: str, progress_text=None, progress_bar=None, current_idx=None, total=None) -> Dict:
    """Process a single URL with progress indicators"""
    if progress_text:
        progress_text.text("Step 1/4: Scraping content...")
    
    # Scrape content
    result = scraper.scrape_single_url(url)
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
        progress_text.text("Step 4/4: Extracting training data...")
    
    # Extract training data
    training_data = llm.extract_training_data(cleaned_content, url)
    
    # Update progress
    if progress_bar is not None and total is not None and current_idx is not None:
        progress = (current_idx + 1) / total
        progress_bar.progress(progress)
    
    # Combine all data
    combined_analysis = {
        "content_analysis": analysis,
        "categorization": categorization,
        "training_data": training_data,
        "raw_content": raw_content
    }
    
    # Save to database
    db.save_data(url, cleaned_content, combined_analysis)
    
    return {
        "success": True,
        "url": url,
        "analysis": combined_analysis
    }

def single_url_mode():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    url = st.text_input("Enter URL to analyze")
    
    if st.button("Analyze"):
        # Validate URL
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
                    "content": result["analysis"].get("raw_content", ""),
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

def batch_processing_mode():
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
                # Update overall progress
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
                
                # Reset current URL progress
                current_url_progress.progress(0)
            
            # Complete overall progress
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
            
            # Download button for batch results
            if successful > 0:
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
    
    st.markdown('</div>', unsafe_allow_html=True)

def dashboard_mode():
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
            start_date_str = start_date.strftime('%Y-%m-%d') if isinstance(start_date, datetime) else None
            end_date_str = end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime) else None
            
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
            st.error(f"Error processing dashboard data: {str(e)}")
    else:
        st.info("No data available yet. Start by analyzing some URLs!")
    
    st.markdown('</div>', unsafe_allow_html=True)

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

if __name__ == "__main__":
    main()