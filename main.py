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
            # Validate URL
            if not processor.validate_url(url):
                st.error("Invalid URL format. Please enter a valid URL.")
                return
            
            # Scrape content
            result = scraper.scrape_single_url(url)
            
            if result["success"]:
                # Clean content
                cleaned_content = processor.clean_text(result["content"])
                
                # Analyze with LLM
                analysis = llm.analyze_content(cleaned_content, url)
                if "error" in analysis:
                    st.error(f"Analysis failed: {analysis['error']}")
                    return
                    
                categorization = llm.categorize_content(cleaned_content)
                if "error" in categorization:
                    st.error(f"Categorization failed: {categorization['error']}")
                    return
                
                # Extract training data
                training_data = llm.extract_training_data(cleaned_content, url)
                
                # Save to database
                combined_analysis = {
                    "content_analysis": analysis,
                    "categorization": categorization,
                    "training_data": training_data
                }
                db.save_data(url, cleaned_content, combined_analysis)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Content Analysis")
                    st.json(analysis)
                
                with col2:
                    st.markdown("### Content Categorization")
                    st.json(categorization)
                
                # Download button for the analysis
                if st.button("Download Analysis"):
                    export_dict = {
                        "url": url,
                        "content": cleaned_content,
                        "analysis": combined_analysis,
                        "metadata": {
                            "timestamp": datetime.utcnow().isoformat(),
                            "version": "1.0"
                        }
                    }
                    json_str = json.dumps(export_dict, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            else:
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
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            results = []
            for idx, url in enumerate(valid_urls):
                progress = (idx + 1) / len(valid_urls)
                progress_bar.progress(progress)
                progress_text.text(f"Processing URL {idx + 1} of {len(valid_urls)}")
                
                result = scraper.scrape_single_url(url)
                if result["success"]:
                    cleaned_content = processor.clean_text(result["content"])
                    analysis = llm.analyze_content(cleaned_content, url)
                    if "error" in analysis:
                        st.warning(f"Analysis failed for {url}: {analysis['error']}")
                        continue
                        
                    categorization = llm.categorize_content(cleaned_content)
                    if "error" in categorization:
                        st.warning(f"Categorization failed for {url}: {categorization['error']}")
                        continue
                        
                    training_data = llm.extract_training_data(cleaned_content, url)
                    
                    combined_analysis = {
                        "content_analysis": analysis,
                        "categorization": categorization,
                        "training_data": training_data
                    }
                    db.save_data(url, cleaned_content, combined_analysis)
                results.append(result)
            
            # Display batch results
            batch_stats = processor.process_batch_results(results)
            st.success("Batch processing completed!")
            st.json(batch_stats)
            
            # Download button for batch results
            if st.button("Download Batch Results"):
                export_data = processor.format_data_for_export(
                    db.get_all_data(),
                    start_date=(datetime.utcnow() - timedelta(hours=1)).isoformat()
                )
                json_str = json.dumps(export_data, indent=2)
                st.download_button(
                    label="Download JSON",
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
            export_data = processor.format_data_for_export(
                data,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat()
            )
            
            # Display recent entries
            st.markdown("### Recent Entries")
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
                
                json_str = json.dumps(download_data, indent=2)
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

if __name__ == "__main__":
    main()
