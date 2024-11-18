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
import asyncio
from collections import defaultdict

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

async def process_url_with_progress(url: str, depth: int = 1, progress_text=None, progress_bar=None, current_idx=None, total=None) -> Dict:
    """Process a single URL with recursive scraping and progress tracking"""
    try:
        if progress_text:
            progress_text.text(f"Step 1/4: Scraping content (Depth: {depth})...")
        
        # Initialize URL tree for visualization
        url_tree = defaultdict(list)
        processed_urls = set()
        
        async def progress_callback(url: str, current_depth: int, total_processed: int):
            """Callback to update progress during recursive scraping"""
            if progress_text:
                progress_text.text(f"Scraping page {total_processed} at depth {current_depth}: {url}")
            if progress_bar:
                progress_bar.progress(min(1.0, total_processed / scraper.max_pages_per_domain))
        
        # Set progress callback
        scraper.set_progress_callback(progress_callback)
        
        # Use recursive scraping if depth > 1
        if depth > 1:
            results = await scraper.recursive_scrape(url)
            # Process each result and update URL tree
            processed_results = []
            for result in results:
                if result["success"]:
                    current_url = result["url"]
                    parent_url = None
                    # Find parent URL
                    for potential_parent, links in url_tree.items():
                        if current_url in links:
                            parent_url = potential_parent
                            break
                    if parent_url:
                        url_tree[parent_url].append(current_url)
                    else:
                        url_tree[url].append(current_url)
                    
                    # Process content with LLM and store in database
                    if progress_text:
                        progress_text.text(f"Processing content from: {current_url}")
                    
                    cleaned_content = processor.clean_text(result["content"])
                    raw_content = result["content"]
                    
                    # Analyze with LLM
                    analysis = llm.analyze_content(cleaned_content, current_url)
                    if "error" in analysis:
                        continue
                    
                    categorization = llm.categorize_content(cleaned_content)
                    if "error" in categorization:
                        continue
                    
                    # Save to database
                    try:
                        processing_metadata = {
                            "processing_timestamp": datetime.utcnow().isoformat(),
                            "processing_status": "completed",
                            "processing_steps": [
                                "content_scraped",
                                "content_cleaned",
                                "content_analyzed",
                                "content_categorized"
                            ],
                            "depth": depth,
                            "parent_url": parent_url
                        }
                        
                        db.save_data(current_url, cleaned_content, raw_content, analysis, processing_metadata)
                        processed_results.append({
                            "url": current_url,
                            "analysis": {
                                "content_analysis": analysis,
                                "categorization": categorization,
                                "raw_content": raw_content
                            }
                        })
                    except Exception as e:
                        logger.error(f"Failed to save data for {current_url}: {str(e)}")
            
            if progress_text:
                progress_text.text("Step 4/4: All pages processed!")
            
            return {
                "success": True,
                "url": url,
                "results": processed_results,
                "url_tree": dict(url_tree)
            }
        else:
            # Single URL processing
            result = await scraper.scrape_single_url(url)
            if not result["success"]:
                return {"success": False, "error": result.get("error", "Unknown error")}
            
            if progress_text:
                progress_text.text("Step 2/4: Cleaning content...")
            
            # Clean content
            cleaned_content = processor.clean_text(result["content"])
            raw_content = result["content"]
            
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
            
            # Save to database
            try:
                processing_metadata = {
                    "processing_timestamp": datetime.utcnow().isoformat(),
                    "processing_status": "completed",
                    "processing_steps": [
                        "content_scraped",
                        "content_cleaned",
                        "content_analyzed",
                        "content_categorized"
                    ],
                    "depth": depth
                }
                
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

def display_url_tree(url_tree: Dict[str, List[str]], container):
    """Display URL tree in a hierarchical format"""
    def _display_branch(url: str, children: List[str], level: int = 0):
        prefix = "  " * level + ("└── " if level > 0 else "")
        container.markdown(f"`{prefix}{url}`")
        for child in children:
            if child in url_tree:
                _display_branch(child, url_tree[child], level + 1)
            else:
                child_prefix = "  " * (level + 1) + "└── "
                container.markdown(f"`{child_prefix}{child}`")
    
    root_urls = [url for url in url_tree.keys() if not any(url in children for children in url_tree.values())]
    for root_url in root_urls:
        _display_branch(root_url, url_tree[root_url])

def main():
    st.markdown("<h1 style='font-weight: 900; font-size: 3.5em; margin-bottom: 0.2em;'>SKRAEPA</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-weight: 400; color: #7792E3;'>Data Harvesting & Analysis Platform</h2>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("Control Panel")
    mode = st.sidebar.radio("Select Mode", ["Single URL", "Batch Processing", "Settings", "Dashboard"])
    
    if mode == "Single URL":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        url = st.text_input("Enter URL to analyze")
        
        # Add depth selector
        depth = st.slider(
            "Scraping Depth",
            min_value=1,
            max_value=5,
            value=1,
            help="1 = Current page only, >1 = Include linked pages up to selected depth"
        )
        
        if st.button("Analyze"):
            if not processor.validate_url(url):
                st.error("Invalid URL format. Please enter a valid URL.")
                return
            
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            with st.spinner("Processing URL..."):
                result = asyncio.run(process_url_with_progress(url, depth, progress_text, progress_bar, 0, 1))
                
                if result["success"]:
                    progress_text.text("Processing completed!")
                    progress_bar.progress(1.0)
                    
                    if depth > 1 and "url_tree" in result:
                        # Display URL tree
                        st.markdown("### Scraped Pages Structure")
                        display_url_tree(result["url_tree"], st)
                        
                        # Display aggregated results
                        st.markdown("### Analysis Results")
                        for page_result in result["results"]:
                            with st.expander(f"Results for {page_result['url']}"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("#### Content Analysis")
                                    st.json(page_result["analysis"]["content_analysis"])
                                with col2:
                                    st.markdown("#### Content Categorization")
                                    st.json(page_result["analysis"]["categorization"])
                    else:
                        # Display single page results
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
                        "depth": depth,
                        "content": result.get("analysis", {}).get("raw_content", ""),
                        "analysis": result.get("analysis", {}),
                        "url_tree": result.get("url_tree", {}),
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
    
    elif mode == "Settings":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Scraper Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_depth = st.slider(
                "Maximum Scraping Depth",
                min_value=1,
                max_value=5,
                value=3,
                help="Maximum depth for recursive scraping. Higher values will scrape more pages but take longer."
            )
            
            max_pages = st.number_input(
                "Maximum Pages per Domain",
                min_value=1,
                max_value=500,
                value=100,
                help="Maximum number of pages to scrape from a single domain."
            )
            
            stay_within_domain = st.toggle(
                "Stay Within Domain",
                value=True,
                help="When enabled, only scrape pages from the same domain as the starting URL."
            )
        
        with col2:
            timeout = st.slider(
                "Request Timeout (seconds)",
                min_value=5,
                max_value=60,
                value=30,
                help="Maximum time to wait for a page to respond."
            )
            
            max_concurrent = st.slider(
                "Concurrent Requests",
                min_value=1,
                max_value=10,
                value=5,
                help="Maximum number of simultaneous page requests. Higher values may be faster but could overwhelm servers."
            )
            
            crawl_strategy = st.selectbox(
                "Crawling Strategy",
                options=["breadth-first", "depth-first"],
                help="Breadth-first explores all pages at the current depth before going deeper. Depth-first explores each path to its maximum depth before backtracking."
            )
        
        # Save configuration
        if st.button("Save Configuration"):
            try:
                scraper.max_depth = max_depth
                scraper.max_pages_per_domain = max_pages
                scraper.stay_within_domain = stay_within_domain
                scraper.timeout = timeout
                scraper.max_concurrent = max_concurrent
                scraper.crawl_strategy = crawl_strategy
                
                st.success("Configuration saved successfully!")
            except Exception as e:
                st.error(f"Failed to save configuration: {str(e)}")
        
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