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

def display_media_preview(media_data: Dict[str, Any], container):
    """Display media preview with download options"""
    if not media_data:
        return
    
    images = media_data.get('images', [])
    videos = media_data.get('videos', [])
    metadata = media_data.get('metadata', {})
    
    # Display statistics
    col1, col2, col3 = container.columns(3)
    col1.metric("Total Media", metadata.get('total_count', 0))
    col2.metric("Processed", metadata.get('processed_count', 0))
    col3.metric("Failed", metadata.get('failed_count', 0))
    
    # Display media content
    if images or videos:
        tabs = container.tabs(["Images", "Videos"])
        
        with tabs[0]:
            if images:
                for idx, img in enumerate(images):
                    col1, col2 = st.columns([2, 3])
                    with col1:
                        st.image(img['url'], caption=img.get('alt', ''), use_column_width=True)
                    with col2:
                        st.markdown(f"**URL:** `{img['url']}`")
                        st.markdown(f"**Size:** {img['metadata']['download_info'].get('size', 'Unknown')} bytes")
                        st.markdown(f"**Type:** {img['metadata']['download_info'].get('content_type', 'Unknown')}")
                        if img.get('alternative_sources'):
                            st.markdown("**Alternative Sources:**")
                            for src in img['alternative_sources']:
                                st.markdown(f"- {src if isinstance(src, str) else src['url']}")
            else:
                st.info("No images found")
        
        with tabs[1]:
            if videos:
                for idx, video in enumerate(videos):
                    st.markdown(f"### Video {idx + 1}")
                    col1, col2 = st.columns([2, 3])
                    with col1:
                        if 'youtube' in video['url'].lower():
                            st.video(video['url'])
                        else:
                            st.markdown(f"<video width='100%' controls><source src='{video['url']}'></video>", 
                                      unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"**URL:** `{video['url']}`")
                        st.markdown(f"**Type:** {video['metadata']['download_info'].get('content_type', 'Unknown')}")
                        st.markdown(f"**Size:** {video['metadata']['download_info'].get('size', 'Unknown')} bytes")
                        if video.get('alternative_sources'):
                            st.markdown("**Alternative Sources:**")
                            for src in video['alternative_sources']:
                                st.markdown(f"- {src if isinstance(src, str) else src['url']}")
            else:
                st.info("No videos found")

def display_media_stats(db_instance: Database, container):
    """Display media statistics in dashboard"""
    try:
        stats = db_instance.get_media_statistics()
        
        container.markdown("### Media Statistics")
        
        # Overall stats
        total = stats['total']
        cols = container.columns(4)
        cols[0].metric("Total Files", total['count'])
        cols[1].metric("Available", total['available'])
        cols[2].metric("Total Size", f"{total['total_size']/1024/1024:.2f} MB")
        cols[3].metric("File Formats", total['formats'])
        
        # Detailed stats by type
        container.markdown("#### By Media Type")
        col1, col2 = container.columns(2)
        
        # Images stats
        with col1:
            st.markdown("**Images**")
            img_stats = stats['images']
            st.markdown(f"""
            - Count: {img_stats['count']}
            - Available: {img_stats['available']}
            - Total Size: {img_stats['total_size']/1024/1024:.2f} MB
            - Formats: {img_stats['formats']}
            """)
        
        # Videos stats
        with col2:
            st.markdown("**Videos**")
            video_stats = stats['videos']
            st.markdown(f"""
            - Count: {video_stats['count']}
            - Available: {video_stats['available']}
            - Total Size: {video_stats['total_size']/1024/1024:.2f} MB
            - Formats: {video_stats['formats']}
            """)
    except Exception as e:
        container.error(f"Failed to load media statistics: {str(e)}")

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
                    
                    # Display media content
                    if "media" in result["analysis"]:
                        st.markdown("### Media Content")
                        display_media_preview(result["analysis"]["media"], st)
                    
                    # Download button for the analysis
                    export_dict = {
                        "url": url,
                        "analysis": result["analysis"]
                    }
                    
                    json_str = json.dumps(export_dict, indent=2)
                    st.download_button(
                        label="Download Analysis",
                        data=json_str,
                        file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    st.error(f"Error: {result.get('error', 'Unknown error occurred')}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif mode == "Batch Processing":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Batch URL Processing")
        
        urls_input = st.text_area(
            "Enter URLs (one per line)",
            height=150,
            help="Enter each URL on a new line"
        )
        
        if st.button("Process Batch"):
            urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
            
            if not urls:
                st.warning("Please enter at least one URL")
                return
            
            invalid_urls = [url for url in urls if not processor.validate_url(url)]
            if invalid_urls:
                st.error(f"Invalid URLs detected:\n{chr(10).join(invalid_urls)}")
                return
            
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            with st.spinner("Processing URLs..."):
                results = []
                for idx, url in enumerate(urls):
                    result = asyncio.run(process_url_with_progress(
                        url,
                        1,
                        progress_text,
                        progress_bar,
                        idx,
                        len(urls)
                    ))
                    results.append(result)
                
                # Display results
                stats = processor.process_batch_results(results)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total URLs", stats['total'])
                col2.metric("Successful", stats['successful'])
                col3.metric("Failed", stats['failed'])
                col4.metric("Success Rate", f"{stats['success_rate']}%")
                
                # Display detailed results
                for result in results:
                    if result["success"]:
                        with st.expander(f"Results for {result['url']}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("#### Content Analysis")
                                st.json(result["analysis"]["content_analysis"])
                            with col2:
                                st.markdown("#### Content Categorization")
                                st.json(result["analysis"]["categorization"])
                            
                            if "media" in result["analysis"]:
                                st.markdown("#### Media Content")
                                display_media_preview(result["analysis"]["media"], st)
                    else:
                        st.error(f"Failed to process {result['url']}: {result.get('error', 'Unknown error')}")
                
                # Export option
                if st.button("Export All Results"):
                    export_data = json.dumps(results, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=export_data,
                        file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif mode == "Dashboard":
        st.markdown("## Dashboard")
        
        # Add media statistics to dashboard
        stats_container = st.container()
        display_media_stats(db, stats_container)
        
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
    
    else:  # Settings mode
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Application Settings")
        
        # Scraping Settings
        st.subheader("Scraping Configuration")
        scraper_settings = {
            "max_depth": st.slider("Max Scraping Depth", 1, 5, 3),
            "max_pages": st.slider("Max Pages per Domain", 10, 200, 100),
            "stay_within_domain": st.checkbox("Stay Within Domain", True),
            "crawl_strategy": st.selectbox("Crawling Strategy", ["breadth-first", "depth-first"])
        }
        
        # LLM Settings
        st.subheader("LLM Configuration")
        llm_settings = {
            "model": st.selectbox("Model Version", ["gpt-4", "gpt-3.5-turbo"], disabled=True),
            "max_tokens": st.slider("Max Tokens", 100, 2000, 1000, disabled=True)
        }
        
        if st.button("Test Connections"):
            col1, col2, col3 = st.columns(3)
            
            # Test Database
            try:
                with db.conn.cursor() as cur:
                    cur.execute("SELECT 1")
                col1.success("Database: Connected")
            except Exception as e:
                col1.error(f"Database Error: {str(e)}")
            
            # Test LLM
            try:
                test_result = llm.test_connection()
                if "error" not in test_result:
                    col2.success("LLM: Connected")
                else:
                    col2.error(f"LLM Error: {test_result['error']}")
            except Exception as e:
                col2.error(f"LLM Error: {str(e)}")
            
            # Test Scraper
            try:
                test_url = "https://example.com"
                result = asyncio.run(scraper.scrape_single_url(test_url))
                if result["success"]:
                    col3.success("Scraper: Working")
                else:
                    col3.error(f"Scraper Error: {result.get('error', 'Unknown error')}")
            except Exception as e:
                col3.error(f"Scraper Error: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()