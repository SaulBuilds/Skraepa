import trafilatura
from typing import List, Dict, Set, Optional, Tuple, Any, Union, TypedDict
from collections import defaultdict
import asyncio
from urllib.parse import urlparse, urljoin
import logging
import time
from requests.exceptions import RequestException, Timeout, ConnectionError, TooManyRedirects
import aiohttp
import async_timeout
from bs4 import BeautifulSoup
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MediaMetadata(TypedDict):
    total_count: int
    processed_count: int
    failed_count: int

class MediaInfo(TypedDict):
    images: List[Dict[str, Any]]
    videos: List[Dict[str, Any]]
    metadata: MediaMetadata

class WebScraper:
    def __init__(self, 
                 max_retries: int = 3, 
                 retry_delay: float = 1.0, 
                 max_concurrent: int = 5,
                 max_depth: int = 3,
                 max_pages_per_domain: int = 100,
                 stay_within_domain: bool = True,
                 timeout: int = 30,
                 crawl_strategy: str = "breadth-first"):
        self.results = []
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_concurrent = max_concurrent
        self.max_depth = max_depth
        self.max_pages_per_domain = max_pages_per_domain
        self.stay_within_domain = stay_within_domain
        self.timeout = timeout
        self.crawl_strategy = crawl_strategy
        self.session: Optional[aiohttp.ClientSession] = None
        self.visited_urls: Set[str] = set()
        self.progress_callback = None
        logger.info(f"WebScraper initialized with settings: max_depth={max_depth}, "
                   f"max_pages_per_domain={max_pages_per_domain}, "
                   f"stay_within_domain={stay_within_domain}, "
                   f"crawl_strategy={crawl_strategy}")

    def set_progress_callback(self, callback):
        """Set callback function for progress tracking"""
        self.progress_callback = callback

    def validate_url(self, url: str) -> bool:
        """Validate URL format and accessibility"""
        try:
            result = urlparse(url)
            valid = all([result.scheme, result.netloc])
            if not valid:
                logger.warning(f"Invalid URL format: {url}")
            return valid
        except ValueError as e:
            logger.error(f"URL validation error for {url}: {str(e)}")
            return False

    async def ensure_session(self) -> None:
        """Ensure aiohttp session exists"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def download_media(self, url: str) -> Optional[Dict[str, Any]]:
        """Download and validate media content"""
        try:
            await self.ensure_session()
            if not self.session:
                raise ValueError("Failed to initialize session")
                
            async with async_timeout.timeout(self.timeout):
                async with self.session.head(url) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to fetch media from {url}: HTTP {response.status}")
                        return None
                    
                    content_type = response.headers.get('content-type', '')
                    content_length = response.headers.get('content-length')
                    
                    return {
                        'content_type': content_type,
                        'size': int(content_length) if content_length else None,
                        'last_modified': response.headers.get('last-modified'),
                        'etag': response.headers.get('etag'),
                        'status': 'available'
                    }
        except Exception as e:
            logger.error(f"Error downloading media from {url}: {str(e)}")
            return None

    async def extract_media(self, html_content: str, base_url: str) -> MediaInfo:
        """Extract media with enhanced attributes and metadata"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            metadata: MediaMetadata = {
                'total_count': 0,
                'processed_count': 0,
                'failed_count': 0
            }
            
            images: List[Dict[str, Any]] = []
            videos: List[Dict[str, Any]] = []

            # Process images with enhanced attributes
            for img in soup.find_all('img'):
                try:
                    current_src = img.get('src', '')
                    srcset = img.get('srcset', '')
                    sources: List[Union[str, Dict[str, Any]]] = []

                    if current_src:
                        sources.append(urljoin(base_url, current_src))
                    
                    # Parse srcset attribute
                    if srcset:
                        for srcset_item in srcset.split(','):
                            parts = srcset_item.strip().split()
                            if len(parts) >= 1:
                                src_url = urljoin(base_url, parts[0])
                                width = parts[1] if len(parts) > 1 else None
                                sources.append({'url': src_url, 'width': width})

                    if not sources:
                        continue

                    metadata['total_count'] += 1
                    
                    # Get primary image URL
                    primary_url = sources[0] if isinstance(sources[0], str) else sources[0]['url']
                    if not self.validate_url(primary_url):
                        metadata['failed_count'] += 1
                        continue

                    # Download and validate media
                    download_info = await self.download_media(primary_url)
                    if not download_info:
                        metadata['failed_count'] += 1
                        continue

                    image_data = {
                        'url': primary_url,
                        'alt': img.get('alt', ''),
                        'title': img.get('title', ''),
                        'loading': img.get('loading', 'eager'),
                        'sizes': img.get('sizes', ''),
                        'alternative_sources': sources[1:] if len(sources) > 1 else [],
                        'metadata': {
                            'width': img.get('width', ''),
                            'height': img.get('height', ''),
                            'class': img.get('class', []),
                            'download_info': download_info
                        }
                    }
                    
                    images.append(image_data)
                    metadata['processed_count'] += 1
                except Exception as e:
                    logger.error(f"Error processing image {img.get('src', 'unknown')}: {str(e)}")
                    metadata['failed_count'] += 1

            # Process videos with enhanced support
            for video_element in soup.find_all(['video', 'iframe', 'source']):
                try:
                    metadata['total_count'] += 1
                    current_src = ''
                    sources: List[Union[str, Dict[str, Any]]] = []
                    
                    # Handle different video elements
                    if video_element.name == 'video':
                        current_src = video_element.get('src', '')
                        if current_src:
                            sources.append(urljoin(base_url, current_src))
                        
                        # Video source elements
                        for source in video_element.find_all('source'):
                            src = source.get('src')
                            if src:
                                sources.append({
                                    'url': urljoin(base_url, src),
                                    'type': source.get('type', ''),
                                    'media': source.get('media', '')
                                })
                    
                    elif video_element.name == 'iframe':
                        current_src = video_element.get('src', '')
                        if current_src:
                            sources = [urljoin(base_url, current_src)]
                    
                    else:  # source element
                        current_src = video_element.get('src', '')
                        if current_src and video_element.parent.name != 'video':  # Avoid duplicates
                            sources = [urljoin(base_url, current_src)]
                    
                    if not sources:
                        metadata['failed_count'] += 1
                        continue

                    # Get primary video URL
                    primary_url = sources[0] if isinstance(sources[0], str) else sources[0]['url']
                    if not self.validate_url(primary_url):
                        metadata['failed_count'] += 1
                        continue

                    # Download and validate media
                    download_info = await self.download_media(primary_url)
                    if not download_info:
                        metadata['failed_count'] += 1
                        continue

                    video_data = {
                        'url': primary_url,
                        'type': video_element.name,
                        'alternative_sources': sources[1:] if len(sources) > 1 else [],
                        'metadata': {
                            'width': video_element.get('width', ''),
                            'height': video_element.get('height', ''),
                            'autoplay': video_element.get('autoplay', False),
                            'controls': video_element.get('controls', False),
                            'loop': video_element.get('loop', False),
                            'muted': video_element.get('muted', False),
                            'poster': video_element.get('poster', ''),
                            'preload': video_element.get('preload', 'auto'),
                            'download_info': download_info
                        }
                    }
                    
                    videos.append(video_data)
                    metadata['processed_count'] += 1
                except Exception as e:
                    logger.error(f"Error processing video {video_element.get('src', 'unknown')}: {str(e)}")
                    metadata['failed_count'] += 1

            return {
                'images': images,
                'videos': videos,
                'metadata': metadata
            }
        except Exception as e:
            logger.error(f"Error extracting media from {base_url}: {str(e)}")
            return {
                'images': [],
                'videos': [],
                'metadata': {
                    'total_count': 0,
                    'processed_count': 0,
                    'failed_count': 0
                }
            }

    def get_domain(self, url: str) -> str:
        """Extract domain from URL"""
        return urlparse(url).netloc

    def should_process_url(self, url: str, base_domain: str) -> bool:
        """Check if URL should be processed based on configuration"""
        if not self.validate_url(url):
            return False
        if url in self.visited_urls:
            return False
        if len(self.visited_urls) >= self.max_pages_per_domain:
            return False
        if self.stay_within_domain and self.get_domain(url) != base_domain:
            return False
        return True

    async def extract_links(self, html_content: str, base_url: str) -> List[str]:
        """Extract and validate internal links from HTML content"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            links = []
            base_domain = self.get_domain(base_url)

            for anchor in soup.find_all('a', href=True):
                href = anchor['href']
                absolute_url = urljoin(base_url, href)
                
                if self.should_process_url(absolute_url, base_domain):
                    links.append(absolute_url)

            return links
        except Exception as e:
            logger.error(f"Error extracting links from {base_url}: {str(e)}")
            return []

    async def scrape_single_url(self, url: str, depth: int = 0) -> Dict[str, Any]:
        """Async method to scrape a single URL"""
        if not self.validate_url(url):
            return {"url": url, "success": False, "error": "Invalid URL format"}
        
        await self.ensure_session()
        if not self.session:
            return {"url": url, "success": False, "error": "Failed to initialize session"}
            
        return await self.fetch_url_content(url, depth)

    def scrape_url(self, url: str) -> Dict[str, Any]:
        """Synchronous wrapper for scrape_single_url"""
        return asyncio.run(self.scrape_single_url(url))

    async def fetch_url_content(self, url: str, depth: int = 0, retries: int = 0) -> Dict[str, Any]:
        """Fetch URL content using aiohttp with retry mechanism"""
        if not self.validate_url(url):
            logger.error(f"Invalid URL format: {url}")
            return {"url": url, "success": False, "error": "Invalid URL format"}

        if url in self.visited_urls:
            return {"url": url, "success": False, "error": "URL already processed"}

        if not self.session:
            return {"url": url, "success": False, "error": "No active session"}

        last_error = None

        while retries < self.max_retries:
            try:
                logger.info(f"Attempting to fetch URL: {url} (Depth: {depth}, Attempt {retries + 1}/{self.max_retries})")
                
                async with async_timeout.timeout(self.timeout):
                    async with self.session.get(url) as response:
                        if response.status != 200:
                            raise aiohttp.ClientError(f"HTTP {response.status}")
                        
                        html_content = await response.text()
                        if not html_content:
                            raise ValueError("Empty response received")

                        content = trafilatura.extract(html_content)
                        if content is None:
                            raise ValueError("No content extracted")

                        # Extract media content
                        media = await self.extract_media(html_content, url)

                        self.visited_urls.add(url)
                        logger.info(f"Successfully scraped URL: {url}")

                        if self.progress_callback:
                            await self.progress_callback(url, depth, len(self.visited_urls))

                        return {
                            "url": url,
                            "success": True,
                            "content": content,
                            "media": media,
                            "links": await self.extract_links(html_content, url)
                        }

            except asyncio.TimeoutError as e:
                logger.warning(f"Timeout error for URL {url}: {str(e)}")
                last_error = f"Connection timeout: {str(e)}"
            except aiohttp.ClientError as e:
                logger.warning(f"Client error for URL {url}: {str(e)}")
                last_error = f"Client error: {str(e)}"
            except ValueError as e:
                logger.warning(f"Content extraction error for URL {url}: {str(e)}")
                last_error = str(e)
            except Exception as e:
                logger.error(f"Unexpected error scraping URL {url}: {str(e)}")
                last_error = str(e)

            retries += 1
            if retries < self.max_retries:
                logger.info(f"Retrying in {self.retry_delay} seconds...")
                await asyncio.sleep(self.retry_delay)
            else:
                logger.error(f"Max retries reached for URL: {url}")
                return {
                    "url": url,
                    "success": False,
                    "error": f"Max retries reached: {last_error}"
                }

    async def recursive_scrape(self, start_url: str) -> List[Dict[str, Any]]:
        """Recursively scrape URLs starting from a given URL"""
        if not self.validate_url(start_url):
            return [{"url": start_url, "success": False, "error": "Invalid URL format"}]
        
        await self.ensure_session()
        if not self.session:
            return [{"url": start_url, "success": False, "error": "Failed to initialize session"}]

        try:
            logger.info(f"Starting recursive scrape from {start_url}")
            queue: deque = deque([(start_url, 0)])  # (url, depth)
            results = []
            
            while queue and len(self.visited_urls) < self.max_pages_per_domain:
                current_url, depth = queue[0] if self.crawl_strategy == "breadth-first" else queue[-1]
                queue.popleft() if self.crawl_strategy == "breadth-first" else queue.pop()

                if depth > self.max_depth:
                    continue

                result = await self.fetch_url_content(current_url, depth)
                if result["success"]:
                    results.append(result)
                    
                    # Process links for next level
                    if depth < self.max_depth:
                        for link in result.get("links", []):
                            if self.should_process_url(link, self.get_domain(start_url)):
                                if self.crawl_strategy == "breadth-first":
                                    queue.append((link, depth + 1))
                                else:
                                    queue.appendleft((link, depth + 1))  # Add to start for DFS

            return results
        except Exception as e:
            logger.error(f"Error in recursive scrape: {str(e)}")
            return [{"url": start_url, "success": False, "error": str(e)}]
        finally:
            if self.session and not self.session.closed:
                await self.session.close()

    async def scrape_batch(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Batch process multiple URLs"""
        try:
            await self.ensure_session()
            if not self.session:
                return [{"url": url, "success": False, "error": "Failed to initialize session"} for url in urls]

            tasks = [self.scrape_single_url(url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            processed_results = []
            for url, result in zip(urls, results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "url": url,
                        "success": False,
                        "error": str(result)
                    })
                else:
                    processed_results.append(result)
            
            return processed_results
        except Exception as e:
            logger.error(f"Error in batch scraping: {str(e)}")
            return [{"url": url, "success": False, "error": str(e)} for url in urls]
        finally:
            if self.session and not self.session.closed:
                await self.session.close()

    async def __aenter__(self):
        await self.ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session and not self.session.closed:
            await self.session.close()

import sqlite3
from typing import List, Dict, Optional, Union

class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        """Connect to the SQLite database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.execute('''CREATE TABLE IF NOT EXISTS scraped_data (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                url TEXT UNIQUE NOT NULL,
                                content TEXT,
                                media_images TEXT,
                                media_videos TEXT,
                                extracted_at DATETIME DEFAULT CURRENT_TIMESTAMP
                            )''')
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")

    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            logger.info(f"Closed database connection: {self.db_path}")

    def insert_data(self, data: Dict) -> Optional[int]:
        """Insert scraped data into the database"""
        try:
            if not self.conn:
                self.connect()
            
            cursor = self.conn.cursor()
            
            # Convert media to JSON strings
            media_images = str(data.get("media", {}).get("images", []))
            media_videos = str(data.get("media", {}).get("videos", []))
            
            cursor.execute("INSERT OR REPLACE INTO scraped_data (url, content, media_images, media_videos) VALUES (?, ?, ?, ?)", 
                           (data["url"], data.get("content", None), media_images, media_videos))
            
            self.conn.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error inserting data: {str(e)}")
            return None

    def get_data_by_url(self, url: str) -> Optional[Dict]:
        """Retrieve data from the database based on URL"""
        try:
            if not self.conn:
                self.connect()
            
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM scraped_data WHERE url = ?", (url,))
            row = cursor.fetchone()
            if row:
                return {
                    "id": row[0],
                    "url": row[1],
                    "content": row[2],
                    "media_images": row[3],
                    "media_videos": row[4],
                    "extracted_at": row[5]
                }
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting data by URL: {str(e)}")
            return None

    def get_all_data(self) -> List[Dict]:
        """Retrieve all scraped data from the database"""
        try:
            if not self.conn:
                self.connect()
            
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM scraped_data")
            rows = cursor.fetchall()
            return [
                {
                    "id": row[0],
                    "url": row[1],
                    "content": row[2],
                    "media_images": row[3],
                    "media_videos": row[4],
                    "extracted_at": row[5]
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Error getting all data: {str(e)}")
            return []

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

import datetime

async def main():
    db = Database("scraped_data.db")
    scraper = WebScraper(max_depth=1, max_pages_per_domain=5)

    start_time = time.time()

    try:
        async with scraper:
            # Example usage for single URL scraping
            # results = await scraper.recursive_scrape("https://www.example.com")
            # print(results)

            # Example usage for batch scraping
            urls = [
                "https://www.example.com",
                "https://www.google.com",
                "https://www.facebook.com"
            ]
            results = await scraper.scrape_batch(urls)

            # Store scraped data in the database
            for result in results:
                if result["success"]:
                    db.insert_data(result)

            # Retrieve and print data from the database
            # print(db.get_data_by_url("https://www.example.com"))
            print(db.get_all_data())

    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())