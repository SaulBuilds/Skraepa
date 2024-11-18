import trafilatura
from typing import List, Dict, Set, Optional, Tuple
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
        self.session = None
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

    async def extract_media(self, html_content: str, base_url: str) -> Dict[str, List[Dict[str, str]]]:
        """Extract media (images and videos) from HTML content"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            media = {
                'images': [],
                'videos': []
            }

            # Extract images
            for img in soup.find_all('img'):
                src = img.get('src')
                if src:
                    absolute_url = urljoin(base_url, src)
                    if self.validate_url(absolute_url):
                        media['images'].append({
                            'url': absolute_url,
                            'alt': img.get('alt', ''),
                            'metadata': {
                                'width': img.get('width', ''),
                                'height': img.get('height', ''),
                                'class': img.get('class', []),
                                'loading': img.get('loading', 'eager')
                            }
                        })

            # Extract videos
            for video in soup.find_all(['video', 'iframe']):
                src = video.get('src')
                if not src and video.name == 'video':
                    source = video.find('source')
                    if source:
                        src = source.get('src')
                
                if src:
                    absolute_url = urljoin(base_url, src)
                    if self.validate_url(absolute_url):
                        media['videos'].append({
                            'url': absolute_url,
                            'type': video.name,
                            'metadata': {
                                'width': video.get('width', ''),
                                'height': video.get('height', ''),
                                'autoplay': video.get('autoplay', False),
                                'controls': video.get('controls', False)
                            }
                        })

            return media
        except Exception as e:
            logger.error(f"Error extracting media from {base_url}: {str(e)}")
            return {'images': [], 'videos': []}

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

    async def scrape_single_url(self, url: str, depth: int = 0) -> Dict:
        """Async method to scrape a single URL"""
        if not self.validate_url(url):
            return {"url": url, "success": False, "error": "Invalid URL format"}
        
        async with aiohttp.ClientSession() as session:
            self.session = session
            return await self.fetch_url_content(url, depth)

    def scrape_url(self, url: str) -> Dict:
        """Synchronous wrapper for scrape_single_url"""
        return asyncio.run(self.scrape_single_url(url))

    async def fetch_url_content(self, url: str, depth: int = 0, retries: int = 0) -> Dict:
        """Fetch URL content using aiohttp with retry mechanism"""
        if not self.validate_url(url):
            logger.error(f"Invalid URL format: {url}")
            return {"url": url, "success": False, "error": "Invalid URL format"}

        if url in self.visited_urls:
            return {"url": url, "success": False, "error": "URL already processed"}

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

    async def recursive_scrape(self, start_url: str) -> List[Dict]:
        """Recursively scrape URLs starting from a given URL"""
        if not self.validate_url(start_url):
            return [{"url": start_url, "success": False, "error": "Invalid URL format"}]

        base_domain = self.get_domain(start_url)
        results = []
        
        if self.crawl_strategy == "breadth-first":
            queue = deque([(start_url, 0)])  # (url, depth)
        else:  # depth-first
            queue = [(start_url, 0)]  # Stack for DFS

        async with aiohttp.ClientSession() as session:
            self.session = session
            
            while queue and len(self.visited_urls) < self.max_pages_per_domain:
                current_url, depth = queue.popleft() if self.crawl_strategy == "breadth-first" else queue.pop()
                
                if not self.should_process_url(current_url, base_domain):
                    continue

                result = await self.fetch_url_content(current_url, depth)
                results.append(result)

                if result["success"] and depth < self.max_depth:
                    links = result.get("links", [])
                    for link in links:
                        if self.should_process_url(link, base_domain):
                            if self.crawl_strategy == "breadth-first":
                                queue.append((link, depth + 1))
                            else:
                                queue.insert(0, (link, depth + 1))  # Add to start for DFS

        return results

    async def scrape_batch(self, urls: List[str]) -> List[Dict]:
        """Scrape multiple URLs concurrently with enhanced error handling"""
        if not urls:
            logger.warning("Empty URL list provided for batch processing")
            return []

        logger.info(f"Starting batch processing of {len(urls)} URLs")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def fetch_with_semaphore(url: str) -> Dict:
            async with semaphore:
                return await self.fetch_url_content(url)

        try:
            async with aiohttp.ClientSession() as session:
                self.session = session
                tasks = [fetch_with_semaphore(url) for url in urls]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results and handle any exceptions
                processed_results = []
                for idx, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Error processing URL {urls[idx]}: {str(result)}")
                        processed_results.append({
                            "url": urls[idx],
                            "success": False,
                            "error": str(result)
                        })
                    else:
                        processed_results.append(result)

                successful = sum(1 for r in processed_results if r["success"])
                logger.info(f"Batch processing completed. Success: {successful}/{len(urls)}")
                return processed_results

        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            return [{"url": url, "success": False, "error": str(e)} for url in urls]

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session and not self.session.closed:
            await self.session.close()
