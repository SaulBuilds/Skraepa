import trafilatura
from typing import List, Dict
import asyncio
from urllib.parse import urlparse
import logging
import time
from requests.exceptions import RequestException, Timeout, ConnectionError, TooManyRedirects
import aiohttp
import async_timeout

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0, max_concurrent: int = 5):
        self.results = []
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_concurrent = max_concurrent
        self.session = None
        logger.info(f"WebScraper initialized with max_retries={max_retries}, retry_delay={retry_delay}, max_concurrent={max_concurrent}")

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

    async def scrape_single_url(self, url: str) -> Dict:
        """Async method to scrape a single URL"""
        if not self.validate_url(url):
            return {"url": url, "success": False, "error": "Invalid URL format"}
        
        async with aiohttp.ClientSession() as session:
            self.session = session
            return await self.fetch_url_content(url)

    def scrape_url(self, url: str) -> Dict:
        """Synchronous wrapper for scrape_single_url"""
        return asyncio.run(self.scrape_single_url(url))

    async def fetch_url_content(self, url: str, retries: int = 0) -> Dict:
        """Fetch URL content using aiohttp with retry mechanism"""
        if not self.validate_url(url):
            logger.error(f"Invalid URL format: {url}")
            return {"url": url, "success": False, "error": "Invalid URL format"}

        last_error = None

        while retries < self.max_retries:
            try:
                logger.info(f"Attempting to fetch URL: {url} (Attempt {retries + 1}/{self.max_retries})")
                
                async with async_timeout.timeout(30):  # 30 seconds timeout
                    async with self.session.get(url) as response:
                        if response.status != 200:
                            raise aiohttp.ClientError(f"HTTP {response.status}")
                        
                        html_content = await response.text()
                        if not html_content:
                            raise ValueError("Empty response received")

                        content = trafilatura.extract(html_content)
                        if content is None:
                            raise ValueError("No content extracted")

                        logger.info(f"Successfully scraped URL: {url}")
                        return {
                            "url": url,
                            "success": True,
                            "content": content
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
