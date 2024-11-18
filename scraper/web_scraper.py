import trafilatura
from typing import List, Dict
import asyncio
from urllib.parse import urlparse
import logging
import time
from requests.exceptions import RequestException, Timeout, ConnectionError, TooManyRedirects
import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.results = []
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        logger.info(f"WebScraper initialized with max_retries={max_retries}, retry_delay={retry_delay}")

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

    def scrape_single_url(self, url: str) -> Dict:
        """Scrape a single URL with retry mechanism"""
        if not self.validate_url(url):
            logger.error(f"Invalid URL format: {url}")
            return {"url": url, "success": False, "error": "Invalid URL format"}

        retries = 0
        last_error = None

        while retries < self.max_retries:
            try:
                logger.info(f"Attempting to scrape URL: {url} (Attempt {retries + 1}/{self.max_retries})")
                downloaded = trafilatura.fetch_url(url)
                
                if downloaded is None:
                    raise ValueError("Failed to download content")

                content = trafilatura.extract(downloaded)
                if content is None:
                    raise ValueError("No content extracted")

                logger.info(f"Successfully scraped URL: {url}")
                return {
                    "url": url,
                    "success": True,
                    "content": content
                }

            except Timeout as e:
                logger.warning(f"Timeout error for URL {url}: {str(e)}")
                last_error = f"Connection timeout: {str(e)}"
            except TooManyRedirects as e:
                logger.warning(f"Too many redirects for URL {url}: {str(e)}")
                last_error = f"Too many redirects: {str(e)}"
            except ConnectionError as e:
                logger.warning(f"Connection error for URL {url}: {str(e)}")
                last_error = f"Connection error: {str(e)}"
            except ValueError as e:
                logger.warning(f"Content extraction error for URL {url}: {str(e)}")
                last_error = str(e)
            except Exception as e:
                logger.error(f"Unexpected error scraping URL {url}: {str(e)}")
                last_error = str(e)

            retries += 1
            if retries < self.max_retries:
                logger.info(f"Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
            else:
                logger.error(f"Max retries reached for URL: {url}")
                return {
                    "url": url,
                    "success": False,
                    "error": f"Max retries reached: {last_error}"
                }

    async def scrape_batch(self, urls: List[str]) -> List[Dict]:
        """Scrape multiple URLs with enhanced error handling"""
        if not urls:
            logger.warning("Empty URL list provided for batch processing")
            return []

        logger.info(f"Starting batch processing of {len(urls)} URLs")
        results = []
        
        for url in urls:
            try:
                logger.info(f"Processing URL in batch: {url}")
                result = self.scrape_single_url(url)
                results.append(result)
                
                if not result["success"]:
                    logger.warning(f"Failed to process URL {url}: {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                logger.error(f"Batch processing error for URL {url}: {str(e)}")
                results.append({
                    "url": url,
                    "success": False,
                    "error": str(e)
                })

        successful = sum(1 for r in results if r["success"])
        logger.info(f"Batch processing completed. Success: {successful}/{len(urls)}")
        return results

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
