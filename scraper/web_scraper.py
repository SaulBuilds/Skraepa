import trafilatura
from typing import List, Dict
import asyncio
from urllib.parse import urlparse
import logging
import time
from requests.exceptions import RequestException, Timeout, ConnectionError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.results = []
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def validate_url(self, url: str) -> bool:
        """Validate URL format and accessibility"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError as e:
            logger.error(f"Invalid URL format: {str(e)}")
            return False

    def scrape_single_url(self, url: str) -> Dict:
        """Scrape a single URL with retry mechanism"""
        if not self.validate_url(url):
            logger.error(f"Invalid URL format: {url}")
            return {"url": url, "success": False, "error": "Invalid URL format"}

        retries = 0
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
                error_msg = "Connection timeout"
            except ConnectionError as e:
                logger.warning(f"Connection error for URL {url}: {str(e)}")
                error_msg = "Connection error"
            except Exception as e:
                logger.error(f"Error scraping URL {url}: {str(e)}")
                error_msg = str(e)

            retries += 1
            if retries < self.max_retries:
                logger.info(f"Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
            else:
                logger.error(f"Max retries reached for URL: {url}")
                return {"url": url, "success": False, "error": f"Max retries reached: {error_msg}"}

    async def scrape_batch(self, urls: List[str]) -> List[Dict]:
        """Scrape multiple URLs"""
        results = []
        for url in urls:
            try:
                result = self.scrape_single_url(url)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch processing error for URL {url}: {str(e)}")
                results.append({
                    "url": url,
                    "success": False,
                    "error": str(e)
                })
        return results
