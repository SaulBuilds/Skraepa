import trafilatura
from typing import List, Dict
import asyncio
from urllib.parse import urlparse

class WebScraper:
    def __init__(self):
        self.results = []

    def validate_url(self, url: str) -> bool:
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def scrape_single_url(self, url: str) -> Dict:
        try:
            if not self.validate_url(url):
                return {"url": url, "success": False, "error": "Invalid URL format"}

            downloaded = trafilatura.fetch_url(url)
            if downloaded is None:
                return {"url": url, "success": False, "error": "Failed to download content"}

            content = trafilatura.extract(downloaded)
            if content is None:
                return {"url": url, "success": False, "error": "No content extracted"}

            return {
                "url": url,
                "success": True,
                "content": content
            }
        except Exception as e:
            return {"url": url, "success": False, "error": str(e)}

    async def scrape_batch(self, urls: List[str]) -> List[Dict]:
        results = []
        for url in urls:
            result = self.scrape_single_url(url)
            results.append(result)
        return results
