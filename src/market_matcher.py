import os
from typing import Dict, List, Any, Tuple
from uuid import uuid5, NAMESPACE_URL
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from src.logger import setup_logging
from src.models import embedding_model
from src.news_rss import RSSNewsPoller

load_dotenv()


class NewsMarketMatcher:
    """
    Class to handle news article fetching and market matching using vector similarity.
    """

    def __init__(self,
                 qdrant_url: str = None,
                 collection_name: str = "markets"):
        """
        Initialize the NewsMarketMatcher.

        Args:
            qdrant_url: Qdrant database URL
            collection_name: Name of the collection in Qdrant
        """
        self.logger = setup_logging()
        self.collection_name = collection_name

        # Initialize Qdrant client
        url = qdrant_url or os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
        self.client = QdrantClient(url=url)

    def generate_uuid(self, string: str) -> str:
        """
        Generate a UUID based on a string using the NAMESPACE_URL namespace.
        """
        return str(uuid5(NAMESPACE_URL, string))

    def find_similar_markets(self, article: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find the top k most similar markets for a given news article.

        Args:
            article: Dictionary containing article data with 'title' and 'summary'
            top_k: Number of similar markets to return

        Returns:
            List of similar markets with their scores
        """
        try:
            # Combine article title and summary for embedding
            article_text = f"{article.get('title', '')} {article.get('summary', '')}"

            # Generate embedding for the article
            article_vector = next(embedding_model.embed([article_text]))

            # Search for similar markets in Qdrant
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=article_vector,
                limit=top_k,
                with_payload=True
            ).points

            # Format results
            similar_markets = []
            for result in search_results:
                market_data = {
                    "condition_id": result.payload.get("condition_id"),
                    "question": result.payload.get("question"),
                    "description": result.payload.get("description"),
                    "similarity_score": result.score,
                    "tokens": result.payload.get("tokens", [])
                }
                similar_markets.append(market_data)

            return similar_markets

        except Exception as e:
            self.logger.error(f"Error finding similar markets for article '{article.get('title', 'Unknown')}': {e}")
            return []

    def process_news_articles(self,
                              news_data: Dict[str, List[Dict[str, Any]]],
                              top_k: int = 5
                              ) -> Dict[str, List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]]:
        """
        Process all news articles and find similar markets for each.

        Args:
            news_data: Dictionary with source names as keys and lists of articles as values
            top_k: Number of similar markets to find for each article

        Returns:
            Dictionary mapping source names to lists of (article, similar_markets) tuples
        """
        results = {}

        for source_name, articles in news_data.items():
            source_results = []

            for article in articles:
                similar_markets = self.find_similar_markets(article, top_k)
                source_results.append((article, similar_markets))

            results[source_name] = source_results

        return results

    def fetch_news_and_match_markets(self,
                                     rss_feeds: Dict[str, str] = None,
                                     state_path: str = "rss_poller_state.json",
                                     max_items_per_feed: int = 10,
                                     top_k_markets: int = 5
                                     ) -> Dict[str, List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]]:
        """
        Main function to run news polling and market matching.

        Args:
            rss_feeds: Dictionary of RSS feed sources and URLs
            state_path: Path to store RSS polling state
            max_items_per_feed: Maximum items to fetch per RSS feed
            top_k_markets: Number of similar markets to find per article

        Returns:
            Dictionary of processed results
        """
        # Initialize RSS poller
        poller = RSSNewsPoller(
            feeds=rss_feeds,
            state_path=state_path,
            max_items_per_feed=max_items_per_feed
        )

        # Fetch new articles
        self.logger.info("Fetching new RSS articles...")
        news_data = poller.single_poll()

        # Process articles to find matching markets
        self.logger.info("Finding matching prediction markets...")
        results = self.process_news_articles(news_data, top_k_markets)

        return results

    @staticmethod
    def format_results(results: Dict[str, List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]],
                       llm_results: Dict[str, Dict[str, str]] = None) -> str:
        """
        Format the news-market matching results for display.

        Args:
            results: Dictionary of news articles and their matching markets
            llm_results: Dictionary of LLM results for each article-market pair

        Returns:
            Formatted string for display
        """
        lines = []
        lines.append("=" * 80)
        lines.append("NEWS ARTICLES & MATCHING PREDICTION MARKETS")
        lines.append("=" * 80)

        total_articles = sum(len(articles) for articles in results.values())
        if total_articles == 0:
            lines.append("\nNo new articles found.")
            return "\n".join(lines)

        for source_name, article_market_pairs in results.items():
            if not article_market_pairs:
                continue

            lines.append(f"\n{source_name.upper()}")
            lines.append("=" * len(source_name))

            for i, (article, markets) in enumerate(article_market_pairs, 1):
                lines.append(f"\n[{i}] NEWS ARTICLE:")
                lines.append(f"Title: {article.get('title', 'No title')}")

                if markets:
                    lines.append(f"\nTOP MATCHING MARKETS:")
                    for j, market in enumerate(markets, 1):
                        score_pct = market.get('similarity_score', 0) * 100
                        market_question = market.get('question', 'No question')

                        # Get LLM result if available
                        article_title = article.get('title', 'No title')
                        llm_key = f"{article_title}_{market_question}"
                        llm_result = ""

                        if llm_results and llm_key in llm_results:
                            llm_status = llm_results[llm_key]
                            if llm_status == 'yes':
                                llm_result = " [LLM:RELATED]"
                            elif llm_status == 'no':
                                llm_result = " [LLM:NOT RELATED]"
                            else:
                                llm_result = f" [LLM:{llm_status.upper()}]"

                        lines.append(f"      {j}. [{score_pct:.1f}%] {market_question}{llm_result}")
                else:
                    lines.append(f"\n    No matching markets found.")

                lines.append("-" * 60)

        return "\n".join(lines)