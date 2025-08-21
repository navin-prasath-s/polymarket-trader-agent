import os
from typing import Dict, List, Any, Tuple
from uuid import uuid5, NAMESPACE_URL
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from src.logger import setup_logging
from src.models import embedding_model
from src.news_rss import RSSNewsPoller

load_dotenv()
logger = setup_logging()

# Qdrant client setup
client = QdrantClient(url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"))
collection_name = "markets"


def generate_uuid(string: str) -> str:
    """
    Generate a UUID based on a string using the NAMESPACE_URL namespace.
    """
    return str(uuid5(NAMESPACE_URL, string))


def find_similar_markets(article: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
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
        search_results = client.query_points(
            collection_name=collection_name,
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
        logger.error(f"Error finding similar markets for article '{article.get('title', 'Unknown')}': {e}")
        return []


def process_news_articles(
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
            similar_markets = find_similar_markets(article, top_k)
            source_results.append((article, similar_markets))

        results[source_name] = source_results

    return results


def format_results(
        results: Dict[str, List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]]
) -> str:
    """
    Format the news-market matching results for display.

    Args:
        results: Dictionary of news articles and their matching markets

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
            lines.append(f"    Title: {article.get('title', 'No title')}")

            if markets:
                lines.append(f"\n    TOP MATCHING MARKETS:")
                for j, market in enumerate(markets, 1):
                    score_pct = market.get('similarity_score', 0) * 100
                    lines.append(f"      {j}. [{score_pct:.1f}%] {market.get('question', 'No question')}")
                    if market.get('description'):
                        desc = market['description'][:150]
                        if len(market['description']) > 150:
                            desc += "..."
                        lines.append(f"         Description: {desc}")
            else:
                lines.append(f"\n    No matching markets found.")

            lines.append("-" * 60)

    return "\n".join(lines)


def run_news_market_matching(
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
    logger.info("Fetching new RSS articles...")
    news_data = poller.single_poll()

    # Process articles to find matching markets
    logger.info("Finding matching prediction markets...")
    results = process_news_articles(news_data, top_k_markets)

    return results


if __name__ == "__main__":
    # Default RSS feeds (you can customize these)
    default_feeds = {
        "BBC News": "https://feeds.bbci.co.uk/news/rss.xml",
        "NPR": "https://feeds.npr.org/1001/rss.xml",
        # "Reuters": "https://feeds.reuters.com/reuters/topNews",
        # "AP News": "https://feeds.apnews.com/rss/apf-topnews",
    }

    try:
        # Run the matching process
        results = run_news_market_matching(
            rss_feeds=default_feeds,
            state_path="rss_poller_state.json",
            max_items_per_feed=10,
            top_k_markets=5
        )

        # Format and print results
        formatted_output = format_results(results)
        print(formatted_output)

        # Print summary statistics
        total_articles = sum(len(articles) for articles in results.values())
        sources_with_articles = sum(1 for articles in results.values() if articles)

        print(f"\nSUMMARY:")
        print(f"Sources checked: {len(results)}")
        print(f"Sources with new articles: {sources_with_articles}")
        print(f"Total new articles: {total_articles}")

    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        print(f"Error occurred: {e}")