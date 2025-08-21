from typing import Dict, List, Any, Tuple
from src.market_matcher import NewsMarketMatcher
from src.models import ask_llm_if_related

def format_llm_input(article: Dict[str, Any], market: Dict[str, Any]) -> str:
    """
    Format article and market data for LLM input.

    Args:
        article: Article dictionary with title, summary, etc.
        market: Market dictionary with question, description, etc.

    Returns:
        Formatted string for LLM
    """
    article_text = f"{article.get('title', '')} {article.get('summary', '')}"
    market_question = market.get('question', 'No question available')

    return f"""
    News: {article_text.strip()}

    Market: {market_question}
    """


def process_with_llm(results: Dict[str, List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]]) -> Dict[str, str]:
    """
    Process all article-market pairs through the LLM.

    Args:
        results: Dictionary of articles and their matching markets

    Returns:
        Dictionary mapping article_title_market_question to LLM result
    """
    llm_results = {}
    total_pairs = 0
    processed_pairs = 0

    # Count total pairs for progress tracking
    for source_name, article_market_pairs in results.items():
        for article, markets in article_market_pairs:
            total_pairs += len(markets)

    print(f"\n🤖 Processing {total_pairs} article-market pairs through LLM...")

    for source_name, article_market_pairs in results.items():
        for article, markets in article_market_pairs:
            article_title = article.get('title', 'No title')

            for market in markets:
                market_question = market.get('question', 'No question')

                # Create unique key for this article-market pair
                key = f"{article_title}_{market_question}"

                # Format input for LLM
                llm_input = format_llm_input(article, market)

                # Use the imported LLM function
                llm_result = ask_llm_if_related(llm_input)
                llm_results[key] = llm_result

                processed_pairs += 1
                if processed_pairs % 5 == 0 or processed_pairs == total_pairs:
                    print(f"Progress: {processed_pairs}/{total_pairs} pairs processed")

    return llm_results


def main():
    """
    Main function to run the complete news-market matching with LLM evaluation.
    """
    # Default RSS feeds
    default_feeds = {
        "BBC News": "https://feeds.bbci.co.uk/news/rss.xml",
        "NPR": "https://feeds.npr.org/1001/rss.xml",
        # Add more feeds as needed
    }

    try:
        # Initialize the matcher
        matcher = NewsMarketMatcher()

        # Run the matching process
        print("🔄 Fetching news and matching markets...")
        results = matcher.fetch_news_and_match_markets(
            rss_feeds=default_feeds,
            state_path="rss_poller_state.json",
            max_items_per_feed=10,
            top_k_markets=5
        )

        # Process through LLM
        llm_results = process_with_llm(results)

        # Format and print results with LLM evaluation
        formatted_output = NewsMarketMatcher.format_results(results, llm_results)
        print(formatted_output)

        # Print summary statistics
        total_articles = sum(len(articles) for articles in results.values())
        sources_with_articles = sum(1 for articles in results.values() if articles)

        # LLM statistics
        llm_yes_count = sum(1 for result in llm_results.values() if result == 'yes')
        llm_no_count = sum(1 for result in llm_results.values() if result == 'no')
        llm_invalid_count = len(llm_results) - llm_yes_count - llm_no_count

        print(f"\nSUMMARY:")
        print(f"Sources checked: {len(results)}")
        print(f"Sources with new articles: {sources_with_articles}")
        print(f"Total new articles: {total_articles}")
        print(f"Total article-market pairs evaluated: {len(llm_results)}")
        print(f"LLM Results - Related: {llm_yes_count}, Not Related: {llm_no_count}, Invalid: {llm_invalid_count}")

        if len(llm_results) > 0:
            related_percentage = (llm_yes_count / len(llm_results)) * 100
            print(f"Percentage of markets deemed related by LLM: {related_percentage:.1f}%")

    except Exception as e:
        print(f"Error in main execution: {e}")


if __name__ == "__main__":
    main()