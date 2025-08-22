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


def process_with_llm(results: Dict[str, List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]]) -> Tuple[
    Dict[str, str], List[Dict[str, Any]]]:
    """
    Process all article-market pairs through the LLM and extract related pairs.

    Args:
        results: Dictionary of articles and their matching markets

    Returns:
        Tuple of (llm_results dict, list of related condition_id/market/news pairs)
    """
    llm_results = {}
    related_pairs = []
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

                # If LLM says yes, add to related pairs with condition_id
                if llm_result.lower() == 'yes':
                    related_pair = {
                        "condition_id": market.get("condition_id"),
                        "market": {
                            "question": market.get("question"),
                            "description": market.get("description"),
                            "tokens": market.get("tokens", []),
                            "similarity_score": market.get("similarity_score")
                        },
                        "news": {
                            "title": article.get("title"),
                            "summary": article.get("summary"),
                            "link": article.get("link"),
                            "published": article.get("published"),
                            "source": source_name
                        }
                    }
                    related_pairs.append(related_pair)

                processed_pairs += 1
                if processed_pairs % 5 == 0 or processed_pairs == total_pairs:
                    print(f"Progress: {processed_pairs}/{total_pairs} pairs processed")

    return llm_results, related_pairs


def format_results_with_condition_id(results: Dict[str, List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]],
                                     llm_results: Dict[str, str]) -> str:
    """
    Format the news-market matching results for display, including condition_id.

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
                    condition_id = market.get('condition_id', 'No ID')

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

                    lines.append(f"      {j}. [{score_pct:.1f}%] ID:{condition_id} | {market_question}{llm_result}")
            else:
                lines.append(f"\n    No matching markets found.")

            lines.append("-" * 60)

    return "\n".join(lines)


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
        print("Fetching news and matching markets...")
        results = matcher.fetch_news_and_match_markets(
            rss_feeds=default_feeds,
            state_path="rss_poller_state.json",
            max_items_per_feed=10,
            top_k_markets=5
        )

        # Process through LLM and get related pairs
        llm_results, related_pairs = process_with_llm(results)

        # Format and print results with LLM evaluation and condition_id
        formatted_output = format_results_with_condition_id(results, llm_results)
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

        # Print related pairs summary
        print(f"\nRELATED PAIRS (LLM said 'yes'):")
        print(f"Total related condition_id/market/news pairs: {len(related_pairs)}")

        if related_pairs:
            print("\nFirst few related pairs:")
            for i, pair in enumerate(related_pairs[:3], 1):
                print(f"  {i}. Condition ID: {pair['condition_id']}")
                print(f"     Market: {pair['market']['question'][:80]}...")
                print(f"     News: {pair['news']['title'][:80]}...")
                print()

        # Return the related pairs for further processing if needed
        return related_pairs

    except Exception as e:
        print(f"Error in main execution: {e}")
        return []


if __name__ == "__main__":
    related_pairs = main()