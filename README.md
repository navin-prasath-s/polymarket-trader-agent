# polymarket-trader


## System Overview

This project is a **fully-automated Polymarket paper-trading system** powered by real-time news signals and vector similarity search. It simulates trading on prediction markets using a modular architecture of webhook ingestion, vector databases, local + cloud LLMs, and performance evaluation.

---

### Flow Summary

1. **Market Ingestion**  
   The paper-trading app emits webhook events for new markets. These are picked up by the trading bot, which:
   - Embeds the market title/description.
   - Stores the vectors in a **Qdrant DB** for future similarity search.

2. **News Matching**  
   The bot consumes breaking news from live **RSS feeds**. For each article:
   - It computes **cosine similarity** and **BM25 scores** against stored market vectors.
   - Matches news articles to potentially relevant markets.

3. **Relevance Judging (Local LLM)**  
   A **locally running Gemma-3 model** acts as a judge, deciding if the matched market-news pair is actually relevant.
   - If relevant, the bot proceeds to trade decisioning.

4. **LLM-based Trade Decision**  
   - Fetches additional metadata about the market via the **Polymarket API**.
   - Sends the combined data to the **OpenAI API**, which chooses whether to buy or sell **YES/NO shares**.

5. **Trade Execution & Logging**  
   - Executes trades in the **Polymarket paper trading app**.
   - Stores the trade actions (e.g. market ID, direction, rationale) in a local **database**.

6. **Post-Trade Monitoring & Evaluation**  
   - Monitors traded markets for **spike detection** using the Polymarket API.
   - Evaluates the system’s **accuracy and decision quality** over time.

---

> 🔧 This system is modular and can be extended with new embedding models, judge agents, or trading strategies.


![Screenshot](arch.png)
