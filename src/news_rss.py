import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional


import feedparser
import requests

from src.logger import setup_logging

logger = setup_logging()

_http = requests.Session()
_http.headers.update({
    "User-Agent": "RSSNewsPoller/1.0 (+https://example.com; contact: you@example.com)"
})
DEFAULT_TIMEOUT = float(os.getenv("RSS_TIMEOUT_SECS", "10"))


def _safe_published(entry) -> Dict[str, Optional[str]]:
    dt_str = getattr(entry, "published", None)
    tstruct = getattr(entry, "published_parsed", None)
    iso = None
    if tstruct:
        try:
            iso = datetime(*tstruct[:6]).isoformat()
        except Exception:
            iso = None
    return {"published": dt_str or "No date", "published_iso": iso}


def make_article_key(entry) -> str:
    """
    Build a stable fingerprint for an article.
    Priority: entry.id > link > title|published (last resort).
    """
    eid = getattr(entry, "id", None)
    if eid:
        return f"id::{eid}"
    link = getattr(entry, "link", None)
    if link:
        return f"link::{link}"
    title = getattr(entry, "title", "") or ""
    published = getattr(entry, "published", "") or ""
    return f"tp::{title.strip()}|{published.strip()}"


class RSSNewsPoller:
    def __init__(
        self,
        feeds: Optional[Dict[str, str]] = None,
        state_path: Optional[str] = None,
    ):
        self.rss_feeds: Dict[str, str] = feeds or {
            "BBC News": "https://feeds.bbci.co.uk/news/rss.xml",
            "NPR": "https://feeds.npr.org/1001/rss.xml",
        }

        self.state_path = state_path
        self.seen_keys: set[str] = set()
        # feed_state: per-source caching hints
        # { source_name: {"etag": str|None, "modified": time.struct_time|None} }
        self.feed_state: Dict[str, Dict[str, Any]] = {}

        if self.state_path and os.path.exists(self.state_path):
            self._load_state()

    # ---------- persistence ----------
    def _load_state(self):
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.seen_keys = set(data.get("seen_keys", []))
            # `modified` can’t be stored as struct_time; store as ISO and parse lazily (optional)
            self.feed_state = {k: {"etag": v.get("etag"), "modified": None}
                               for k, v in data.get("feed_state", {}).items()}
            logger.info(f"Loaded state from {self.state_path}: {len(self.seen_keys)} seen keys")
        except Exception as e:
            logger.warning(f"Failed to load state from {self.state_path}: {e}")

    def _save_state(self):
        if not self.state_path:
            return
        try:
            data = {
                "seen_keys": sorted(self.seen_keys),
                "feed_state": {k: {"etag": v.get("etag")} for k, v in self.feed_state.items()},
            }
            tmp = self.state_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.state_path)
        except Exception as e:
            logger.warning(f"Failed to save state to {self.state_path}: {e}")


    # ---------- fetching ----------
    def _parse_with_cache(self, source_name: str, feed_url: str):
        """
        Use etag/modified caching if we have it for this source.
        """
        etag = self.feed_state.get(source_name, {}).get("etag")
        modified = self.feed_state.get(source_name, {}).get("modified")

        # Optional HEAD to fail fast
        try:
            _http.head(feed_url, timeout=DEFAULT_TIMEOUT)
        except Exception as e:
            logger.warning(f"HEAD failed for {source_name}: {e}")

        feed = feedparser.parse(
            feed_url,
            agent=_http.headers.get("User-Agent"),
            etag=etag,
            modified=modified,
        )
        # Save new validators if present
        st = self.feed_state.setdefault(source_name, {})
        if getattr(feed, "etag", None):
            st["etag"] = feed.etag
        if getattr(feed, "modified", None):
            # feed.modified is a time.struct_time
            st["modified"] = feed.modified
        return feed

    def fetch_feed_data(self, feed_url: str, source_name: str) -> List[Dict[str, Any]]:
        """
        Fetch and parse RSS feed data from a single source (returns ALL from source, unscreened).
        """
        try:
            feed = self._parse_with_cache(source_name, feed_url)

            if getattr(feed, "status", None) == 304:
                # Not modified — nothing new
                logger.info(f"{source_name}: 304 Not Modified (cached)")
                return []

            if getattr(feed, "bozo", False):
                logger.warning(f"Feed bozo flag set for {source_name} (may have parsing issues)")

            articles: List[Dict[str, Any]] = []
            for entry in getattr(feed, "entries", [])[:20]:  # slightly higher cap
                pub = _safe_published(entry)
                categories = [
                    getattr(tag, "term", None) or str(tag)
                    for tag in (getattr(entry, "tags", []) or [])
                ]
                article = {
                    "key": make_article_key(entry),  # <-- fingerprint here
                    "source": source_name,
                    "title": getattr(entry, "title", None) or "No title",
                    "link": getattr(entry, "link", None) or "No link",
                    "summary": getattr(entry, "summary", None) or "No summary",
                    "author": getattr(entry, "author", None) or "Unknown author",
                    "categories": categories,
                    **pub,
                }
                articles.append(article)

            # Optional: sort newest first if ISO available
            articles.sort(key=lambda a: a.get("published_iso") or "", reverse=True)
            return articles

        except Exception as e:
            logger.error(f"Error fetching feed from {source_name}: {e}", exc_info=True)
            return []

    # ---------- polling ----------
    def poll_all_feeds(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Poll all RSS feeds and return ONLY NEW (de-duplicated) articles per source.
        """
        logger.info(f"Polling RSS feeds at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        all_new: Dict[str, List[Dict[str, Any]]] = {}

        for source_name, feed_url in self.rss_feeds.items():
            logger.info(f"Fetching from {source_name}...")
            articles = self.fetch_feed_data(feed_url, source_name)
            new_items = []

            for art in articles:
                k = art["key"]
                if k in self.seen_keys:
                    continue
                # first time seeing this item → emit it and remember it
                self.seen_keys.add(k)
                new_items.append(art)

            logger.info(f"{source_name}: {len(new_items)} new (out of {len(articles)} fetched)")
            all_new[source_name] = new_items

        # Persist dedupe + validators if configured
        self._save_state()
        return all_new

    # ---------- display ----------
    def display_latest_headlines(
        self,
        news_data: Dict[str, List[Dict[str, Any]]],
        max_per_source: int = 5,
        verbose: bool = False,
    ):
        """
        Display (pretty-print) the provided news_data (assumed de-duped).
        If verbose=True, also prints article summary and categories.
        """
        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("LATEST NEWS HEADLINES (NEW ITEMS ONLY)")
        lines.append("=" * 60)

        for source, articles in news_data.items():
            lines.append(f"\n{source.upper()}")
            lines.append("-" * 30)
            for i, article in enumerate(articles[:max_per_source], 1):
                lines.append(f"{i}. {article['title']}")
                lines.append(f"   Published: {article.get('published')}")
                lines.append(f"   Link: {article['link']}")

                if verbose:
                    # print summary and categories if available
                    summary = (article.get("summary") or "").strip()
                    if summary:
                        lines.append(f"   Summary: {summary[:300]}{'...' if len(summary) > 300 else ''}")
                    if article.get("categories"):
                        lines.append(f"   Categories: {', '.join(article['categories'])}")
                lines.append("")
        print("\n".join(lines))

    # ---------- loops ----------
    def run_continuous_polling(self, interval_minutes: int = 30):
        """
        Run continuous polling with specified interval (deduped between cycles).
        """
        logger.info(f"Starting RSS news polling (interval: {interval_minutes} minutes). Press Ctrl+C to stop.")
        try:
            while True:
                new_data = self.poll_all_feeds()
                self.display_latest_headlines(new_data)
                logger.info(f"Waiting {interval_minutes} minutes until next poll...")
                time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            logger.info("Polling stopped by user")

    def single_poll(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform a single poll of all feeds and return ONLY NEW articles.
        """
        new_data = self.poll_all_feeds()
        self.display_latest_headlines(new_data)
        return new_data


if __name__ == "__main__":
    # Example usage with persistence so duplicates are suppressed across restarts
    poller = RSSNewsPoller(state_path="rss_poller_state.json")


    # Option 2: Single poll (returns only unseen items)
    new_data = poller.single_poll()

    # Option 3: Continuous polling
    # poller.run_continuous_polling(interval_minutes=15)