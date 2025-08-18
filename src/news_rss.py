import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import feedparser
import requests

from src.logger import setup_logging

# Initialize project-wide logging (configured in src.logger)
logger = setup_logging()

# Shared HTTP session
_http = requests.Session()
_http.headers.update({
    "User-Agent": "RSSNewsPoller/1.0 (+https://example.com; contact: you@example.com)"
})
DEFAULT_TIMEOUT = float(os.getenv("RSS_TIMEOUT_SECS", "10"))


# --------------------------------- Internal Utilities ----------------------------------

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


# ---------------------------- Formatting / Transform Helpers ----------------------------

def format_headlines(
    news_data: Dict[str, List[Dict[str, Any]]],
    include_summary: bool = False
) -> str:
    """
    Produce a human-readable string of headlines for ALL items passed in.
    Caller chooses whether to print or not. No printing here.
    """
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("LATEST NEWS HEADLINES (NEW ITEMS ONLY)")
    lines.append("=" * 60)

    for source, articles in news_data.items():
        lines.append(f"\n{source.upper()}")
        lines.append("-" * 30)
        for i, article in enumerate(articles, 1):
            lines.append(f"{i}. {article['title']}")
            lines.append(f"   Published: {article.get('published')}")
            lines.append(f"   Link: {article['link']}")
            if include_summary:
                summary = (article.get("summary") or "").strip()
                if summary:
                    lines.append(f"   Summary: {summary[:300]}{'...' if len(summary) > 300 else ''}")
            if article.get("categories"):
                lines.append(f"   Categories: {', '.join(article['categories'])}")
    return "\n".join(lines)

def pretty_print(
    news_data: Dict[str, List[Dict[str, Any]]],
    include_summary: bool = False
):
    """
    Print formatted headlines to stdout (console pretty print).
    """
    print(format_headlines(news_data, include_summary=include_summary))


def prepare_records(
    news_data: Dict[str, List[Dict[str, Any]]],
    include_summary: bool = False,
    flatten: bool = True,
) -> Dict[str, List[Dict[str, Any]]] | List[Dict[str, Any]]:
    """
    Prepare a slimmed-down payload for downstream (DB, API, etc).
    Always includes: title, published. Optionally includes: summary.
    Uses ALL items passed in; does NOT slice or decide counts.
    """
    def _project(article: Dict[str, Any]) -> Dict[str, Any]:
        rec = {
            "title": article.get("title"),
            "published": article.get("published"),
        }
        if include_summary:
            rec["summary"] = article.get("summary")
        return rec

    projected: Dict[str, List[Dict[str, Any]]] = {
        source: [_project(a) for a in items]
        for source, items in news_data.items()
    }

    if flatten:
        flat: List[Dict[str, Any]] = []
        for items in projected.values():
            flat.extend(items)
        return flat

    return projected


# ------------------------------------- Core Class --------------------------------------

class RSSNewsPoller:
    def __init__(
        self,
        feeds: Optional[Dict[str, str]] = None,
        state_path: Optional[str] = None,
        max_items_per_feed: int = 10,  # uniform cap requested; feeds may provide fewer
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
        self.max_items_per_feed = max_items_per_feed

        if self.state_path and os.path.exists(self.state_path):
            self._load_state()

    # ---------- persistence ----------
    def _load_state(self):
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.seen_keys = set(data.get("seen_keys", []))
            # For simplicity on reload: persist etag only; modified re-learned later.
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
                logger.info(f"{source_name}: 304 Not Modified (cached)")
                return []

            if getattr(feed, "bozo", False):
                logger.warning(f"Feed bozo flag set for {source_name} (may have parsing issues)")

            entries = list(getattr(feed, "entries", []))
            available = len(entries)
            if available < self.max_items_per_feed:
                logger.info(f"{source_name}: only {available} items available (requested {self.max_items_per_feed})")

            articles: List[Dict[str, Any]] = []
            for entry in entries[: self.max_items_per_feed]:
                pub = _safe_published(entry)
                categories = [
                    getattr(tag, "term", None) or str(tag)
                    for tag in (getattr(entry, "tags", []) or [])
                ]
                article = {
                    "key": make_article_key(entry),
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
        No printing inside this function.
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

    # ---------- loops ----------
    def run_continuous_polling(self, interval_minutes: int = 30, on_batch=None):
        """
        Continuous polling with no printing. If `on_batch` is provided, it's called
        as `on_batch(new_data)` each cycle for DB writes, webhooks, etc.
        """
        logger.info(f"Starting RSS polling (interval: {interval_minutes}m). Ctrl+C to stop.")
        try:
            while True:
                new_data = self.poll_all_feeds()
                if on_batch is not None:
                    try:
                        on_batch(new_data)
                    except Exception as e:
                        logger.warning(f"on_batch failed: {e}", exc_info=True)
                logger.info(f"Sleeping {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            logger.info("Polling stopped by user")

    def single_poll(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform a single poll of all feeds and return ONLY NEW articles.
        No printing here.
        """
        return self.poll_all_feeds()




# ------------------------------------- CLI Example -------------------------------------

if __name__ == "__main__":
    # Enforce a uniform cap; if a feed provides fewer, you'll see a log line.
    poller = RSSNewsPoller(state_path="rss_poller_state.json", max_items_per_feed=10)

    new_data = poller.single_poll()

    pretty_print(new_data, include_summary=False)

    # Minimal downstream records: title + published (no summary), for ALL items fetched
    records = prepare_records(new_data, include_summary=False, flatten=True)
    print(records)

    # If you want title + published + summary:
    # records_with_summary = prepare_records(new_data, include_summary=True, flatten=True)
    # print(records_with_summary)

    # If you want a human preview:
    # print(format_headlines(new_data, include_summary=False))
