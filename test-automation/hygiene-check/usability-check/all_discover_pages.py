# discover_pages.py
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse, urlencode
import urllib3
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def safe_parse_xml(content, source_url):
    if not content or not content.strip():
        print(f"  [SKIP] Empty response from: {source_url}")
        return None
    try:
        return ET.fromstring(content)
    except ET.ParseError as e:
        print(f"  [SKIP] XML parse error from {source_url}: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# SOURCE 1: Sitemap XML (existing logic, kept as-is)
# ─────────────────────────────────────────────────────────────
def fetch_sitemap_urls(base_url: str, sitemap_path="/sitemap.xml") -> list[str]:
    sitemap_url = urljoin(base_url, sitemap_path)
    print(f"\n[SOURCE 1] Fetching sitemap: {sitemap_url}")

    try:
        resp = requests.get(sitemap_url, timeout=15, verify=False)
        root = safe_parse_xml(resp.content, sitemap_url)
    except Exception as e:
        print(f"  [ERROR] {e}")
        return []

    if root is None:
        fallback_url = urljoin(base_url, "/sitemap_index.xml")
        print(f"  Trying fallback: {fallback_url}")
        try:
            resp = requests.get(fallback_url, timeout=15, verify=False)
            root = safe_parse_xml(resp.content, fallback_url)
        except Exception as e:
            print(f"  [ERROR] {e}")
            return []

    if root is None:
        return []

    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = []

    child_sitemaps = root.findall("sm:sitemap/sm:loc", ns)
    if child_sitemaps:
        print(f"  Found sitemap index with {len(child_sitemaps)} child sitemaps")

        def fetch_child(loc):
            child_url = loc.text.strip()
            try:
                sub_resp = requests.get(child_url, timeout=15, verify=False)
                sub_root = safe_parse_xml(sub_resp.content, child_url)
                if sub_root:
                    return [u.text.strip() for u in sub_root.findall("sm:url/sm:loc", ns) if u.text]
            except Exception as e:
                print(f"  [SKIP] {child_url}: {e}")
            return []

        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = {ex.submit(fetch_child, loc): loc for loc in child_sitemaps}
            for future in as_completed(futures):
                urls += future.result()

    direct_urls = root.findall("sm:url/sm:loc", ns)
    if direct_urls:
        urls += [u.text.strip() for u in direct_urls if u.text]

    print(f"  -> {len(set(urls))} URLs from sitemap")
    return list(set(urls))


# ─────────────────────────────────────────────────────────────
# SOURCE 2: Liferay Headless Delivery API
# Fetches ALL pages including hidden, all groups, all layouts
# ─────────────────────────────────────────────────────────────
def fetch_liferay_api_urls(base_url: str, auth: tuple, group_ids: list[str] = None) -> list[str]:
    print(f"\n[SOURCE 2] Fetching pages via Liferay Headless API")
    all_urls = []

    # Auto-discover group IDs if not provided
    if not group_ids:
        group_ids = discover_group_ids(base_url, auth)

    for group_id in group_ids:
        print(f"  Scanning groupId: {group_id}")
        page_num = 1
        while True:
            api_url = f"{base_url}/o/headless-delivery/v1.0/sites/{group_id}/site-pages"
            params = {"page": page_num, "pageSize": 50, "flatten": True}
            try:
                resp = requests.get(api_url, params=params, auth=auth, verify=False, timeout=15)
                if resp.status_code == 401:
                    print(f"  [AUTH ERROR] Check your credentials for groupId {group_id}")
                    break
                if resp.status_code == 404:
                    print(f"  [SKIP] groupId {group_id} not accessible")
                    break
                data = resp.json()
            except Exception as e:
                print(f"  [ERROR] groupId {group_id} page {page_num}: {e}")
                break

            items = data.get("items", [])
            if not items:
                break

            for item in items:
                friendly_path = item.get("friendlyUrlPath", "")
                if friendly_path:
                    all_urls.append(base_url + friendly_path)

            last_page = data.get("lastPage", 1)
            print(f"    Page {page_num}/{last_page} - {len(items)} items", end="\r")
            if page_num >= last_page:
                break
            page_num += 1

        print(f"  -> groupId {group_id}: {len(all_urls)} total URLs so far")

    unique = list(set(all_urls))
    print(f"  -> {len(unique)} total URLs from Liferay API")
    return unique


def discover_group_ids(base_url: str, auth: tuple) -> list[str]:
    """Auto-discover all site group IDs via Liferay API"""
    print("  Auto-discovering site groups...")
    group_ids = []
    try:
        # Headless admin user API to list sites
        url = f"{base_url}/o/headless-admin-user/v1.0/sites"
        params = {"page": 1, "pageSize": 100}
        resp = requests.get(url, params=params, auth=auth, verify=False, timeout=15)
        data = resp.json()
        for site in data.get("items", []):
            gid = str(site.get("id", ""))
            name = site.get("name", "")
            if gid:
                print(f"    Found site: {name} (groupId: {gid})")
                group_ids.append(gid)
    except Exception as e:
        print(f"  [WARN] Could not auto-discover groups: {e}")

    if not group_ids:
        print("  [WARN] No groups found via API - try passing --group-ids manually")
    return group_ids


# ─────────────────────────────────────────────────────────────
# SOURCE 3: Recursive Spider / Crawler
# Follows href links across the site - catches pages with no
# sitemap entry and no API equivalent (e.g. redirect targets)
# ─────────────────────────────────────────────────────────────
def spider_site(base_url: str, max_pages: int = 2000, concurrency: int = 8) -> list[str]:
    from bs4 import BeautifulSoup
    from collections import deque

    print(f"\n[SOURCE 3] Spidering site (max {max_pages} pages)...")
    visited = set()
    queue = deque([base_url + "/"])
    domain = urlparse(base_url).netloc

    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 QA-Spider/1.0"

    def crawl(url):
        try:
            resp = session.get(url, timeout=10, verify=False, allow_redirects=True)
            if "text/html" not in resp.headers.get("Content-Type", ""):
                return []
            soup = BeautifulSoup(resp.text, "html.parser")
            found = []
            for a in soup.find_all("a", href=True):
                href = a["href"].split("#")[0].split("?")[0]  # strip anchors & params
                full = urljoin(url, href)
                parsed = urlparse(full)
                if parsed.netloc == domain and parsed.scheme in ("http", "https"):
                    clean = parsed.scheme + "://" + parsed.netloc + parsed.path.rstrip("/")
                    if clean not in visited:
                        found.append(clean)
            return found
        except Exception:
            return []

    while queue and len(visited) < max_pages:
        batch = []
        while queue and len(batch) < concurrency:
            url = queue.popleft()
            if url not in visited:
                visited.add(url)
                batch.append(url)

        if not batch:
            break

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {ex.submit(crawl, url): url for url in batch}
            for future in as_completed(futures):
                new_links = future.result()
                for link in new_links:
                    if link not in visited:
                        queue.append(link)

        print(f"  Visited: {len(visited)} | Queue: {len(queue)}", end="\r")

    print(f"\n  -> {len(visited)} URLs found via spider")
    return list(visited)


# ─────────────────────────────────────────────────────────────
# MERGE + DEDUPLICATE all sources
# ─────────────────────────────────────────────────────────────
def merge_and_report(sitemap_urls, api_urls, spider_urls, base_url):
    all_sets = {
        "sitemap": set(sitemap_urls),
        "api":     set(api_urls),
        "spider":  set(spider_urls),
    }

    merged = set()
    for s in all_sets.values():
        merged.update(s)

    # Only keep URLs belonging to this domain
    domain = urlparse(base_url).netloc
    merged = {u for u in merged if urlparse(u).netloc == domain}

    print(f"\n{'─'*50}")
    print(f"  Sitemap URLs  : {len(all_sets['sitemap'])}")
    print(f"  API URLs      : {len(all_sets['api'])}")
    print(f"  Spider URLs   : {len(all_sets['spider'])}")
    print(f"  TOTAL UNIQUE  : {len(merged)}")

    # Show what each source missed
    only_in_api    = all_sets['api']    - all_sets['sitemap']
    only_in_spider = all_sets['spider'] - all_sets['sitemap'] - all_sets['api']
    print(f"\n  Pages in API but NOT in sitemap    : {len(only_in_api)}")
    print(f"  Pages found by spider only         : {len(only_in_spider)}")
    print(f"{'─'*50}\n")

    return sorted(merged)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-source Liferay page discovery for usability testing.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Sitemap only (no auth needed)
  python3 discover_pages.py --url https://example.com

  # Sitemap + Liferay API (recommended)
  python3 discover_pages.py --url https://example.com --user admin@liferay.com --password test

  # Sitemap + API + Spider (most complete)
  python3 discover_pages.py --url https://example.com --user admin@liferay.com --password test --spider

  # Specify group IDs manually if auto-discovery fails
  python3 discover_pages.py --url https://example.com --user admin@liferay.com --password test --group-ids 20119 20200

  # Limit spider depth
  python3 discover_pages.py --url https://example.com --spider --max-pages 500
        """
    )
    parser.add_argument("--url",       required=True,  help="Base URL (e.g. https://example.com)")
    parser.add_argument("--sitemap",   default="/sitemap.xml", help="Sitemap path (default: /sitemap.xml)")
    parser.add_argument("--output",    default="pages.txt",    help="Output file (default: pages.txt)")
    parser.add_argument("--user",      default="",  help="Liferay admin email for API auth")
    parser.add_argument("--password",  default="",  help="Liferay admin password for API auth")
    parser.add_argument("--group-ids", nargs="*", default=[], metavar="ID",
                        help="Liferay site group IDs to scan (auto-discovered if not set)")
    parser.add_argument("--spider",    action="store_true",
                        help="Also spider the site by following links (slowest but most complete)")
    parser.add_argument("--max-pages", type=int, default=2000,
                        help="Max pages for spider (default: 2000)")
    parser.add_argument("--sources",   nargs="*", default=["sitemap", "api", "spider"],
                        choices=["sitemap", "api", "spider"],
                        help="Which sources to use (default: all three)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base_url = args.url.rstrip("/")

    sitemap_urls, api_urls, spider_urls = [], [], []

    # Source 1: Sitemap
    if "sitemap" in args.sources:
        sitemap_urls = fetch_sitemap_urls(base_url, args.sitemap)

    # Source 2: Liferay API
    if "api" in args.sources and args.user:
        auth = (args.user, args.password)
        api_urls = fetch_liferay_api_urls(base_url, auth, args.group_ids or None)
    elif "api" in args.sources and not args.user:
        print("\n[SOURCE 2] Skipped - pass --user and --password to enable API discovery")

    # Source 3: Spider
    if "spider" in args.sources or args.spider:
        spider_urls = spider_site(base_url, max_pages=args.max_pages)

    # Merge everything
    all_urls = merge_and_report(sitemap_urls, api_urls, spider_urls, base_url)

    if not all_urls:
        print("[FATAL] No URLs discovered from any source. Check connectivity and credentials.")
        sys.exit(1)

    with open(args.output, "w") as f:
        f.write("\n".join(all_urls))

    print(f"Done. {len(all_urls)} pages written to {args.output}")
