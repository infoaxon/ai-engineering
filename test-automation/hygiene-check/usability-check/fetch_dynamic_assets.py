# fetch_dynamic_assets.py
# Fetches all dynamic Liferay asset URLs (blogs, web content, KB, documents)
# and merges them into an existing pages.txt for usability testing pipelines.

import requests
import urllib3
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ─────────────────────────────────────────────────────────────
# SITE GROUP DISCOVERY
# ─────────────────────────────────────────────────────────────

def discover_group_ids(base_url, auth):
    """Get all site group IDs and their friendly URLs via Liferay admin API."""
    print("\nDiscovering site groups...")
    groups = []
    try:
        url  = f"{base_url}/o/headless-admin-user/v1.0/sites"
        resp = requests.get(url, params={"pageSize": 100}, auth=auth, verify=False, timeout=15)

        if resp.status_code == 401:
            print("  [AUTH ERROR] Invalid credentials. Check --user and --password.")
            return []
        if resp.status_code == 403:
            print("  [FORBIDDEN] User does not have admin access to list sites.")
            return []

        data = resp.json()
        for site in data.get("items", []):
            group = {
                "id"          : str(site.get("id", "")),
                "name"        : site.get("name", ""),
                "friendlyUrl" : site.get("friendlyUrlPath", "").strip("/")
            }
            if group["id"]:
                groups.append(group)
                print(f"  Found site: {group['name']:40s} | groupId: {group['id']:10s} | path: /{group['friendlyUrl']}")

    except Exception as e:
        print(f"  [ERROR] Could not discover groups: {e}")

    if not groups:
        print("  [WARN] No groups discovered. Use --group-ids to specify manually.")

    return groups


# ─────────────────────────────────────────────────────────────
# ASSET FETCHERS
# ─────────────────────────────────────────────────────────────

def _paginate(base_url, api_path, auth, params=None):
    """Generic paginated GET helper. Yields items across all pages."""
    page = 1
    while True:
        p = {"page": page, "pageSize": 50}
        if params:
            p.update(params)
        try:
            resp = requests.get(
                f"{base_url}{api_path}",
                params=p, auth=auth, verify=False, timeout=15
            )
            if resp.status_code in (401, 403):
                print(f"    [AUTH ERROR] {api_path}")
                break
            if resp.status_code == 404:
                break  # Asset type not present in this group - skip silently
            if resp.status_code != 200:
                print(f"    [HTTP {resp.status_code}] {api_path}")
                break

            data  = resp.json()
            items = data.get("items", [])
            if not items:
                break

            yield from items

            if page >= data.get("lastPage", 1):
                break
            page += 1

        except Exception as e:
            print(f"    [ERROR] {api_path} page {page}: {e}")
            break


def fetch_blogs(base_url, auth, group):
    """
    GET /o/headless-delivery/v1.0/sites/{siteId}/blog-postings
    URL pattern: /web/{site-path}/-/blogs/{friendly-url-slug}
    """
    site_path = group["friendlyUrl"] or "guest"
    urls      = []

    for item in _paginate(
        base_url,
        f"/o/headless-delivery/v1.0/sites/{group['id']}/blog-postings",
        auth,
        params={"fields": "friendlyUrlPath,id,headline"}
    ):
        slug = (item.get("friendlyUrlPath") or "").strip("/")
        bid  = item.get("id", "")
        if slug:
            urls.append(f"{base_url}/web/{site_path}/-/blogs/{slug}")
        elif bid:
            # Fallback if no friendly URL is set
            urls.append(f"{base_url}/web/{site_path}/-/blogs/{bid}")

    return urls


def fetch_web_content(base_url, auth, group):
    """
    GET /o/headless-delivery/v1.0/sites/{siteId}/structured-contents
    URL pattern: /web/{site-path}/w/{friendly-url-slug}
    """
    site_path = group["friendlyUrl"] or "guest"
    urls      = []

    for item in _paginate(
        base_url,
        f"/o/headless-delivery/v1.0/sites/{group['id']}/structured-contents",
        auth,
        params={"fields": "friendlyUrlPath,id,title"}
    ):
        slug = (item.get("friendlyUrlPath") or "").strip("/")
        cid  = item.get("id", "")
        if slug:
            urls.append(f"{base_url}/web/{site_path}/w/{slug}")
            # Also include asset publisher URL as fallback variant
            urls.append(f"{base_url}/web/{site_path}/-/asset_publisher/content/{cid}/content/{slug}")
        elif cid:
            urls.append(f"{base_url}/web/{site_path}/-/asset_publisher/content/{cid}")

    return urls


def fetch_kb_articles(base_url, auth, group):
    """
    GET /o/headless-delivery/v1.0/sites/{siteId}/knowledge-base-articles
    URL pattern: /web/{site-path}/-/knowledge_base/article/{friendly-url-slug}
    """
    site_path = group["friendlyUrl"] or "guest"
    urls      = []

    for item in _paginate(
        base_url,
        f"/o/headless-delivery/v1.0/sites/{group['id']}/knowledge-base-articles",
        auth,
        params={"fields": "friendlyUrlPath,id,title"}
    ):
        slug = (item.get("friendlyUrlPath") or "").strip("/")
        if slug:
            urls.append(f"{base_url}/web/{site_path}/-/knowledge_base/article/{slug}")

    return urls


def fetch_documents(base_url, auth, group):
    """
    GET /o/headless-delivery/v1.0/sites/{siteId}/documents
    Returns public-facing content/download URLs.
    """
    urls = []

    for item in _paginate(
        base_url,
        f"/o/headless-delivery/v1.0/sites/{group['id']}/documents",
        auth,
        params={"fields": "contentUrl,id,title"}
    ):
        content_url = item.get("contentUrl", "")
        if content_url:
            full = content_url if content_url.startswith("http") else base_url + content_url
            urls.append(full)

    return urls


# Registry of all available asset fetchers
ASSET_FETCHERS = {
    "blogs"       : fetch_blogs,
    "web-content" : fetch_web_content,
    "kb-articles" : fetch_kb_articles,
    "documents"   : fetch_documents,
}


# ─────────────────────────────────────────────────────────────
# MASTER FETCH - runs all asset types across all groups
# ─────────────────────────────────────────────────────────────

def fetch_all_assets(base_url, auth, group_ids_override=None, fetchers=None):
    """
    Discover all groups (or use overrides), then run each fetcher
    across every group in parallel.
    """
    if fetchers is None:
        fetchers = ASSET_FETCHERS

    groups = discover_group_ids(base_url, auth)

    if group_ids_override:
        groups = [g for g in groups if g["id"] in group_ids_override]
        if not groups:
            # Build minimal group objects from IDs if discovery returned nothing
            groups = [{"id": gid, "name": gid, "friendlyUrl": "guest"} for gid in group_ids_override]

    if not groups:
        print("[FATAL] No groups to scan. Pass --group-ids manually.")
        return {}

    results = {asset_type: [] for asset_type in fetchers}

    for group in groups:
        print(f"\n  ── Group: {group['name']} (id: {group['id']}) ──")
        for asset_type, fetcher_fn in fetchers.items():
            urls = fetcher_fn(base_url, auth, group)
            results[asset_type] += urls
            status = f"{len(urls)} URLs" if urls else "none found"
            print(f"    {asset_type:20s} -> {status}")

    # Deduplicate per type
    for k in results:
        results[k] = list(set(results[k]))

    return results


# ─────────────────────────────────────────────────────────────
# MERGE WITH EXISTING pages.txt
# ─────────────────────────────────────────────────────────────

def merge_with_pages_file(asset_results, pages_file="pages.txt"):
    """Merge discovered asset URLs into existing pages.txt and report delta."""
    try:
        with open(pages_file) as f:
            existing = set(line.strip() for line in f if line.strip())
        print(f"\nLoaded {len(existing)} existing URLs from {pages_file}")
    except FileNotFoundError:
        existing = set()
        print(f"\nNo existing {pages_file} found - starting fresh.")

    all_new = set()
    for urls in asset_results.values():
        all_new.update(urls)

    before = len(existing)
    merged = existing | all_new
    added  = len(merged) - before

    print(f"\n{'─'*55}")
    print(f"  {'Source':<25} {'Discovered':>12}")
    print(f"  {'─'*25} {'─'*12}")
    for asset_type, urls in asset_results.items():
        print(f"  {asset_type:<25} {len(urls):>12}")
    print(f"  {'─'*25} {'─'*12}")
    print(f"  {'Existing (pages.txt)':<25} {before:>12}")
    print(f"  {'Net new URLs added':<25} {added:>12}")
    print(f"  {'TOTAL MERGED':<25} {len(merged):>12}")
    print(f"{'─'*55}")

    with open(pages_file, "w") as f:
        f.write("\n".join(sorted(merged)))

    with open("asset-urls.json", "w") as f:
        json.dump({k: sorted(v) for k, v in asset_results.items()}, f, indent=2)

    print(f"\nSaved {len(merged)} URLs to {pages_file}")
    print(f"Saved asset breakdown to asset-urls.json")

    return merged


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch all dynamic Liferay asset URLs (blogs, web content, KB, documents).",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Fetch all asset types
  python3 fetch_dynamic_assets.py \\
      --url https://example.com \\
      --user admin@liferay.com \\
      --password yourpassword

  # Blogs only
  python3 fetch_dynamic_assets.py \\
      --url https://example.com \\
      --user admin@liferay.com \\
      --password yourpassword \\
      --types blogs

  # Specific group IDs (skip auto-discovery)
  python3 fetch_dynamic_assets.py \\
      --url https://example.com \\
      --user admin@liferay.com \\
      --password yourpassword \\
      --group-ids 20119 20200

  # Custom pages file
  python3 fetch_dynamic_assets.py \\
      --url https://example.com \\
      --user admin@liferay.com \\
      --password yourpassword \\
      --pages-file dolphin-pages.txt
        """
    )
    parser.add_argument("--url",        required=True,  help="Base URL (e.g. https://example.com)")
    parser.add_argument("--user",       required=True,  help="Liferay admin email")
    parser.add_argument("--password",   required=True,  help="Liferay admin password")
    parser.add_argument("--group-ids",  nargs="*", default=[], metavar="ID",
                        help="Liferay site group IDs to scan (auto-discovered if not set)")
    parser.add_argument("--pages-file", default="pages.txt",
                        help="pages.txt file to merge results into (default: pages.txt)")
    parser.add_argument("--types",      nargs="*", default=list(ASSET_FETCHERS.keys()),
                        choices=list(ASSET_FETCHERS.keys()),
                        help=f"Asset types to fetch (default: all). Choices: {', '.join(ASSET_FETCHERS.keys())}")
    return parser.parse_args()


if __name__ == "__main__":
    args     = parse_args()
    base_url = args.url.rstrip("/")
    auth     = (args.user, args.password)

    # Only run fetchers for requested asset types
    active_fetchers = {k: v for k, v in ASSET_FETCHERS.items() if k in args.types}

    print(f"Target  : {base_url}")
    print(f"User    : {args.user}")
    print(f"Types   : {', '.join(active_fetchers.keys())}")
    print(f"Output  : {args.pages_file}")

    asset_results = fetch_all_assets(
        base_url,
        auth,
        group_ids_override=args.group_ids or None,
        fetchers=active_fetchers
    )

    merge_with_pages_file(asset_results, pages_file=args.pages_file)
