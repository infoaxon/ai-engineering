# fetch_dynamic_assets.py
# Fetches all dynamic Liferay asset URLs (blogs, web content, KB, documents)
# Uses JSON Web Services API as primary (works even when headless APIs are blocked)
# Falls back to sitemap groupId extraction if JSON WS is also unavailable.

import requests
import urllib3
import argparse
import json
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, parse_qs, urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ─────────────────────────────────────────────────────────────
# COMPANY ID DISCOVERY
# ─────────────────────────────────────────────────────────────

def discover_company_id(base_url, auth):
    """Fetch the Liferay companyId via JSON WS - required for group queries."""
    print("  Fetching companyId via JSON WS...")
    try:
        resp = requests.get(
            f"{base_url}/api/jsonws/company/get-companies",
            auth=auth, verify=False, timeout=15
        )
        print(f"  [company/get-companies] HTTP {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and data:
                cid = str(data[0].get("companyId", ""))
                print(f"  Found companyId: {cid}")
                return cid
    except Exception as e:
        print(f"  [ERROR] {e}")

    # Common default Liferay companyIds to try
    print("  Falling back to common default companyIds...")
    for cid in ["20116", "20099", "20098", "1"]:
        try:
            resp = requests.get(
                f"{base_url}/api/jsonws/group/get-groups",
                params={"companyId": cid, "parentGroupId": 0, "site": True},
                auth=auth, verify=False, timeout=10
            )
            if resp.status_code == 200 and isinstance(resp.json(), list):
                print(f"  companyId {cid} works!")
                return cid
        except Exception:
            continue

    print("  [WARN] Could not determine companyId automatically.")
    return None


# ─────────────────────────────────────────────────────────────
# SITE GROUP DISCOVERY - 4 fallback methods
# ─────────────────────────────────────────────────────────────

def discover_group_ids(base_url, auth):
    """
    Try 4 methods to discover site group IDs, in order of reliability.
    Method 1: JSON WS group/get-groups          (needs auth, always available)
    Method 2: JSON WS group/get-user-sites-groups (fallback if method 1 fails)
    Method 3: Headless delivery /sites          (blocked on some servers)
    Method 4: Extract groupIds from sitemap XML (zero auth needed, always works)
    """
    print("\nDiscovering site groups...")
    groups = []

    # ── Method 1: JSON WS group/get-groups ────────────────────
    company_id = discover_company_id(base_url, auth)
    if company_id:
        try:
            url    = f"{base_url}/api/jsonws/group/get-groups"
            params = {"companyId": company_id, "parentGroupId": 0, "site": True}
            resp   = requests.get(url, params=params, auth=auth, verify=False, timeout=15)
            print(f"  [Method 1] jsonws/group/get-groups -> HTTP {resp.status_code}")

            if resp.status_code == 200 and isinstance(resp.json(), list):
                for site in resp.json():
                    gid = str(site.get("groupId", ""))
                    if gid:
                        groups.append({
                            "id"          : gid,
                            "name"        : site.get("descriptiveName", site.get("nameCurrentValue", gid)),
                            "friendlyUrl" : site.get("friendlyURL", "").strip("/")
                        })
                if groups:
                    _print_groups(groups, "JSON WS get-groups")
                    return groups
        except Exception as e:
            print(f"  [Method 1 ERROR] {e}")

    # ── Method 2: JSON WS get-user-sites-groups ───────────────
    try:
        url  = f"{base_url}/api/jsonws/group/get-user-sites-groups"
        resp = requests.get(url, auth=auth, verify=False, timeout=15)
        print(f"  [Method 2] jsonws/get-user-sites-groups -> HTTP {resp.status_code}")

        if resp.status_code == 200 and isinstance(resp.json(), list):
            for site in resp.json():
                gid = str(site.get("groupId", ""))
                if gid:
                    groups.append({
                        "id"          : gid,
                        "name"        : site.get("descriptiveName", site.get("nameCurrentValue", gid)),
                        "friendlyUrl" : site.get("friendlyURL", "").strip("/")
                    })
            if groups:
                _print_groups(groups, "JSON WS get-user-sites-groups")
                return groups
    except Exception as e:
        print(f"  [Method 2 ERROR] {e}")

    # ── Method 3: Headless delivery API ───────────────────────
    for api_path in [
        "/o/headless-admin-user/v1.0/sites",
        "/o/headless-delivery/v1.0/sites"
    ]:
        try:
            resp = requests.get(
                f"{base_url}{api_path}",
                params={"pageSize": 100},
                auth=auth, verify=False, timeout=15
            )
            print(f"  [Method 3] {api_path} -> HTTP {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                for site in data.get("items", []):
                    gid = str(site.get("id", ""))
                    if gid:
                        groups.append({
                            "id"          : gid,
                            "name"        : site.get("name", gid),
                            "friendlyUrl" : site.get("friendlyUrlPath", "").strip("/")
                        })
                if groups:
                    _print_groups(groups, "Headless API")
                    return groups
        except Exception as e:
            print(f"  [Method 3 ERROR] {api_path}: {e}")

    # ── Method 4: Extract from sitemap XML (zero-permission fallback) ─
    print("  [Method 4] Extracting groupIds directly from sitemap XML...")
    try:
        groups = _extract_groups_from_sitemap(base_url)
        if groups:
            _print_groups(groups, "sitemap XML extraction")
            return groups
    except Exception as e:
        print(f"  [Method 4 ERROR] {e}")

    print("  [WARN] All discovery methods failed.")
    print("  Tip: Run with --diagnose then pass --group-ids manually.")
    return []


def _extract_groups_from_sitemap(base_url):
    """
    Parse the sitemap index and extract unique groupId values from child URLs.
    e.g. /sitemap.xml?groupId=20119&p_l_id=1877 -> groupId 20119
    This always works because we already confirmed the sitemap is accessible.
    """
    ns   = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    resp = requests.get(f"{base_url}/sitemap.xml", verify=False, timeout=15)
    root = ET.fromstring(resp.content)

    seen = {}
    for loc in root.findall("sm:sitemap/sm:loc", ns):
        href = loc.text.strip()
        qs   = parse_qs(urlparse(href).query)
        gid  = qs.get("groupId", [None])[0]
        if gid and gid not in seen:
            seen[gid] = {
                "id"          : gid,
                "name"        : f"Group-{gid}",
                "friendlyUrl" : "guest"
            }

    return list(seen.values())


def _print_groups(groups, source):
    print(f"  Found {len(groups)} group(s) via {source}:")
    for g in groups:
        print(f"    groupId: {g['id']:10s} | name: {g['name']:40s} | path: /{g['friendlyUrl']}")


# ─────────────────────────────────────────────────────────────
# GENERIC PAGINATORS
# ─────────────────────────────────────────────────────────────

def _paginate_jsonws(base_url, service_path, auth, params=None):
    """
    Paginate through a Liferay JSON WS endpoint.
    JSON WS uses -start/-end for pagination (not page/pageSize).
    """
    page_size = 50
    start     = 0

    while True:
        p = {"start": start, "end": start + page_size}
        if params:
            p.update(params)
        try:
            resp = requests.get(
                f"{base_url}/api/jsonws/{service_path}",
                params=p, auth=auth, verify=False, timeout=15
            )
            if resp.status_code in (401, 403):
                print(f"    [AUTH {resp.status_code}] {service_path}")
                break
            if resp.status_code == 404:
                break
            if resp.status_code != 200:
                print(f"    [HTTP {resp.status_code}] {service_path}")
                break

            data = resp.json()

            # JSON WS returns an error dict on failure
            if isinstance(data, dict) and "exception" in data:
                print(f"    [JWS ERROR] {service_path}: {data.get('exception', '')[:120]}")
                break

            if not isinstance(data, list) or not data:
                break

            yield from data

            if len(data) < page_size:
                break  # Last page reached
            start += page_size

        except Exception as e:
            print(f"    [ERROR] {service_path} start={start}: {e}")
            break


def _paginate_headless(base_url, api_path, auth, params=None):
    """
    Paginate through a Liferay Headless Delivery endpoint.
    Fallback when JSON WS doesn't support a given asset type.
    """
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
            if resp.status_code in (401, 403, 404):
                break
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


# ─────────────────────────────────────────────────────────────
# ASSET FETCHERS
# ─────────────────────────────────────────────────────────────

def fetch_blogs(base_url, auth, group):
    """
    Primary  : JSON WS  blogsentry/get-group-entries
    Fallback : Headless /o/headless-delivery/v1.0/sites/{id}/blog-postings
    URL pattern: /web/{site-path}/-/blogs/{url-title}
    """
    group_id  = group["id"]
    site_path = group["friendlyUrl"] or "guest"
    urls      = []
    jws_found = False

    for item in _paginate_jsonws(
        base_url, "blogsentry/get-group-entries", auth,
        params={"groupId": group_id, "status": 0}
    ):
        jws_found = True
        url_title = item.get("urlTitle", "").strip("/")
        eid       = item.get("entryId", "")
        if url_title:
            urls.append(f"{base_url}/web/{site_path}/-/blogs/{url_title}")
        elif eid:
            urls.append(f"{base_url}/web/{site_path}/-/blogs/{eid}")

    if not jws_found:
        for item in _paginate_headless(
            base_url,
            f"/o/headless-delivery/v1.0/sites/{group_id}/blog-postings",
            auth, params={"fields": "friendlyUrlPath,id"}
        ):
            slug = (item.get("friendlyUrlPath") or "").strip("/")
            bid  = item.get("id", "")
            if slug:
                urls.append(f"{base_url}/web/{site_path}/-/blogs/{slug}")
            elif bid:
                urls.append(f"{base_url}/web/{site_path}/-/blogs/{bid}")

    return urls


def fetch_web_content(base_url, auth, group):
    """
    Primary  : JSON WS  journalarticle/get-articles
    Fallback : Headless /o/headless-delivery/v1.0/sites/{id}/structured-contents
    URL pattern: /web/{site-path}/w/{url-title}
    """
    group_id  = group["id"]
    site_path = group["friendlyUrl"] or "guest"
    urls      = []
    jws_found = False

    for item in _paginate_jsonws(
        base_url, "journalarticle/get-articles", auth,
        params={"groupId": group_id, "status": 0}
    ):
        jws_found  = True
        url_title  = item.get("urlTitle", "").strip("/")
        article_id = item.get("id", item.get("resourcePrimKey", ""))
        if url_title:
            urls.append(f"{base_url}/web/{site_path}/w/{url_title}")
            urls.append(f"{base_url}/web/{site_path}/-/asset_publisher/content/{article_id}/content/{url_title}")

    if not jws_found:
        for item in _paginate_headless(
            base_url,
            f"/o/headless-delivery/v1.0/sites/{group_id}/structured-contents",
            auth, params={"fields": "friendlyUrlPath,id"}
        ):
            slug = (item.get("friendlyUrlPath") or "").strip("/")
            if slug:
                urls.append(f"{base_url}/web/{site_path}/w/{slug}")

    return urls


def fetch_kb_articles(base_url, auth, group):
    """
    Primary  : JSON WS  kbarticle/get-kb-articles
    Fallback : Headless /o/headless-delivery/v1.0/sites/{id}/knowledge-base-articles
    URL pattern: /web/{site-path}/-/knowledge_base/article/{url-title}
    """
    group_id  = group["id"]
    site_path = group["friendlyUrl"] or "guest"
    urls      = []
    jws_found = False

    for item in _paginate_jsonws(
        base_url, "kbarticle/get-kb-articles", auth,
        params={"groupId": group_id, "parentResourcePrimKey": 0, "status": 0}
    ):
        jws_found = True
        url_title = item.get("urlTitle", "").strip("/")
        if url_title:
            urls.append(f"{base_url}/web/{site_path}/-/knowledge_base/article/{url_title}")

    if not jws_found:
        for item in _paginate_headless(
            base_url,
            f"/o/headless-delivery/v1.0/sites/{group_id}/knowledge-base-articles",
            auth, params={"fields": "friendlyUrlPath,id"}
        ):
            slug = (item.get("friendlyUrlPath") or "").strip("/")
            if slug:
                urls.append(f"{base_url}/web/{site_path}/-/knowledge_base/article/{slug}")

    return urls


def fetch_documents(base_url, auth, group):
    """
    Primary  : JSON WS  dlfileentry/get-file-entries
    Returns public-facing document download URLs.
    """
    group_id = group["id"]
    urls     = []

    for item in _paginate_jsonws(
        base_url, "dlfileentry/get-file-entries", auth,
        params={"groupId": group_id, "folderId": 0, "status": 0}
    ):
        uuid  = item.get("uuid", "")
        feid  = item.get("fileEntryId", "")
        title = item.get("title", "unknown")
        if feid:
            urls.append(f"{base_url}/documents/{group_id}/0/{title}/{uuid}")

    return urls


# Registry - add new fetchers here
ASSET_FETCHERS = {
    "blogs"       : fetch_blogs,
    "web-content" : fetch_web_content,
    "kb-articles" : fetch_kb_articles,
    "documents"   : fetch_documents,
}


# ─────────────────────────────────────────────────────────────
# MASTER FETCH
# ─────────────────────────────────────────────────────────────

def fetch_all_assets(base_url, auth, group_ids_override=None, fetchers=None):
    if fetchers is None:
        fetchers = ASSET_FETCHERS

    groups = discover_group_ids(base_url, auth)

    if group_ids_override:
        filtered = [g for g in groups if g["id"] in group_ids_override]
        if filtered:
            groups = filtered
        else:
            # Build minimal group stubs if discovery returned nothing
            groups = [{"id": gid, "name": f"Group-{gid}", "friendlyUrl": "guest"}
                      for gid in group_ids_override]

    if not groups:
        print("[FATAL] No groups to scan. Pass --group-ids manually.")
        return {}

    results = {asset_type: [] for asset_type in fetchers}

    for group in groups:
        print(f"\n  ── Group: {group['name']} (id: {group['id']}) ──")
        for asset_type, fetcher_fn in fetchers.items():
            urls = fetcher_fn(base_url, auth, group)
            results[asset_type] += urls
            status = f"{len(urls)} URLs found" if urls else "none found"
            print(f"    {asset_type:<20} -> {status}")

    # Deduplicate per type
    for k in results:
        results[k] = list(set(results[k]))

    return results


# ─────────────────────────────────────────────────────────────
# MERGE WITH EXISTING pages.txt
# ─────────────────────────────────────────────────────────────

def merge_with_pages_file(asset_results, pages_file="pages.txt"):
    try:
        with open(pages_file) as f:
            existing = set(line.strip() for line in f if line.strip())
        print(f"\nLoaded {len(existing)} existing URLs from {pages_file}")
    except FileNotFoundError:
        existing = set()
        print(f"\nNo existing {pages_file} - starting fresh.")

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
# DIAGNOSTICS
# ─────────────────────────────────────────────────────────────

def run_diagnostics(base_url, auth):
    """Quick health check of all available Liferay API endpoints."""
    print(f"\n{'═'*60}")
    print(f"  API DIAGNOSTICS: {base_url}")
    print(f"{'═'*60}")

    endpoints = [
        ("JSON WS - build number",          "/api/jsonws/portal/get-build-number"),
        ("JSON WS - company list",           "/api/jsonws/company/get-companies"),
        ("JSON WS - user sites groups",      "/api/jsonws/group/get-user-sites-groups"),
        ("JSON WS - blogs (groupId=20119)",  "/api/jsonws/blogsentry/get-group-entries?groupId=20119&start=0&end=1&status=0"),
        ("JSON WS - journal articles",       "/api/jsonws/journalarticle/get-articles?groupId=20119&start=0&end=1&status=0"),
        ("JSON WS - kb articles",            "/api/jsonws/kbarticle/get-kb-articles?groupId=20119&parentResourcePrimKey=0&start=0&end=1&status=0"),
        ("JSON WS - documents",              "/api/jsonws/dlfileentry/get-file-entries?groupId=20119&folderId=0&start=0&end=1&status=0"),
        ("Headless - admin sites",           "/o/headless-admin-user/v1.0/sites"),
        ("Headless - delivery sites",        "/o/headless-delivery/v1.0/sites"),
        ("Sitemap",                          "/sitemap.xml"),
    ]

    for label, path in endpoints:
        try:
            resp = requests.get(f"{base_url}{path}", auth=auth, verify=False, timeout=10)
            ct   = resp.headers.get("Content-Type", "")[:35]
            # Peek at response to show if it's an error payload
            try:
                body = resp.json()
                if isinstance(body, dict) and "exception" in body:
                    note = f"  ⚠ exception: {body['exception'][:60]}"
                elif isinstance(body, dict) and "status" in body:
                    note = f"  ⚠ {body.get('status','')}: {body.get('title','')}"
                elif isinstance(body, list):
                    note = f"  ✓ {len(body)} items"
                elif isinstance(body, dict) and body:
                    note = f"  ✓ keys: {list(body.keys())[:5]}"
                else:
                    note = ""
            except Exception:
                note = f"  (non-JSON)"
            print(f"  {label:<40} HTTP {resp.status_code}  {note}")
        except Exception as e:
            print(f"  {label:<40} ERROR: {str(e)[:60]}")

    print(f"{'═'*60}\n")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch all dynamic Liferay asset URLs via JSON WS API.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Auto-discover groups and fetch all asset types
  python3 fetch_dynamic_assets.py \\
      --url https://example.com \\
      --user admin@liferay.com \\
      --password yourpassword

  # Run diagnostics only to see which APIs are available
  python3 fetch_dynamic_assets.py \\
      --url https://example.com \\
      --user admin@liferay.com \\
      --password yourpassword \\
      --diagnose

  # Blogs only with known group ID
  python3 fetch_dynamic_assets.py \\
      --url https://example.com \\
      --user admin@liferay.com \\
      --password yourpassword \\
      --types blogs \\
      --group-ids 20119
        """
    )
    parser.add_argument("--url",        required=True)
    parser.add_argument("--user",       required=True,  help="Liferay admin email")
    parser.add_argument("--password",   required=True,  help="Liferay admin password")
    parser.add_argument("--group-ids",  nargs="*", default=[], metavar="ID",
                        help="Liferay groupIds to scan (auto-discovered if not set)")
    parser.add_argument("--pages-file", default="pages.txt",
                        help="pages.txt to merge into (default: pages.txt)")
    parser.add_argument("--types",      nargs="*", default=list(ASSET_FETCHERS.keys()),
                        choices=list(ASSET_FETCHERS.keys()),
                        help=f"Asset types to fetch. Choices: {', '.join(ASSET_FETCHERS.keys())}")
    parser.add_argument("--diagnose",   action="store_true",
                        help="Run API diagnostics only - no fetching")
    return parser.parse_args()


if __name__ == "__main__":
    args     = parse_args()
    base_url = args.url.rstrip("/")
    auth     = (args.user, args.password)

    print(f"Target  : {base_url}")
    print(f"User    : {args.user}")
    print(f"Types   : {', '.join(args.types)}")
    print(f"Output  : {args.pages_file}")

    # Always run diagnostics first so you can see what APIs are reachable
    run_diagnostics(base_url, auth)

    if args.diagnose:
        print("Diagnostics complete. Re-run without --diagnose to fetch assets.")
    else:
        active_fetchers = {k: v for k, v in ASSET_FETCHERS.items() if k in args.types}

        asset_results = fetch_all_assets(
            base_url,
            auth,
            group_ids_override=args.group_ids or None,
            fetchers=active_fetchers
        )

        merge_with_pages_file(asset_results, pages_file=args.pages_file)
