# discover_pages.py
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urljoin
import urllib3
import argparse
import sys

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def safe_parse_xml(content, source_url):
    """Parse XML safely, return None if empty or malformed."""
    if not content or not content.strip():
        print(f"  [SKIP] Empty response from: {source_url}")
        return None
    try:
        return ET.fromstring(content)
    except ET.ParseError as e:
        print(f"  [SKIP] XML parse error from {source_url}: {e}")
        return None

def fetch_sitemap_urls(base_url: str, sitemap_path="/sitemap.xml") -> list[str]:
    sitemap_url = urljoin(base_url, sitemap_path)
    print(f"Fetching root sitemap: {sitemap_url}")

    resp = requests.get(sitemap_url, timeout=15, verify=False)
    root = safe_parse_xml(resp.content, sitemap_url)

    if root is None:
        print("[WARN] Primary sitemap failed. Trying /sitemap_index.xml ...")
        fallback_url = urljoin(base_url, "/sitemap_index.xml")
        resp = requests.get(fallback_url, timeout=15, verify=False)
        root = safe_parse_xml(resp.content, fallback_url)

    if root is None:
        print("[FATAL] Could not parse any sitemap.")
        return []

    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = []

    # Case 1: Sitemap Index
    child_sitemaps = root.findall("sm:sitemap/sm:loc", ns)
    if child_sitemaps:
        print(f"Found sitemap index with {len(child_sitemaps)} child sitemaps")
        for sitemap_loc in child_sitemaps:
            child_url = sitemap_loc.text.strip()
            print(f"  Fetching child sitemap: {child_url}")
            try:
                sub_resp = requests.get(child_url, timeout=15, verify=False)
                sub_root = safe_parse_xml(sub_resp.content, child_url)
                if sub_root is not None:
                    found = [u.text.strip() for u in sub_root.findall("sm:url/sm:loc", ns) if u.text]
                    print(f"    -> {len(found)} URLs found")
                    urls += found
            except requests.RequestException as e:
                print(f"  [SKIP] Request failed for {child_url}: {e}")

    # Case 2: Direct sitemap
    direct_urls = root.findall("sm:url/sm:loc", ns)
    if direct_urls:
        found = [u.text.strip() for u in direct_urls if u.text]
        print(f"Found {len(found)} URLs directly in sitemap")
        urls += found

    return list(set(urls))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Discover all pages of a website via sitemap for usability testing.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python3 discover_pages.py --url https://example.com
  python3 discover_pages.py --url https://example.com --sitemap /custom-sitemap.xml
  python3 discover_pages.py --url https://example.com --output my-pages.txt
  python3 discover_pages.py --url https://example.com --seeds /car-insurance /health-insurance
        """
    )
    parser.add_argument(
        "--url",
        required=True,
        help="Base URL of the website to scan (e.g. https://example.com)"
    )
    parser.add_argument(
        "--sitemap",
        default="/sitemap.xml",
        help="Sitemap path relative to base URL (default: /sitemap.xml)"
    )
    parser.add_argument(
        "--output",
        default="pages.txt",
        help="Output file to write discovered URLs (default: pages.txt)"
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        default=[],
        metavar="PATH",
        help="Fallback seed paths if sitemap fails (e.g. /car-insurance /home)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Normalize base URL - strip trailing slash
    base_url = args.url.rstrip("/")

    urls = fetch_sitemap_urls(base_url, sitemap_path=args.sitemap)

    if not urls:
        print("\n[WARNING] No URLs discovered from sitemap.")
        if args.seeds:
            urls = [base_url + path for path in args.seeds]
            print(f"Using {len(urls)} provided seed URLs as fallback.")
        else:
            # Built-in generic insurance page seeds as last resort
            default_seeds = [
                "/", "/car-insurance", "/health-insurance",
                "/two-wheeler-insurance", "/travel-insurance",
                "/home-insurance", "/contact-us", "/about-us"
            ]
            urls = [base_url + path for path in default_seeds]
            print(f"No seeds provided. Using {len(urls)} default insurance page seeds.")
            print("Tip: pass --seeds /your-page /another-page for custom fallback paths.")

    with open(args.output, "w") as f:
        f.write("\n".join(urls))

    print(f"\nDone. {len(urls)} pages written to {args.output}")
