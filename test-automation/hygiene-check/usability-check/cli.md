# Basic usage
python3 discover_pages.py --url https://ppdolphin.brobotinsurance.com

# Custom sitemap path
python3 discover_pages.py --url https://ppdolphin.brobotinsurance.com --sitemap /sitemap_index.xml

# Custom output file
python3 discover_pages.py --url https://ppdolphin.brobotinsurance.com --output dolphin-pages.txt

# With manual seed fallback pages
python3 discover_pages.py --url https://ppdolphin.brobotinsurance.com \
  --seeds /car-insurance /health-insurance /two-wheeler-insurance /about-us

# See all options
python3 discover_pages.py --help

# Fetch all asset types and merge into existing pages.txt
python3 fetch_dynamic_assets.py \
  --url https://ppdolphin.brobotinsurance.com \
  --user admin@liferay.com \
  --password yourpassword

# Only blogs (fastest check first)
python3 fetch_dynamic_assets.py \
  --url https://ppdolphin.brobotinsurance.com \
  --user admin@liferay.com \
  --password yourpassword \
  --types blogs

# Specific group IDs only
python3 fetch_dynamic_assets.py \
  --url https://ppdolphin.brobotinsurance.com \
  --user admin@liferay.com \
  --password yourpassword \
  --group-ids 20119

# Specific diagnosis using fetching by JSON WS when Headless services are blocked and we have to fallback on the JSON Web Services API

python3 fetch_dynamic_assets_by_jsonws.py \
  --url https://ppdolphin.brobotinsurance.com \
  --user iainfra@infoaxon.com \
  --password Server123@ \
  --diagnose

## What This Produces
```
pages.txt          ← merged sitemap + all asset URLs (your full 1776 target)
asset-urls.json    ← breakdown by type: blogs / web-content / kb / documents
