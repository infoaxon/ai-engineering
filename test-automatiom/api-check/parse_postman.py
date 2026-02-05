#!/usr/bin/env python3
"""Parse Postman collection and generate YAML configuration."""

import json
import yaml
from pathlib import Path


def detect_content_type(body: str, options: dict) -> str:
    """Detect content type from body content and Postman options."""
    # Check Postman language option first
    raw_options = options.get('raw', {})
    language = raw_options.get('language', '').lower()

    if language == 'json':
        return 'application/json'
    elif language == 'xml':
        return 'text/xml'

    # Auto-detect from body content
    body_stripped = body.strip()
    if body_stripped.startswith('{') or body_stripped.startswith('['):
        return 'application/json'
    elif body_stripped.startswith('<'):
        return 'text/xml'

    return 'text/plain'


def parse_postman_collection(collection_path: str) -> dict:
    """Parse Postman collection and extract API definitions."""
    with open(collection_path, 'r') as f:
        collection = json.load(f)

    apis = []

    for item in collection.get('item', []):
        name = item.get('name', 'Unknown API')
        request = item.get('request', {})
        method = request.get('method', 'GET')

        # Extract URL
        url = request.get('url', {})
        if isinstance(url, str):
            url_str = url
        else:
            url_str = url.get('raw', '')

        # Extract headers
        headers = {}
        for header in request.get('header', []):
            if not header.get('disabled', False):
                headers[header.get('key', '')] = header.get('value', '')

        # Extract body
        body = request.get('body', {})
        raw_body = None
        content_type = None

        if body.get('mode') == 'raw':
            raw_body = body.get('raw', '')
            body_options = body.get('options', {})
            content_type = detect_content_type(raw_body, body_options)

        api_config = {
            'name': name,
            'url': url_str,
            'method': method,
            'check_error_field': True,
            'error_field': 'ErrorMessages',
            'latency_threshold_ms': 5000,
        }

        if headers:
            api_config['headers'] = headers

        if raw_body:
            api_config['raw_body'] = raw_body

        if content_type:
            api_config['content_type'] = content_type

        apis.append(api_config)

        print(f"  - {name}: {method} (Content-Type: {content_type})")

    return apis


def generate_yaml_config(apis: list, output_path: str):
    """Generate YAML configuration file."""
    config = {
        'settings': {
            'check_interval_minutes': 5,
            'default_timeout_seconds': 30,
            'latency_threshold_ms': 5000,
            'retention_days': 7,
        },
        'environments': {
            'dev': {
                'name': 'Development',
                'apis': apis,
            },
            'sit': {
                'name': 'System Integration Testing',
                'apis': apis,
            },
            'uat': {
                'name': 'User Acceptance Testing',
                'apis': apis,
            },
            'production': {
                'name': 'Production',
                'apis': apis,
            },
        }
    }

    # Custom representer for multiline strings
    def str_representer(dumper, data):
        if '\n' in data or len(data) > 100:
            return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
        return dumper.represent_scalar('tag:yaml.org,2002:str', data)

    yaml.add_representer(str, str_representer)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, width=1000)

    print(f"\nConfiguration written to {output_path}")


def main():
    collection_path = 'RGI_premium_APIs.postman_collection'
    output_path = 'config/apis.yaml'

    print(f"Parsing Postman collection: {collection_path}\n")
    print("APIs found:")
    apis = parse_postman_collection(collection_path)

    print(f"\nTotal: {len(apis)} APIs")
    print(f"\nGenerating YAML configuration...")
    generate_yaml_config(apis, output_path)


if __name__ == '__main__':
    main()
