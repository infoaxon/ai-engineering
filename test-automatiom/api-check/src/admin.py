"""Admin routes for managing customers and APIs."""

import json
import os
import re
import secrets
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile, File, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates

from .config import CustomerManager
from .storage import Storage

admin_router = APIRouter(prefix="/admin", tags=["admin"])
security = HTTPBasic()

templates: Jinja2Templates = None
customer_manager: CustomerManager = None
storage: Storage = None
base_dir: Path = None


def setup_admin_routes(
    _templates: Jinja2Templates,
    _customer_manager: CustomerManager,
    _storage: Storage,
    _base_dir: Path
) -> None:
    """Initialize admin route dependencies."""
    global templates, customer_manager, storage, base_dir
    templates = _templates
    customer_manager = _customer_manager
    storage = _storage
    base_dir = _base_dir


def verify_admin(credentials: Annotated[HTTPBasicCredentials, Depends(security)]) -> str:
    """Verify admin credentials using HTTP Basic Auth."""
    admin_username = os.environ.get("ADMIN_USERNAME", "admin")
    admin_password = os.environ.get("ADMIN_PASSWORD", "admin123")

    username_correct = secrets.compare_digest(
        credentials.username.encode("utf-8"),
        admin_username.encode("utf-8")
    )
    password_correct = secrets.compare_digest(
        credentials.password.encode("utf-8"),
        admin_password.encode("utf-8")
    )

    if not (username_correct and password_correct):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials.username


def slugify(text: str) -> str:
    """Convert text to URL-safe slug."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '_', text)
    return text


def extract_name_from_url(url: str) -> str:
    """Extract a human-readable name from a URL."""
    parsed = urlparse(url)
    path = parsed.path.strip('/')

    if path:
        parts = path.split('/')
        name_parts = []
        for part in parts[-2:]:
            part = re.sub(r'[_-]', ' ', part)
            name_parts.append(part.title())
        return ' '.join(name_parts) or 'API'

    return parsed.netloc or 'Unknown API'


def parse_simple_url_list(text: str) -> list[dict]:
    """Parse simple URL list (one URL per line).

    Format:
        https://api.example.com/health
        https://api.example.com/users

    Returns list of API configs with auto-generated names.
    """
    apis = []
    lines = text.strip().split('\n')

    for line in lines:
        url = line.strip()
        if not url or url.startswith('#'):
            continue

        name = extract_name_from_url(url)

        apis.append({
            'name': name,
            'url': url,
            'method': 'POST',
            'check_error_field': True,
            'error_field': 'ErrorMessages',
            'latency_threshold_ms': 5000,
        })

    return apis


def parse_name_url_format(text: str) -> list[dict]:
    """Parse Name|URL pipe-separated format.

    Format:
        Health Check|https://api.example.com/health
        User Service|https://api.example.com/users

    Returns list of API configs.
    """
    apis = []
    lines = text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        if '|' in line:
            parts = line.split('|', 1)
            name = parts[0].strip()
            url = parts[1].strip()
        else:
            url = line
            name = extract_name_from_url(url)

        apis.append({
            'name': name,
            'url': url,
            'method': 'POST',
            'check_error_field': True,
            'error_field': 'ErrorMessages',
            'latency_threshold_ms': 5000,
        })

    return apis


def create_postman_collection(customer_id: str, customer_name: str, apis: list[dict]) -> dict:
    """Create a Postman collection format from API list."""
    items = []

    for api in apis:
        item = {
            'name': api['name'],
            'request': {
                'method': api.get('method', 'POST'),
                'url': {
                    'raw': api['url']
                },
                'header': []
            }
        }

        if api.get('headers'):
            for key, value in api['headers'].items():
                item['request']['header'].append({
                    'key': key,
                    'value': value
                })

        if api.get('raw_body'):
            item['request']['body'] = {
                'mode': 'raw',
                'raw': api['raw_body'],
                'options': {
                    'raw': {
                        'language': 'json' if api.get('content_type') == 'application/json' else 'text'
                    }
                }
            }

        items.append(item)

    return {
        'info': {
            'name': customer_name,
            '_postman_id': customer_id,
            'schema': 'https://schema.getpostman.com/json/collection/v2.1.0/collection.json'
        },
        'item': items
    }


def save_collection_file(customer_id: str, customer_name: str, apis: list[dict]) -> str:
    """Save APIs as a Postman-format JSON collection file.

    Returns the relative path to the collection file.
    """
    collection = create_postman_collection(customer_id, customer_name, apis)

    collections_dir = base_dir / 'collections'
    collections_dir.mkdir(exist_ok=True)

    filename = f"{customer_id}.postman_collection.json"
    file_path = collections_dir / filename

    with open(file_path, 'w') as f:
        json.dump(collection, f, indent=2)

    return f"collections/{filename}"


@admin_router.get("", response_class=HTMLResponse)
async def admin_dashboard(request: Request, username: str = Depends(verify_admin)):
    """Admin dashboard showing all customers."""
    customers = customer_manager.customers

    customer_list = []
    for customer_id, customer_config in customers.items():
        config = customer_manager.get_all_configs().get(customer_id)
        api_count = 0
        if config:
            for env_config in config.environments.values():
                api_count = len(env_config.apis)
                break

        customer_list.append({
            'id': customer_id,
            'name': customer_config.name,
            'description': customer_config.description,
            'environments': customer_config.environments,
            'api_count': api_count,
            'active': customer_config.active
        })

    return templates.TemplateResponse(
        "admin/dashboard.html",
        {
            "request": request,
            "customers": customer_list,
            "username": username
        }
    )


@admin_router.get("/customer/new", response_class=HTMLResponse)
async def new_customer_form(request: Request, username: str = Depends(verify_admin)):
    """Display new customer form."""
    return templates.TemplateResponse(
        "admin/customer_form.html",
        {
            "request": request,
            "customer": None,
            "is_new": True,
            "username": username
        }
    )


@admin_router.post("/customer/new", response_class=HTMLResponse)
async def create_customer(
    request: Request,
    username: str = Depends(verify_admin),
    name: str = Form(...),
    description: str = Form(""),
    customer_id: str = Form(""),
    environments: list[str] = Form(default=["dev", "sit", "uat", "production"]),
    api_format: str = Form("simple"),
    api_text: str = Form(""),
    postman_file: UploadFile | None = File(None)
):
    """Create a new customer."""
    if not customer_id:
        customer_id = slugify(name)

    if customer_id in customer_manager.customers:
        return templates.TemplateResponse(
            "admin/customer_form.html",
            {
                "request": request,
                "customer": None,
                "is_new": True,
                "username": username,
                "error": f"Customer ID '{customer_id}' already exists"
            }
        )

    apis = []
    collection_path = ""

    if api_format == "postman" and postman_file and postman_file.filename:
        content = await postman_file.read()
        collection_data = json.loads(content.decode('utf-8'))

        collections_dir = base_dir / 'collections'
        collections_dir.mkdir(exist_ok=True)
        filename = f"{customer_id}.postman_collection.json"
        file_path = collections_dir / filename

        with open(file_path, 'w') as f:
            json.dump(collection_data, f, indent=2)

        collection_path = f"collections/{filename}"
    elif api_text.strip():
        if api_format == "simple":
            apis = parse_simple_url_list(api_text)
        elif api_format == "name_url":
            apis = parse_name_url_format(api_text)

        if apis:
            collection_path = save_collection_file(customer_id, name, apis)

    if not collection_path:
        return templates.TemplateResponse(
            "admin/customer_form.html",
            {
                "request": request,
                "customer": None,
                "is_new": True,
                "username": username,
                "error": "No APIs provided. Please enter URLs or upload a Postman collection."
            }
        )

    success = customer_manager.add_customer(
        customer_id=customer_id,
        name=name,
        description=description,
        postman_collection=collection_path,
        environments=environments
    )

    if not success:
        return templates.TemplateResponse(
            "admin/customer_form.html",
            {
                "request": request,
                "customer": None,
                "is_new": True,
                "username": username,
                "error": "Failed to create customer"
            }
        )

    return RedirectResponse(url="/admin?success=created", status_code=303)


@admin_router.get("/customer/{customer_id}", response_class=HTMLResponse)
async def edit_customer_form(
    request: Request,
    customer_id: str,
    username: str = Depends(verify_admin)
):
    """Display edit customer form."""
    if customer_id not in customer_manager.customers:
        raise HTTPException(status_code=404, detail="Customer not found")

    customer_config = customer_manager.customers[customer_id]
    apis = customer_manager.get_customer_apis(customer_id)

    api_text = ""
    for api in apis:
        api_text += f"{api['name']}|{api['url']}\n"

    customer = {
        'id': customer_id,
        'name': customer_config.name,
        'description': customer_config.description,
        'environments': customer_config.environments,
        'api_text': api_text.strip()
    }

    return templates.TemplateResponse(
        "admin/customer_form.html",
        {
            "request": request,
            "customer": customer,
            "is_new": False,
            "username": username
        }
    )


@admin_router.post("/customer/{customer_id}", response_class=HTMLResponse)
async def update_customer(
    request: Request,
    customer_id: str,
    username: str = Depends(verify_admin),
    name: str = Form(...),
    description: str = Form(""),
    environments: list[str] = Form(default=["dev", "sit", "uat", "production"]),
    api_format: str = Form("name_url"),
    api_text: str = Form(""),
    postman_file: UploadFile | None = File(None)
):
    """Update an existing customer."""
    if customer_id not in customer_manager.customers:
        raise HTTPException(status_code=404, detail="Customer not found")

    apis = []
    collection_path = ""

    if api_format == "postman" and postman_file and postman_file.filename:
        content = await postman_file.read()
        collection_data = json.loads(content.decode('utf-8'))

        collections_dir = base_dir / 'collections'
        collections_dir.mkdir(exist_ok=True)
        filename = f"{customer_id}.postman_collection.json"
        file_path = collections_dir / filename

        with open(file_path, 'w') as f:
            json.dump(collection_data, f, indent=2)

        collection_path = f"collections/{filename}"
    elif api_text.strip():
        if api_format == "simple":
            apis = parse_simple_url_list(api_text)
        elif api_format == "name_url":
            apis = parse_name_url_format(api_text)

        if apis:
            collection_path = save_collection_file(customer_id, name, apis)

    existing_customer = customer_manager.customers[customer_id]
    if not collection_path:
        collection_path = existing_customer.postman_collection

    success = customer_manager.update_customer(
        customer_id=customer_id,
        name=name,
        description=description,
        postman_collection=collection_path,
        environments=environments
    )

    if not success:
        return templates.TemplateResponse(
            "admin/customer_form.html",
            {
                "request": request,
                "customer": {
                    'id': customer_id,
                    'name': name,
                    'description': description,
                    'environments': environments,
                    'api_text': api_text
                },
                "is_new": False,
                "username": username,
                "error": "Failed to update customer"
            }
        )

    return RedirectResponse(url="/admin?success=updated", status_code=303)


@admin_router.post("/customer/{customer_id}/delete")
async def delete_customer(
    customer_id: str,
    username: str = Depends(verify_admin)
):
    """Delete a customer."""
    if customer_id not in customer_manager.customers:
        raise HTTPException(status_code=404, detail="Customer not found")

    success = customer_manager.delete_customer(customer_id)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete customer")

    return RedirectResponse(url="/admin?success=deleted", status_code=303)


@admin_router.post("/customer/{customer_id}/preview-apis")
async def preview_apis(
    customer_id: str,
    username: str = Depends(verify_admin),
    api_format: str = Form("simple"),
    api_text: str = Form("")
):
    """Preview parsed APIs before saving."""
    apis = []

    if api_text.strip():
        if api_format == "simple":
            apis = parse_simple_url_list(api_text)
        elif api_format == "name_url":
            apis = parse_name_url_format(api_text)

    return {"apis": apis}
