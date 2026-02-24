# Production Deployment Guide - Red Hat / CentOS / Rocky Linux

## Prerequisites

- RHEL 8/9, CentOS Stream, Rocky Linux, or AlmaLinux
- Python 3.11+ installed
- Root or sudo access
- Domain name pointing to your server (for Let's Encrypt)

## Quick Start

```bash
# 1. Clone/upload the application
cd /opt
sudo git clone <your-repo-url> api-health-check
cd api-health-check

# 2. Run the deployment script
sudo ./deploy/install.sh
```

## Manual Installation Steps

### 1. Install System Dependencies

```bash
# Enable EPEL repository
sudo dnf install -y epel-release

# Install required packages
sudo dnf install -y python3.11 python3.11-pip python3.11-devel \
    nginx certbot python3-certbot-nginx \
    gcc make

# Create application user
sudo useradd -r -s /sbin/nologin apihealth
```

### 2. Set Up Application

```bash
# Create application directory
sudo mkdir -p /opt/api-health-check
cd /opt/api-health-check

# Copy application files (or git clone)
# sudo cp -r /path/to/source/* .

# Create virtual environment
sudo python3.11 -m venv venv
sudo ./venv/bin/pip install --upgrade pip
sudo ./venv/bin/pip install -r requirements.txt

# Create necessary directories
sudo mkdir -p logs config collections

# Set permissions
sudo chown -R apihealth:apihealth /opt/api-health-check
sudo chmod 750 /opt/api-health-check
```

### 3. Configure Environment

```bash
# Copy and edit environment file
sudo cp deploy/api-health-check.env /etc/api-health-check.env
sudo chmod 600 /etc/api-health-check.env
sudo chown apihealth:apihealth /etc/api-health-check.env

# Edit with your settings
sudo vi /etc/api-health-check.env
```

### 4. Install systemd Service

```bash
sudo cp deploy/api-health-check.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable api-health-check
sudo systemctl start api-health-check

# Check status
sudo systemctl status api-health-check
```

### 5. Configure Nginx

```bash
# Copy nginx config
sudo cp deploy/nginx.conf /etc/nginx/conf.d/api-health-check.conf

# Edit domain name
sudo vi /etc/nginx/conf.d/api-health-check.conf

# Test and reload
sudo nginx -t
sudo systemctl reload nginx
```

### 6. Set Up SSL with Let's Encrypt

```bash
# Obtain certificate (replace with your domain)
sudo certbot --nginx -d healthcheck.yourdomain.com

# Auto-renewal is configured automatically
sudo systemctl enable certbot-renew.timer
```

### 7. Configure Firewall

```bash
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
```

## Configuration Files

| File | Location | Purpose |
|------|----------|---------|
| `api-health-check.env` | `/etc/api-health-check.env` | Environment variables |
| `api-health-check.service` | `/etc/systemd/system/` | systemd service |
| `nginx.conf` | `/etc/nginx/conf.d/api-health-check.conf` | Nginx reverse proxy |

## Management Commands

```bash
# Service management
sudo systemctl start api-health-check
sudo systemctl stop api-health-check
sudo systemctl restart api-health-check
sudo systemctl status api-health-check

# View logs
sudo journalctl -u api-health-check -f
sudo tail -f /opt/api-health-check/logs/app.log

# Reload after config changes
sudo systemctl reload api-health-check
```

## Backup

```bash
# Backup script location
/opt/api-health-check/deploy/backup.sh

# Schedule daily backups
sudo cp deploy/backup-cron /etc/cron.d/api-health-check-backup
```

## Troubleshooting

### Check if service is running
```bash
sudo systemctl status api-health-check
curl -I http://localhost:8081/health
```

### View application logs
```bash
sudo journalctl -u api-health-check -n 100 --no-pager
```

### SELinux issues
```bash
# Allow nginx to connect to upstream
sudo setsebool -P httpd_can_network_connect 1

# Check for SELinux denials
sudo ausearch -m avc -ts recent
```

### Permission issues
```bash
sudo chown -R apihealth:apihealth /opt/api-health-check
sudo chmod -R 750 /opt/api-health-check
```
