#!/bin/bash
# API Health Check - Production Installation Script
# Run as root: sudo ./deploy/install.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="api-health-check"
APP_USER="apihealth"
APP_DIR="/opt/${APP_NAME}"
ENV_FILE="/etc/${APP_NAME}.env"
PYTHON_VERSION="python3.11"

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    log_error "This script must be run as root (use sudo)"
    exit 1
fi

log_info "Starting API Health Check installation..."

# Detect RHEL version
if [ -f /etc/redhat-release ]; then
    RHEL_VERSION=$(rpm -E %{rhel})
    log_info "Detected RHEL/CentOS version: ${RHEL_VERSION}"
else
    log_error "This script is designed for RHEL/CentOS/Rocky Linux"
    exit 1
fi

# Install EPEL repository
log_info "Installing EPEL repository..."
dnf install -y epel-release

# Install system dependencies
log_info "Installing system dependencies..."
dnf install -y \
    ${PYTHON_VERSION} \
    ${PYTHON_VERSION}-pip \
    ${PYTHON_VERSION}-devel \
    nginx \
    certbot \
    python3-certbot-nginx \
    gcc \
    make

# Create application user
if id "${APP_USER}" &>/dev/null; then
    log_info "User ${APP_USER} already exists"
else
    log_info "Creating application user: ${APP_USER}"
    useradd -r -s /sbin/nologin ${APP_USER}
fi

# Create application directory
log_info "Setting up application directory..."
mkdir -p ${APP_DIR}/{logs,collections,config}

# Copy application files if not already there
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$(dirname "${SCRIPT_DIR}")"

if [ "${SOURCE_DIR}" != "${APP_DIR}" ]; then
    log_info "Copying application files to ${APP_DIR}..."
    cp -r "${SOURCE_DIR}"/* ${APP_DIR}/
fi

# Create virtual environment
log_info "Creating Python virtual environment..."
${PYTHON_VERSION} -m venv ${APP_DIR}/venv
${APP_DIR}/venv/bin/pip install --upgrade pip wheel

# Install Python dependencies
log_info "Installing Python dependencies..."
${APP_DIR}/venv/bin/pip install -r ${APP_DIR}/requirements.txt

# Install production server
log_info "Installing production dependencies..."
${APP_DIR}/venv/bin/pip install gunicorn

# Set up environment file
if [ ! -f "${ENV_FILE}" ]; then
    log_info "Creating environment file..."
    cp ${APP_DIR}/deploy/api-health-check.env ${ENV_FILE}

    # Generate random admin password
    ADMIN_PASS=$(openssl rand -base64 16 | tr -dc 'a-zA-Z0-9' | head -c 16)
    sed -i "s/CHANGE_ME_TO_SECURE_PASSWORD/${ADMIN_PASS}/" ${ENV_FILE}

    log_warn "Generated admin password: ${ADMIN_PASS}"
    log_warn "Please save this password securely!"
else
    log_info "Environment file already exists, skipping..."
fi

# Set permissions on env file
chmod 600 ${ENV_FILE}
chown ${APP_USER}:${APP_USER} ${ENV_FILE}

# Set application permissions
log_info "Setting file permissions..."
chown -R ${APP_USER}:${APP_USER} ${APP_DIR}
chmod -R 750 ${APP_DIR}
chmod 770 ${APP_DIR}/logs
chmod 770 ${APP_DIR}/collections
chmod 770 ${APP_DIR}/config

# Install systemd service
log_info "Installing systemd service..."
cp ${APP_DIR}/deploy/api-health-check.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable ${APP_NAME}

# Install nginx configuration
log_info "Installing nginx configuration..."
cp ${APP_DIR}/deploy/nginx.conf /etc/nginx/conf.d/${APP_NAME}.conf

# SELinux configuration
if command -v getenforce &> /dev/null && [ "$(getenforce)" != "Disabled" ]; then
    log_info "Configuring SELinux..."
    setsebool -P httpd_can_network_connect 1
fi

# Configure firewall
if systemctl is-active --quiet firewalld; then
    log_info "Configuring firewall..."
    firewall-cmd --permanent --add-service=http
    firewall-cmd --permanent --add-service=https
    firewall-cmd --reload
fi

# Create backup cron job
log_info "Setting up backup cron job..."
cat > /etc/cron.d/${APP_NAME}-backup << 'EOF'
# Daily backup at 2 AM
0 2 * * * apihealth /opt/api-health-check/deploy/backup.sh >> /opt/api-health-check/logs/backup.log 2>&1
EOF

# Start services
log_info "Starting services..."
systemctl start ${APP_NAME}
sleep 2

# Test if application is running
if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8081/dashboard | grep -q "200\|302"; then
    log_info "Application is running successfully!"
else
    log_error "Application may not be running correctly. Check: sudo journalctl -u ${APP_NAME}"
fi

# Reload nginx
nginx -t && systemctl reload nginx

log_info "============================================"
log_info "Installation complete!"
log_info "============================================"
log_info ""
log_info "Next steps:"
log_info "1. Edit nginx config with your domain:"
log_info "   sudo vi /etc/nginx/conf.d/${APP_NAME}.conf"
log_info ""
log_info "2. Obtain SSL certificate:"
log_info "   sudo certbot --nginx -d yourdomain.com"
log_info ""
log_info "3. Access the application:"
log_info "   Dashboard: https://yourdomain.com/dashboard"
log_info "   Admin: https://yourdomain.com/admin"
log_info ""
log_info "Useful commands:"
log_info "   sudo systemctl status ${APP_NAME}"
log_info "   sudo journalctl -u ${APP_NAME} -f"
log_info ""
