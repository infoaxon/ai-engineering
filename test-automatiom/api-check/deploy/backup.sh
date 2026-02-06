#!/bin/bash
# API Health Check - Backup Script
# Backs up database, collections, and configuration

set -e

# Configuration
APP_DIR="/opt/api-health-check"
BACKUP_DIR="/opt/api-health-check/backups"
RETENTION_DAYS=7
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="backup_${DATE}"

# Create backup directory
mkdir -p ${BACKUP_DIR}

echo "[$(date)] Starting backup: ${BACKUP_NAME}"

# Create temporary backup directory
TEMP_DIR=$(mktemp -d)
mkdir -p ${TEMP_DIR}/${BACKUP_NAME}

# Backup database
if [ -f "${APP_DIR}/health_checks.db" ]; then
    echo "[$(date)] Backing up database..."
    cp "${APP_DIR}/health_checks.db" "${TEMP_DIR}/${BACKUP_NAME}/"
fi

# Backup collections
if [ -d "${APP_DIR}/collections" ]; then
    echo "[$(date)] Backing up collections..."
    cp -r "${APP_DIR}/collections" "${TEMP_DIR}/${BACKUP_NAME}/"
fi

# Backup configuration
if [ -d "${APP_DIR}/config" ]; then
    echo "[$(date)] Backing up configuration..."
    cp -r "${APP_DIR}/config" "${TEMP_DIR}/${BACKUP_NAME}/"
fi

# Backup environment file
if [ -f "/etc/api-health-check.env" ]; then
    echo "[$(date)] Backing up environment file..."
    cp "/etc/api-health-check.env" "${TEMP_DIR}/${BACKUP_NAME}/"
fi

# Create compressed archive
echo "[$(date)] Creating compressed archive..."
cd ${TEMP_DIR}
tar -czf "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" ${BACKUP_NAME}

# Cleanup temp directory
rm -rf ${TEMP_DIR}

# Remove old backups
echo "[$(date)] Removing backups older than ${RETENTION_DAYS} days..."
find ${BACKUP_DIR} -name "backup_*.tar.gz" -mtime +${RETENTION_DAYS} -delete

# Show backup info
BACKUP_SIZE=$(du -h "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" | cut -f1)
echo "[$(date)] Backup complete: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz (${BACKUP_SIZE})"

# List current backups
echo "[$(date)] Current backups:"
ls -lh ${BACKUP_DIR}/*.tar.gz 2>/dev/null || echo "No backups found"
