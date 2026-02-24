# Deployment Guide - API Health Check

## Server: aidemo.infoaxon.com
## Path: /home/shubham.nagar/ai-projects/api-health

---

## Step 1: Copy Files to Server

**From your local machine:**
```bash
# Create a tar archive (excludes unnecessary files)
cd /Users/shubhamnagar/ai-engineering/test-automatiom/api-check
tar --exclude='venv' --exclude='*.db' --exclude='__pycache__' --exclude='.git' \
    -czvf /tmp/api-health.tar.gz .

# Copy to server
scp /tmp/api-health.tar.gz shubham.nagar@aidemo.infoaxon.com:/tmp/
```

---

## Step 2: SSH to Server and Extract

```bash
ssh shubham.nagar@aidemo.infoaxon.com

# Create directory and extract
mkdir -p /home/shubham.nagar/ai-projects/api-health
cd /home/shubham.nagar/ai-projects/api-health
tar -xzvf /tmp/api-health.tar.gz

# Clean up
rm /tmp/api-health.tar.gz
```

---

## Step 3: Install System Dependencies

```bash
# Install Python and nginx
sudo dnf install -y python3.11 python3.11-pip python3.11-devel nginx

# Or if python3.11 not available:
sudo dnf install -y python3 python3-pip python3-devel nginx
```

---

## Step 4: Set Up Python Environment

```bash
cd /home/shubham.nagar/ai-projects/api-health

# Create virtual environment
python3 -m venv venv

# Upgrade pip and install dependencies
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
```

---

## Step 5: Create Required Directories

```bash
mkdir -p logs collections config
```

---

## Step 6: Install systemd Service

```bash
# Copy service file
sudo cp deploy/api-health-check-user.service /etc/systemd/system/api-health-check.service

# Reload systemd
sudo systemctl daemon-reload

# Enable and start service
sudo systemctl enable api-health-check
sudo systemctl start api-health-check

# Check status
sudo systemctl status api-health-check
```

---

## Step 7: Configure Nginx

```bash
# Remove default config if exists
sudo rm -f /etc/nginx/conf.d/default.conf

# Remove any old SSL config that might cause issues
sudo rm -f /etc/nginx/conf.d/api-health-check.conf

# Copy the no-SSL config
sudo cp deploy/nginx-nossl.conf /etc/nginx/conf.d/api-health-check.conf

# Test nginx configuration
sudo nginx -t

# Enable and start nginx
sudo systemctl enable nginx
sudo systemctl start nginx

# Or reload if already running
sudo systemctl reload nginx
```

---

## Step 8: Configure SELinux (if enabled)

```bash
# Check if SELinux is enabled
getenforce

# If "Enforcing", run:
sudo setsebool -P httpd_can_network_connect 1
```

---

## Step 9: Configure Firewall

```bash
# Open HTTP port
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --reload

# Verify
sudo firewall-cmd --list-all
```

---

## Step 10: Test the Application

```bash
# Test backend directly
curl http://localhost:8081/health

# Test through nginx
curl http://localhost/health

# Test from external
curl http://aidemo.infoaxon.com/health
```

---

## Step 11: Access the Application

- **Dashboard:** http://aidemo.infoaxon.com/dashboard
- **Admin Panel:** http://aidemo.infoaxon.com/admin
  - Username: `admin`
  - Password: `admin123` (change in deploy/api-health-check.env)

---

## Optional: Add SSL Later

Once HTTP is working, add SSL:

```bash
# Install certbot
sudo dnf install -y certbot python3-certbot-nginx

# Get certificate (certbot will modify nginx config automatically)
sudo certbot --nginx -d aidemo.infoaxon.com

# Verify
sudo nginx -t
sudo systemctl reload nginx

# Open HTTPS port
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
```

---

## Useful Commands

```bash
# View application logs
sudo journalctl -u api-health-check -f

# Restart application
sudo systemctl restart api-health-check

# Restart nginx
sudo systemctl restart nginx

# Check application status
sudo systemctl status api-health-check

# Check nginx status
sudo systemctl status nginx
```

---

## Troubleshooting

### Application won't start
```bash
# Check logs
sudo journalctl -u api-health-check -n 50 --no-pager

# Try running manually
cd /home/shubham.nagar/ai-projects/api-health
./venv/bin/uvicorn src.main:app --host 127.0.0.1 --port 8081
```

### Nginx 502 Bad Gateway
```bash
# Check if app is running
curl http://localhost:8081/health

# Check SELinux
sudo setsebool -P httpd_can_network_connect 1
```

### Permission denied errors
```bash
# Fix ownership
sudo chown -R shubham.nagar:shubham.nagar /home/shubham.nagar/ai-projects/api-health
```
