# Portal Setup Guide

## Overview
This sets up a master portal with password-protected access to test automation tools.

---

## Step 1: Copy Portal Files to Server

**From your local machine:**
```bash
scp -r /Users/shubhamnagar/ai-engineering/test-automatiom/api-check/deploy/portal shubham.nagar@aidemo.infoaxon.com:/tmp/
scp /Users/shubhamnagar/ai-engineering/test-automatiom/api-check/deploy/nginx-portal.conf shubham.nagar@aidemo.infoaxon.com:/tmp/
```

---

## Step 2: Set Up Portal on Server

```bash
# Create portal directory
sudo mkdir -p /var/www/portal

# Copy portal files
sudo cp /tmp/portal/* /var/www/portal/

# Set permissions
sudo chown -R nginx:nginx /var/www/portal
sudo chmod -R 755 /var/www/portal
```

---

## Step 3: Create Password File

**Install htpasswd utility:**
```bash
sudo dnf install -y httpd-tools
```

**Create first user (e.g., admin):**
```bash
sudo htpasswd -c /etc/nginx/.htpasswd admin
# Enter password when prompted
```

**Add more users (customers/teams):**
```bash
# For each customer, add a user
sudo htpasswd /etc/nginx/.htpasswd rgi_team
sudo htpasswd /etc/nginx/.htpasswd indusind_team
sudo htpasswd /etc/nginx/.htpasswd customer1
```

**Secure the password file:**
```bash
sudo chown nginx:nginx /etc/nginx/.htpasswd
sudo chmod 640 /etc/nginx/.htpasswd
```

---

## Step 4: Update Nginx Configuration

```bash
# Backup current config
sudo cp /etc/nginx/conf.d/api-health-check.conf /etc/nginx/conf.d/api-health-check.conf.bak

# Replace with portal config
sudo cp /tmp/nginx-portal.conf /etc/nginx/conf.d/portal.conf
sudo rm /etc/nginx/conf.d/api-health-check.conf

# Test and reload
sudo nginx -t
sudo systemctl reload nginx
```

---

## Step 5: Test

```bash
# Test without auth (should get 401)
curl -I http://localhost/

# Test with auth
curl -u admin:yourpassword http://localhost/

# Test API Health Check
curl -u admin:yourpassword http://localhost/api-health/dashboard
```

---

## Access URLs

| URL | Description |
|-----|-------------|
| `http://aidemo.infoaxon.com/` | Main Portal (login required) |
| `http://aidemo.infoaxon.com/api-health/dashboard` | API Health Dashboard |
| `http://aidemo.infoaxon.com/api-health/admin` | API Health Admin |
| `http://aidemo.infoaxon.com/health` | Health check (no auth) |

---

## Managing Users

**List all users:**
```bash
cat /etc/nginx/.htpasswd
```

**Add new user:**
```bash
sudo htpasswd /etc/nginx/.htpasswd newuser
```

**Delete user:**
```bash
sudo htpasswd -D /etc/nginx/.htpasswd username
```

**Change password:**
```bash
sudo htpasswd /etc/nginx/.htpasswd existinguser
```

---

## Per-Customer Access Control (Advanced)

For different customers to see only their data, you have two options:

### Option A: Separate Password Files per Path

Create customer-specific configs:

```nginx
# Customer 1 - only sees their dashboard
location /api-health/customer1/ {
    auth_basic "Customer 1 Portal";
    auth_basic_user_file /etc/nginx/.htpasswd-customer1;
    # ... proxy config
}
```

### Option B: Application-Level Access Control

Modify the API Health Check application to:
1. Read the authenticated username from nginx (`X-Remote-User` header)
2. Filter dashboard data based on the username
3. Map usernames to customer_ids

Add this to nginx:
```nginx
proxy_set_header X-Remote-User $remote_user;
```

Then in the application, use this header to filter which customers the user can see.

---

## Troubleshooting

**401 Unauthorized:**
- Check password file exists: `ls -la /etc/nginx/.htpasswd`
- Verify user exists: `grep username /etc/nginx/.htpasswd`

**403 Forbidden:**
- Check portal files: `ls -la /var/www/portal/`
- Check SELinux: `sudo setsebool -P httpd_read_user_content 1`

**502 Bad Gateway:**
- Check if app is running: `curl http://localhost:8081/health`
- Check service: `sudo systemctl status api-health-check`
