# Authentication Guide

## API Key Authentication

API keys are the primary authentication method. Each key is scoped to specific permissions.

### Getting Your API Key

1. Sign into the dashboard
2. Go to Account Settings
3. Select "API Keys" from the left menu
4. Click "Generate New Key"
5. Give it a descriptive name
6. Select permissions scope (read, write, delete)
7. Copy the key immediately - you won't see it again

### Using Your API Key

Add to request header:
```
curl -H "Authorization: Bearer sk_live_abc123xyz789" \
     https://api.example.com/api/v2/users
```

Or in Python:
```python
import requests
headers = {"Authorization": f"Bearer {api_key}"}
response = requests.get("https://api.example.com/api/v2/users", headers=headers)
```

### Key Security

- Keys expire after 90 days
- Rotate keys regularly
- Never commit keys to version control
- Use environment variables to store keys
- If compromised, regenerate immediately

### Webhook Signing

Webhooks are signed with HMAC-SHA256 for verification.

Header: `X-Webhook-Signature`

Verification process:
1. Extract signature from header
2. Compute HMAC-SHA256 of webhook body with your secret key
3. Compare values in constant-time manner

Example verification:
```python
import hmac
import hashlib

def verify_webhook(payload, signature, secret):
    computed = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(computed, signature)
```