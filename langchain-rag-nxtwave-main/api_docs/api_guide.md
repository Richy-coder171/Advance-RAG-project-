# API Documentation

## Authentication

Authentication requires an API key. To obtain your API key:
1. Log into the dashboard
2. Navigate to Settings > API Keys
3. Click 'Generate New Key'
4. Copy the key immediately (it won't be shown again)

Important: API keys expire after 90 days for security.

Add the API key to your request header:
```
Authorization: Bearer YOUR_API_KEY
```

## Rate Limiting

API endpoints enforce rate limits to ensure fair usage:
- Free tier: 100 requests/minute
- Pro tier: 1000 requests/minute
- Enterprise: Custom limits

When hitting rate limits, implement exponential backoff:
- Start with 1 second delay
- Double on each retry (1s, 2s, 4s, 8s...)
- Max backoff: 32 seconds

The API returns a 429 status code when rate limited.

## Pagination

Results are paginated to prevent overwhelming responses.

Use cursor-based pagination:
```
GET /api/v1/users?limit=50&cursor=abc123
```

The response includes:
- data: Array of results
- cursor: Next page cursor (null if no more pages)
- has_more: Boolean indicating more results exist

## Error Handling

All errors return consistent format:
```json
{
  "error": "error_code",
  "message": "Human readable message",
  "details": {...}
}
```

Common errors:
- 401 Unauthorized: Invalid API key
- 429 Too Many Requests: Rate limit exceeded
- 500 Server Error: Try again with exponential backoff
- 400 Bad Request: Invalid parameters

## Webhooks

Webhooks allow real-time notifications for events.

Retry logic for webhooks:
- Linear backoff: Wait 30 seconds between retries
- Max retries: 5 times
- Exponential backoff available for custom plans

## Versioning

API versions follow semantic versioning (v1, v2, etc).

Important: v1 endpoints are deprecated as of Jan 2024.
Migrate to v2 immediately. v1 support ends June 2024.

Use header: `Accept: application/vnd.api+json; version=2`

The v2 API includes:
- Better error messages
- Improved pagination
- New endpoints for bulk operations
- Enhanced security headers
