# Endpoints Reference

## Users Endpoints

### GET /api/v2/users
List all users with pagination support.

Parameters:
- limit: Number of results (max: 100)
- cursor: Pagination cursor

Returns: Array of user objects with metadata.

### POST /api/v2/users
Create a new user.

Request body:
```json
{
  "email": "user@example.com",
  "name": "John Doe",
  "role": "admin|user|viewer"
}
```

Note: Email must be unique. Duplicate emails return 409 Conflict.

### GET /api/v2/users/{id}
Get a specific user by ID.

Returns: Single user object.

### PUT /api/v2/users/{id}
Update user information.

Updatable fields: name, role, metadata

### DELETE /api/v2/users/{id}
Delete a user and all associated data.

Note: This is permanent and cannot be undone.

## Organizations Endpoints

### GET /api/v2/organizations
List all organizations.

### POST /api/v2/organizations
Create organization. Only enterprise tier.

## Bulk Operations

### POST /api/v2/users/bulk
Create multiple users in one request.

Request body:
```json
{
  "users": [
    {"email": "user1@example.com", "name": "User 1"},
    {"email": "user2@example.com", "name": "User 2"}
  ]
}
```

Returns: Array of created user objects with creation timestamps.