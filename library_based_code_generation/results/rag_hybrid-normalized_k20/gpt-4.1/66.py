```python
from hpeOneView.oneview_client import OneViewClient
from hpeOneView.exceptions import HPEOneViewException

# Establish connection using configuration file
config_file = 'oneview_config.json'
client = OneViewClient.from_json_file(config_file)

# Create a scope
scope_data = {
    "name": "TestScope",
    "description": "Scope for testing user management"
}
scope = client.scopes.create(scope_data)

# Create a user with specific permissions
user_data = {
    "userName": "testuser1",
    "fullName": "Test User One",
    "emailAddress": "testuser1@example.com",
    "enabled": True,
    "password": "Password123!",
    "roles": ["Infrastructure administrator"],
    "scopeUris": [scope['uri']]
}
user1 = client.users.create(user_data)

# Create multiple users with different permissions
multiple_users_data = [
    {
        "userName": "testuser2",
        "fullName": "Test User Two",
        "emailAddress": "testuser2@example.com",
        "enabled": True,
        "password": "Password123!",
        "roles": ["Read only"],
        "scopeUris": [scope['uri']]
    },
    {
        "userName": "testuser3",
        "fullName": "Test User Three",
        "emailAddress": "testuser3@example.com",
        "enabled": True,
        "password": "Password123!",
        "roles": ["Server administrator"],
        "scopeUris": [scope['uri']]
    }
]
created_users = client.users.create_multiple_user(multiple_users_data)

# Update a user's password
update_password_data = {
    "password": "NewPassword456!"
}
client.users.update(user1['uri'], update_password_data)

# Add a new role to an existing user
client.users.add_role_to_userName("testuser1", ["Network administrator"])

# Update the roles of a user (replace all roles)
client.users.update_role_to_userName("testuser2", ["Storage administrator"])

# Remove certain roles from a user (set roles to a subset)
client.users.update_role_to_userName("testuser3", ["Read only"])

# Retrieve a user by their username
retrieved_user = client.users.get_by_userName("testuser1")

# Retrieve all users
all_users = client.users.get_all()

# Validate if a full name or username is already in use
is_username_in_use = client.users.validate_user_name("testuser1")
is_fullname_in_use = client.users.validate_user_name("Test User One")

# Get the roles associated with a user
user_roles = client.users.get_role_associated_with_userName("testuser1")

# Get users by their role
users_with_role = client.users.get_user_by_role("Read only")

# Delete a single user
client.users.delete(user1['uri'])

# Delete multiple users
user_uris_to_delete = [user['uri'] for user in created_users]
for uri in user_uris_to_delete:
    client.users.delete(uri)
```