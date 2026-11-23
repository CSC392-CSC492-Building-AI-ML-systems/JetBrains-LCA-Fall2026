```python
from hpeOneView.oneview_client import OneViewClient

# Establish connection using configuration file
config = 'oneview_config.json'
oneview_client = OneViewClient.from_json_file(config)
users_client = oneview_client.users

# Create a scope
scope_data = {
    "name": "TestScope",
    "description": "Scope for testing"
}
scope = oneview_client.scopes.create(scope_data)

# Create a user with specific permissions
user_data = {
    "userName": "testuser1",
    "fullName": "Test User One",
    "emailAddress": "testuser1@example.com",
    "enabled": True,
    "password": "Password123!",
    "roles": ["Infrastructure administrator"],
    "scopesUri": [scope['uri']]
}
user1 = users_client.create(user_data)

# Create multiple users with different permissions
multiple_users_data = [
    {
        "userName": "testuser2",
        "fullName": "Test User Two",
        "emailAddress": "testuser2@example.com",
        "enabled": True,
        "password": "Password123!",
        "roles": ["Read only"],
        "scopesUri": [scope['uri']]
    },
    {
        "userName": "testuser3",
        "fullName": "Test User Three",
        "emailAddress": "testuser3@example.com",
        "enabled": True,
        "password": "Password123!",
        "roles": ["Server administrator"],
        "scopesUri": [scope['uri']]
    }
]
created_users = users_client.create_multiple_user(multiple_users_data)

# Update a user's password
update_password_data = {
    "password": "NewPassword456!"
}
users_client.update(user1['userName'], update_password_data)

# Add a new role to an existing user
add_role_data = {
    "roleName": "Network administrator"
}
users_client.add_role_to_userName(user1['userName'], add_role_data)

# Update the roles of a user
update_roles_data = {
    "roles": ["Read only", "Server administrator"]
}
users_client.update_role_to_userName("testuser2", update_roles_data)

# Remove certain roles from a user
remove_roles_data = {
    "roles": ["Read only"]
}
users_client.remove_role_to_userName("testuser2", remove_roles_data)

# Retrieve a user by their username
retrieved_user = users_client.get_by_userName("testuser1")

# Retrieve all users
all_users = users_client.get_all()

# Validate if a full name or username is already in use
is_username_valid = users_client.validate_user_name("testuser1")
is_fullname_valid = users_client.validate_user_name("Test User One")

# Get the roles associated with a user
user_roles = users_client.get_role_associated_with_userName("testuser1")

# Get users by their role
users_with_role = users_client.get_user_by_role("Server administrator")

# Delete a single user
users_client.delete("testuser3")

# Delete multiple users
users_client.delete_multiple_user(["testuser1", "testuser2"])
```