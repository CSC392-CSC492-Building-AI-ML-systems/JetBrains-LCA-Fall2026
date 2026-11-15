```python
from hpeOneView.oneview_client import OneViewClient

# Establish connection using configuration file
config_file = 'oneview_config.json'
oneview_client = OneViewClient.from_json_file(config_file)

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
    "password": "Password123!",
    "roles": ["Infrastructure administrator"],
    "enabled": True
}
user1 = oneview_client.users.create(user_data)

# Create multiple users with different permissions
users_data = [
    {
        "userName": "testuser2",
        "fullName": "Test User Two",
        "emailAddress": "testuser2@example.com",
        "password": "Password123!",
        "roles": ["Read only"],
        "enabled": True
    },
    {
        "userName": "testuser3",
        "fullName": "Test User Three",
        "emailAddress": "testuser3@example.com",
        "password": "Password123!",
        "roles": ["Scope operator"],
        "enabled": True
    }
]
created_users = oneview_client.users.create_multiple_user(users_data)

# Update the user's password
update_password_data = {
    "oldPassword": "Password123!",
    "newPassword": "NewPassword123!"
}
oneview_client.users.update_password("testuser1", update_password_data)

# Add a new role to an existing user
add_role_data = {
    "roleName": "Backup administrator"
}
oneview_client.users.add_role_to_userName("testuser1", add_role_data)

# Update the roles of a user
update_roles_data = {
    "roles": ["Read only", "Scope operator"]
}
oneview_client.users.update_role_to_userName("testuser2", update_roles_data)

# Remove certain roles from a user
remove_role_data = {
    "roleName": "Scope operator"
}
oneview_client.users.remove_role_to_userName("testuser2", remove_role_data)

# Retrieve a user by their username
user = oneview_client.users.get_a_specific_resource("testuser1")

# Retrieve all users
all_users = oneview_client.users.get_all()

# Validate if a full name or username is already in use
is_username_in_use = oneview_client.users.validate_userName("testuser1")
is_fullname_in_use = oneview_client.users.validate_fullName("Test User One")

# Get the roles associated with a user
user_roles = oneview_client.users.get_role_associated_with_userName("testuser1")

# Get users by their role
users_by_role = oneview_client.users.get_user_by_role("Read only")

# Delete a single user
oneview_client.users.delete("testuser3")

# Delete multiple users
usernames_to_delete = ["testuser1", "testuser2"]
oneview_client.users.delete_multiple_user(usernames_to_delete)
```