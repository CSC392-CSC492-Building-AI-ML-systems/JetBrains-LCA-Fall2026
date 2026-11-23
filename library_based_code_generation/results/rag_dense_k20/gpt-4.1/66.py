```python
from hpeOneView.oneview_client import OneViewClient
from hpeOneView.exceptions import HPEOneViewException

# Load configuration and connect to OneView
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
    "password": "Password123!",
    "roles": ["Infrastructure administrator"],
    "enabled": True
}
user1 = client.users.create(user_data)

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
        "roles": ["Server administrator"],
        "enabled": True
    }
]
created_users = client.users.create_multiple_user(users_data)

# Update a user's password
update_password_data = {
    "password": "NewPassword456!"
}
client.users.update(user1['userName'], update_password_data)

# Add a new role to an existing user
user1_roles = user1['roles']
user1_roles.append("Network administrator")
update_roles_data = {
    "roles": user1_roles
}
client.users.update(user1['userName'], update_roles_data)

# Update the roles of a user (replace all roles)
new_roles_data = {
    "roles": ["Read only", "Server administrator"]
}
client.users.update("testuser2", new_roles_data)

# Remove certain roles from a user
user3 = client.users.get_by('userName', 'testuser3')[0]
user3_roles = user3['roles']
if "Server administrator" in user3_roles:
    user3_roles.remove("Server administrator")
update_roles_remove_data = {
    "roles": user3_roles
}
client.users.update("testuser3", update_roles_remove_data)

# Retrieve a user by their username
retrieved_user = client.users.get_by('userName', 'testuser1')[0]

# Retrieve all users
all_users = client.users.get_all()

# Validate if a full name or username is already in use
def is_fullname_in_use(full_name):
    users = client.users.get_all()
    return any(u['fullName'] == full_name for u in users)

def is_username_in_use(username):
    users = client.users.get_all()
    return any(u['userName'] == username for u in users)

fullname_in_use = is_fullname_in_use("Test User One")
username_in_use = is_username_in_use("testuser2")

# Get the roles associated with a user
roles_of_user1 = client.users.get_role_associated_with_userName("testuser1")

# Get users by their role
def get_users_by_role(role_name):
    users = client.users.get_all()
    return [u for u in users if role_name in u.get('roles', [])]

users_with_readonly = get_users_by_role("Read only")

# Delete a single user
client.users.delete("testuser1")

# Delete multiple users
usernames_to_delete = ["testuser2", "testuser3"]
for uname in usernames_to_delete:
    client.users.delete(uname)
```