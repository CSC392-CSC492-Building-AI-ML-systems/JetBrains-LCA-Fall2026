```python
from hpeOneView.oneview_client import OneViewClient
from hpeOneView.exceptions import HPEOneViewException

# Establish connection using configuration file
config_file = 'oneview_config.json'
oneview_client = OneViewClient.from_json_file(config_file)

# Create a scope
scope_data = {
    "name": "TestScope",
    "description": "Scope for testing user management"
}
scope = oneview_client.scopes.create(scope_data)

# Create a user with specific permissions
user_data = {
    "userName": "testuser1",
    "fullName": "Test User One",
    "password": "Password123!",
    "enabled": True,
    "roles": ["Infrastructure administrator"],
    "initialScopeUris": [scope['uri']]
}
user1 = oneview_client.users.create(user_data)

# Create multiple users with different permissions
users_data = [
    {
        "userName": "testuser2",
        "fullName": "Test User Two",
        "password": "Password123!",
        "enabled": True,
        "roles": ["Read only"],
        "initialScopeUris": [scope['uri']]
    },
    {
        "userName": "testuser3",
        "fullName": "Test User Three",
        "password": "Password123!",
        "enabled": True,
        "roles": ["Server administrator"],
        "initialScopeUris": [scope['uri']]
    }
]
users = []
for udata in users_data:
    user = oneview_client.users.create(udata)
    users.append(user)

# Update a user's password
user1_update = {
    "password": "NewPassword123!"
}
oneview_client.users.update(user1['uri'], user1_update)

# Add a new role to an existing user
user1_roles = user1.get('roles', [])
if "Network administrator" not in user1_roles:
    user1_roles.append("Network administrator")
oneview_client.users.update(user1['uri'], {"roles": user1_roles})

# Update the roles of a user (replace all roles)
oneview_client.users.update(user1['uri'], {"roles": ["Read only", "Server administrator"]})

# Remove certain roles from a user
user1_roles = oneview_client.users.get(user1['uri'])['roles']
roles_to_remove = ["Read only"]
updated_roles = [role for role in user1_roles if role not in roles_to_remove]
oneview_client.users.update(user1['uri'], {"roles": updated_roles})

# Retrieve a user by their username
retrieved_user = oneview_client.users.get_by('userName', 'testuser1')
if retrieved_user:
    retrieved_user = retrieved_user[0]

# Retrieve all users
all_users = oneview_client.users.get_all()

# Validate if a full name or username is already in use
def is_username_in_use(username):
    return bool(oneview_client.users.get_by('userName', username))

def is_fullname_in_use(fullname):
    return bool(oneview_client.users.get_by('fullName', fullname))

username_in_use = is_username_in_use('testuser1')
fullname_in_use = is_fullname_in_use('Test User One')

# Get the roles associated with a user
user_roles = oneview_client.users.get(user1['uri'])['roles']

# Get users by their role
def get_users_by_role(role_name):
    return [user for user in oneview_client.users.get_all() if role_name in user.get('roles', [])]

users_with_readonly = get_users_by_role("Read only")

# Delete a single user
oneview_client.users.delete(user1['uri'])

# Delete multiple users
for user in users:
    oneview_client.users.delete(user['uri'])
```