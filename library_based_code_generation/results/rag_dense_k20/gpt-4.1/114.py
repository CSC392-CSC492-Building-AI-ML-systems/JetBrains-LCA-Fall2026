```python
import os
import logging
import json
import datetime
from nucypher.characters.lawful import Alice, Bob
from nucypher.blockchain.eth.agents import EthereumAgent
from nucypher.blockchain.eth.signers import Signer
from nucypher.blockchain.eth.wallet import unlock_keystore
from nucypher.blockchain.eth.payment import SubscriptionManagerPayment
from nucypher.policy.label import Label
from nucypher.policy.policy import generate_policy_parameters
from nucypher.network.middleware import NucypherMiddlewareClient

# 1. Set up logging and environment variables
logging.basicConfig(level=logging.INFO)
ETH_RPC_ENDPOINT = os.environ.get('ETH_RPC_ENDPOINT')
WALLET_FILEPATH = os.environ.get('WALLET_FILEPATH')
ALICE_ETH_ADDRESS = os.environ.get('ALICE_ETH_ADDRESS')
L2_PROVIDER_URI = os.environ.get('L2_PROVIDER_URI', ETH_RPC_ENDPOINT)
DOCTOR_PUBLIC_KEY_FILE = os.environ.get('DOCTOR_PUBLIC_KEY_FILE', 'doctor_pubkeys.json')
POLICY_INFO_FILE = os.environ.get('POLICY_INFO_FILE', 'policy_info.json')
HEART_RATE_FILE = os.environ.get('HEART_RATE_FILE', 'heart_rate_samples.txt')

# 2. Connect to the Ethereum provider and layer 2 provider
eth_agent = EthereumAgent(provider_uri=ETH_RPC_ENDPOINT)
middleware_client = NucypherMiddlewareClient(provider_uri=L2_PROVIDER_URI)

# 3. Unlock Alice's Ethereum wallet using a password
with open(WALLET_FILEPATH, 'r') as f:
    keystore = f.read()
alice_password = os.environ.get('ALICE_WALLET_PASSWORD')
alice_signer = unlock_keystore(keystore, alice_password)

# 4. Set up Alice's payment method
payment_method = SubscriptionManagerPayment(
    eth_provider_uri=ETH_RPC_ENDPOINT,
    signer=alice_signer,
    eth_address=ALICE_ETH_ADDRESS
)

# 5. Create an instance of Alice
alice = Alice(
    ethereum_address=ALICE_ETH_ADDRESS,
    signer=alice_signer,
    domain=NuCypherNetworkName("mainnet"),
    eth_provider_uri=ETH_RPC_ENDPOINT,
    payment_method=payment_method,
    middleware_uri=L2_PROVIDER_URI
)

# 6. Start Alice's learning loop
alice.start_learning_loop(now=True)

# 7. Create a policy label and get the policy public key
policy_label = Label(b"heart-rate-data")
policy_pubkey = alice.get_policy_pubkey_from_label(policy_label)

# 8. Generate heart rate samples and save as a file
import random
heart_rates = [random.randint(60, 100) for _ in range(100)]
with open(HEART_RATE_FILE, 'w') as f:
    for hr in heart_rates:
        f.write(f"{hr}\n")

# 9. Get the public keys of the recipient (Doctor) and create Bob
with open(DOCTOR_PUBLIC_KEY_FILE, 'r') as f:
    doctor_pubkeys = json.load(f)
bob = Bob.from_public_keys(
    verifying_key=doctor_pubkeys['verifying_key'],
    encrypting_key=doctor_pubkeys['encrypting_key']
)

# 10. Set up policy details
policy_expiration = datetime.datetime.utcnow() + datetime.timedelta(days=30)
m, n = 2, 3  # m-out-of-n shares

# 11. Grant access to Bob by creating a policy
policy_params = generate_policy_parameters(
    label=policy_label,
    m=m,
    n=n,
    expiration=policy_expiration
)
policy = alice.grant(
    bob=bob,
    label=policy_label,
    m=m,
    n=n,
    expiration=policy_expiration
)

# 12. Store additional information about the policy in a JSON file
policy_info = {
    "policy_id": policy.id.hex(),
    "label": policy_label.to_bytes().decode(),
    "policy_pubkey": policy_pubkey.to_bytes().hex(),
    "expiration": policy_expiration.isoformat(),
    "m": m,
    "n": n,
    "bob_verifying_key": doctor_pubkeys['verifying_key'],
    "bob_encrypting_key": doctor_pubkeys['encrypting_key'],
    "heart_rate_file": HEART_RATE_FILE
}
with open(POLICY_INFO_FILE, 'w') as f:
    json.dump(policy_info, f, indent=2)
```