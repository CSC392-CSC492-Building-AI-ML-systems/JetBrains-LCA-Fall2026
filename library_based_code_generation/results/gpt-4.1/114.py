```python
import os
import logging
import json
import datetime
from nucypher.characters.lawful import Alice, Bob
from nucypher.blockchain.eth.interfaces import BlockchainPolicyManager
from nucypher.blockchain.eth.signers.local_signer import LocalAccountSigner
from nucypher.blockchain.eth.payment import SubscriptionManagerPayment
from nucypher.blockchain.eth.agents import EthereumAgent
from nucypher.crypto.kits import UmbralMessageKit
from nucypher.crypto.powers import DecryptingPower, SigningPower
from nucypher.policy.label import Label
from nucypher.utilities.logging import set_log_level
from eth_account import Account

# 1. Set up logging and environment variables
set_log_level(logging.INFO)
ETH_RPC_ENDPOINT = os.environ.get("ETH_RPC_ENDPOINT")
WALLET_FILEPATH = os.environ.get("WALLET_FILEPATH")
ALICE_ETH_ADDRESS = os.environ.get("ALICE_ETH_ADDRESS")
ALICE_WALLET_PASSWORD = os.environ.get("ALICE_WALLET_PASSWORD")
L2_PROVIDER_URI = os.environ.get("L2_PROVIDER_URI")
DOCTOR_PUBLIC_KEY = os.environ.get("DOCTOR_PUBLIC_KEY")  # hex-encoded
DOCTOR_SIGNING_KEY = os.environ.get("DOCTOR_SIGNING_KEY")  # hex-encoded

# 2. Connect to the Ethereum provider and layer 2 provider
eth_agent = EthereumAgent(provider_uri=ETH_RPC_ENDPOINT)
l2_agent = EthereumAgent(provider_uri=L2_PROVIDER_URI)

# 3. Unlock Alice's Ethereum wallet using a password
with open(WALLET_FILEPATH, 'r') as wallet_file:
    wallet_json = wallet_file.read()
alice_account = Account.decrypt(wallet_json, ALICE_WALLET_PASSWORD)
alice_signer = LocalAccountSigner(Account.from_key(alice_account))

# 4. Set up Alice's payment method
payment_method = SubscriptionManagerPayment(
    eth_provider_uri=ETH_RPC_ENDPOINT,
    signer=alice_signer,
    account=Account.from_key(alice_account)
)

# 5. Create an instance of Alice
alice = Alice(
    ethereum_address=ALICE_ETH_ADDRESS,
    signer=alice_signer,
    domain="nucypher.network",
    eth_provider_uri=ETH_RPC_ENDPOINT,
    payment_method=payment_method
)

# 6. Start Alice's learning loop
alice.start_learning_loop(now=True)

# 7. Create a policy label and get the policy public key
policy_label = Label(b"heart-rate-data")
policy_pubkey = alice.get_policy_pubkey_from_label(policy_label)

# 8. Generate heart rate samples and save as a file
import random
heart_rate_samples = [random.randint(60, 100) for _ in range(100)]
heart_rate_file = "heart_rate_samples.txt"
with open(heart_rate_file, "w") as f:
    for sample in heart_rate_samples:
        f.write(f"{sample}\n")

# 9. Get the public keys of the recipient (Doctor) and create Bob
doctor_encrypting_power = DecryptingPower.from_public_bytes(bytes.fromhex(DOCTOR_PUBLIC_KEY))
doctor_signing_power = SigningPower.from_public_bytes(bytes.fromhex(DOCTOR_SIGNING_KEY))
bob = Bob(
    decrypting_power=doctor_encrypting_power,
    signing_power=doctor_signing_power
)

# 10. Set up policy details
policy_expiration = datetime.datetime.utcnow() + datetime.timedelta(days=30)
m = 2
n = 3

# 11. Grant access to Bob by creating a policy
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
    "label": policy_label.to_bytes().hex(),
    "policy_pubkey": policy_pubkey.to_bytes().hex(),
    "expiration": policy_expiration.isoformat(),
    "m": m,
    "n": n,
    "bob_encrypting_key": DOCTOR_PUBLIC_KEY,
    "bob_signing_key": DOCTOR_SIGNING_KEY
}
with open("policy_info.json", "w") as f:
    json.dump(policy_info, f, indent=4)
```