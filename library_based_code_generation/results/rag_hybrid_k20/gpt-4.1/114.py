```python
import os
import logging
import json
import datetime
from nucypher.characters.lawful import Alice, Bob
from nucypher.blockchain.eth.agents import EthereumAgent
from nucypher.blockchain.eth.signers import Signer
from nucypher.blockchain.eth.payment import SubscriptionManagerPayment
from nucypher.blockchain.eth.client import EthereumClient
from nucypher.policy.label import Label
from nucypher.policy.policy import generate_policy_parameters
from nucypher.utilities.logging import GlobalLoggerSettings
from nucypher.crypto.kits import UmbralMessageKit

# 1. Set up logging and environment variables
GlobalLoggerSettings.start_console_logging()
logging.basicConfig(level=logging.INFO)
ETH_RPC_ENDPOINT = os.environ.get('ETH_RPC_ENDPOINT')
WALLET_FILEPATH = os.environ.get('WALLET_FILEPATH')
ALICE_ETH_ADDRESS = os.environ.get('ALICE_ETH_ADDRESS')
ALICE_PASSWORD = os.environ.get('ALICE_PASSWORD')
DOCTOR_PUBLIC_KEY = os.environ.get('DOCTOR_PUBLIC_KEY')
DOCTOR_SIGNING_KEY = os.environ.get('DOCTOR_SIGNING_KEY')

# 2. Connect to the Ethereum provider and layer 2 provider
eth_client = EthereumClient(provider_uri=ETH_RPC_ENDPOINT)
layer2_provider_uri = ETH_RPC_ENDPOINT  # For simplicity, using the same endpoint

# 3. Unlock Alice's Ethereum wallet using a password
signer = Signer.from_keystore_file(WALLET_FILEPATH, password=ALICE_PASSWORD)

# 4. Set up Alice's payment method using the SubscriptionManagerPayment class
payment_method = SubscriptionManagerPayment(
    eth_provider_uri=ETH_RPC_ENDPOINT,
    signer=signer,
    domain="ropsten"
)

# 5. Create an instance of Alice
alice = Alice(
    ethereum_address=ALICE_ETH_ADDRESS,
    signer=signer,
    domain="ropsten",
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

# 9. Get the public keys of the recipient (Doctor) and create an instance of Bob
from nucypher.crypto.signing import SigningKey
from nucypher.crypto.kem import DecryptingKey

doctor_encrypting_key = DecryptingKey.from_bytes(bytes.fromhex(DOCTOR_PUBLIC_KEY))
doctor_signing_key = SigningKey.from_bytes(bytes.fromhex(DOCTOR_SIGNING_KEY))
bob = Bob(
    decrypting_key=doctor_encrypting_key,
    verifying_key=doctor_signing_key
)

# 10. Set up policy details
policy_expiration = datetime.datetime.utcnow() + datetime.timedelta(days=30)
m, n = 2, 3  # m-out-of-n shares

# 11. Grant access to Bob by creating a policy and sending it to the NuCypher network
policy_params = generate_policy_parameters(
    label=policy_label,
    m=m,
    n=n,
    expiration=policy_expiration
)
policy = alice.create_policy(
    bob=bob,
    label=policy_label,
    m=m,
    n=n,
    expiration=policy_expiration
)

# 12. Store additional information about the policy in a JSON file
policy_info = {
    "policy_label": policy_label.to_bytes().hex(),
    "policy_pubkey": policy_pubkey.to_bytes().hex(),
    "expiration": policy_expiration.isoformat(),
    "m": m,
    "n": n,
    "alice_address": ALICE_ETH_ADDRESS,
    "doctor_encrypting_key": DOCTOR_PUBLIC_KEY,
    "doctor_signing_key": DOCTOR_SIGNING_KEY,
    "policy_id": policy.id.hex()
}
with open("policy_info.json", "w") as f:
    json.dump(policy_info, f, indent=4)
```