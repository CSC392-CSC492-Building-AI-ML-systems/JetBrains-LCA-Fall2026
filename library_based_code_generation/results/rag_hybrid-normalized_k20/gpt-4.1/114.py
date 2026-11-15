```python
import os
import logging
import json
import datetime
from nucypher.characters.lawful import Alice, Bob
from nucypher.blockchain.eth.signers.local_signer import LocalAccountSigner
from nucypher.blockchain.eth.agents import EthereumClient
from nucypher.blockchain.eth.payment import SubscriptionManagerPayment
from nucypher.policy.label import Label
from nucypher.policy.models import Policy
from nucypher.crypto.kits import UmbralMessageKit
from nucypher.utilities.logging import GlobalLoggerSettings
from nucypher.utilities.sandbox.constants import TEMPORARY_DOMAIN
from nucypher.data_sources import DataSource
from nucypher.crypto.powers import DecryptingPower, SigningPower

# 1. Set up logging and environment variables
GlobalLoggerSettings.start_console_logging()
logging.basicConfig(level=logging.INFO)
ETH_RPC_ENDPOINT = os.environ.get("ETH_RPC_ENDPOINT", "http://localhost:8545")
WALLET_FILEPATH = os.environ.get("WALLET_FILEPATH", "./alice_wallet.json")
ALICE_ETH_ADDRESS = os.environ.get("ALICE_ETH_ADDRESS")
ALICE_PASSWORD = os.environ.get("ALICE_PASSWORD")
DOCTOR_ETH_ADDRESS = os.environ.get("DOCTOR_ETH_ADDRESS")
DOCTOR_PUBLIC_KEY = os.environ.get("DOCTOR_PUBLIC_KEY")  # hex-encoded

if not (ALICE_ETH_ADDRESS and ALICE_PASSWORD and DOCTOR_ETH_ADDRESS and DOCTOR_PUBLIC_KEY):
    raise EnvironmentError("Missing required environment variables.")

# 2. Connect to the Ethereum provider and layer 2 provider
eth_client = EthereumClient(provider_uri=ETH_RPC_ENDPOINT)
layer2_domain = TEMPORARY_DOMAIN  # For testnet/demo purposes

# 3. Unlock Alice's Ethereum wallet using a password
signer = LocalAccountSigner.from_wallet(
    wallet_filepath=WALLET_FILEPATH,
    password=ALICE_PASSWORD
)

# 4. Set up Alice's payment method
payment_method = SubscriptionManagerPayment(
    eth_provider_uri=ETH_RPC_ENDPOINT,
    signer=signer,
    domain=layer2_domain
)

# 5. Create an instance of Alice
alice = Alice(
    ethereum_address=ALICE_ETH_ADDRESS,
    signer=signer,
    domain=layer2_domain,
    eth_provider_uri=ETH_RPC_ENDPOINT,
    payment_method=payment_method
)

# 6. Start Alice's learning loop
alice.start_learning_loop(now=True)

# 7. Create a policy label and get the policy public key
policy_label = Label(b"heart-rate-data")
policy_pubkey = alice.get_policy_encrypting_key_from_label(policy_label)

# 8. Generate heart rate samples and save as a file
import random
heart_rate_samples = [random.randint(60, 100) for _ in range(100)]
heart_rate_file = "heart_rate_samples.txt"
with open(heart_rate_file, "w") as f:
    for sample in heart_rate_samples:
        f.write(f"{sample}\n")

# 9. Get the public keys of the recipient (Doctor) and create Bob
from nucypher.crypto.kits import UmbralPublicKey
doctor_encrypting_key = UmbralPublicKey.from_bytes(bytes.fromhex(DOCTOR_PUBLIC_KEY))
bob = Bob(
    decrypting_power=DecryptingPower(doctor_encrypting_key),
    verifying_power=SigningPower(doctor_encrypting_key)
)

# 10. Set up policy details
policy_expiration = datetime.datetime.utcnow() + datetime.timedelta(days=30)
m = 2
n = 3

# 11. Grant access to Bob by creating a policy
policy = alice.create_policy(
    bob=doctor_encrypting_key,
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
    "alice_eth_address": ALICE_ETH_ADDRESS,
    "doctor_eth_address": DOCTOR_ETH_ADDRESS,
    "heart_rate_file": heart_rate_file
}
with open("policy_info.json", "w") as f:
    json.dump(policy_info, f, indent=4)
```