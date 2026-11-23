```python
import os
import logging
import json
import datetime
from nucypher.characters.lawful import Alice, Bob
from nucypher.blockchain.eth.interfaces import EthereumClient
from nucypher.blockchain.eth.signers import Signer
from nucypher.blockchain.eth.payment import SubscriptionManagerPayment
from nucypher.policy.label import Label
from nucypher.policy.policy import Policy
from nucypher.crypto.kits import UmbralMessageKit
from nucypher.crypto.powers import DecryptingPower, SigningPower
from nucypher.crypto.signing import Signer as CryptoSigner
from nucypher.utilities.logging import GlobalLoggerSettings

# 1. Set up logging and environment variables
GlobalLoggerSettings.start_console_logging()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ETH_RPC_ENDPOINT = os.environ.get('ETH_RPC_ENDPOINT')
WALLET_FILEPATH = os.environ.get('WALLET_FILEPATH')
ALICE_ETH_ADDRESS = os.environ.get('ALICE_ETH_ADDRESS')
ALICE_WALLET_PASSWORD = os.environ.get('ALICE_WALLET_PASSWORD')
DOCTOR_PUBLIC_KEY = os.environ.get('DOCTOR_PUBLIC_KEY')  # hex-encoded
DOCTOR_SIGNING_KEY = os.environ.get('DOCTOR_SIGNING_KEY')  # hex-encoded

if not all([ETH_RPC_ENDPOINT, WALLET_FILEPATH, ALICE_ETH_ADDRESS, ALICE_WALLET_PASSWORD, DOCTOR_PUBLIC_KEY, DOCTOR_SIGNING_KEY]):
    raise EnvironmentError("Missing one or more required environment variables.")

# 2. Connect to the Ethereum provider and layer 2 provider
eth_client = EthereumClient(provider_uri=ETH_RPC_ENDPOINT)

# 3. Unlock Alice's Ethereum wallet using a password
signer = Signer.from_keystore_file(WALLET_FILEPATH, password=ALICE_WALLET_PASSWORD)

# 4. Set up Alice's payment method
payment_method = SubscriptionManagerPayment(eth_client=eth_client, signer=signer)

# 5. Create an instance of Alice
alice = Alice(
    ethereum_address=ALICE_ETH_ADDRESS,
    signer=signer,
    domain="localhost",
    ethereum_provider_uri=ETH_RPC_ENDPOINT,
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
from nucypher.crypto.kits import UmbralMessageKit
from nucypher.crypto.powers import DecryptingPower, SigningPower
from nucypher.crypto.signing import Signer as CryptoSigner
from umbral.keys import UmbralPublicKey

doctor_enc_pubkey = UmbralPublicKey.from_bytes(bytes.fromhex(DOCTOR_PUBLIC_KEY))
doctor_sig_pubkey = UmbralPublicKey.from_bytes(bytes.fromhex(DOCTOR_SIGNING_KEY))
bob = Bob(
    decrypting_power=DecryptingPower(doctor_enc_pubkey),
    signing_power=SigningPower(doctor_sig_pubkey)
)

# 10. Set up policy details
policy_expiration = datetime.datetime.utcnow() + datetime.timedelta(days=7)
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
    "policy_pubkey": policy_pubkey.to_bytes().hex(),
    "label": policy_label.to_bytes().hex(),
    "expiration": policy_expiration.isoformat(),
    "m": m,
    "n": n,
    "recipient_enc_pubkey": DOCTOR_PUBLIC_KEY,
    "recipient_sig_pubkey": DOCTOR_SIGNING_KEY,
    "heart_rate_file": heart_rate_file
}
with open("policy_info.json", "w") as f:
    json.dump(policy_info, f, indent=4)
```