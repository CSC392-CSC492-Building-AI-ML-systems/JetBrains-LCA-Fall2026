```python
from bitcoinlib.encoding import *

# Base conversion examples: (value, from_base, to_base)
base_conversion_examples = [
    ('1010', 2, 10),
    ('ff', 16, 10),
    ('123', 10, 16),
    ('777', 8, 2),
    ('deadbeef', 16, 2),
    ('100', 10, 8),
]

print("Base Conversion Examples:")
for value, from_base, to_base in base_conversion_examples:
    result = change_base(value, from_base, to_base)
    print(f"Value: {value} (base {from_base}) -> (base {to_base}): {result}")

# Bitcoin addresses to public key hashes
btc_addresses = [
    '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa',
    '3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy',
    'bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kygt080'
]

print("\nBitcoin Address to Public Key Hash:")
for addr in btc_addresses:
    try:
        pubkey_hash = addr_to_pubkeyhash(addr)
        print(f"Address: {addr} -> PubKeyHash: {pubkey_hash.hex()}")
    except Exception as e:
        print(f"Address: {addr} -> Error: {e}")

# Public key hashes to Bitcoin addresses
pubkey_hashes = [
    bytes.fromhex('62e907b15cbf27d5425399ebf6f0fb50ebb88f18'),
    bytes.fromhex('e34cce70c86373273efcc54ce7d2a491bb4a0e84'),
]

print("\nPublic Key Hash to Bitcoin Address:")
for pk_hash in pubkey_hashes:
    addr = pubkeyhash_to_addr(pk_hash)
    print(f"PubKeyHash: {pk_hash.hex()} -> Address: {addr}")

# Create public key hash from redeem script
redeem_script = '52210279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f817982103828d7e19b05d0b8b0e2e0e7e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e1e0e