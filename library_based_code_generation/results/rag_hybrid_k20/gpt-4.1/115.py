```python
from bitcoinlib.encoding import *

# Base conversion examples: (value, from_base, to_base)
base_examples = [
    ('1010', 2, 10),
    ('ff', 16, 10),
    ('123', 10, 16),
    ('abcdef', 16, 2),
    ('777', 8, 16),
]

print("Base Conversion Examples:")
for value, from_base, to_base in base_examples:
    result = change_base(value, from_base, to_base)
    print(f"Value: {value}, From base: {from_base}, To base: {to_base} => Result: {result}")

# Bitcoin addresses to public key hashes
btc_addresses = [
    '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa',
    '3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy',
    'bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kygt080'
]

print("\nBitcoin Address to Public Key Hash:")
for addr in btc_addresses:
    try:
        pkh = addr_to_pubkeyhash(addr)
        print(f"Address: {addr} => Public Key Hash: {pkh.hex()}")
    except Exception as e:
        print(f"Address: {addr} => Error: {e}")

# Public key hashes to Bitcoin addresses
pubkey_hashes = [
    bytes.fromhex('62e907b15cbf27d5425399ebf6f0fb50ebb88f18'),
    bytes.fromhex('e34cce70c86373273efcc54ce7d2a491bb4a0e84')
]

print("\nPublic Key Hash to Bitcoin Address:")
for pkh in pubkey_hashes:
    addr = pubkeyhash_to_addr(pkh)
    print(f"Public Key Hash: {pkh.hex()} => Address: {addr}")

# Create public key hash from redeem script
redeem_script = '52210279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f817982103828d7e19b05b6b1b7b2a3b4e2e0e7e3b6c2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6e2e1b6