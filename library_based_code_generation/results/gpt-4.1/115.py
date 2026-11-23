```python
from bitcoinlib.encoding import *

# Base conversion examples: (value, from_base, to_base)
base_examples = [
    ('1010', 2, 10),
    ('ff', 16, 10),
    ('123', 10, 16),
    ('777', 8, 2),
    ('deadbeef', 16, 2),
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
    pubkey_hash = addr_to_pubkeyhash(addr)
    print(f"Address: {addr} => Public Key Hash: {pubkey_hash.hex()}")

# Public key hashes to Bitcoin addresses
pubkey_hashes = [
    bytes.fromhex('62e907b15cbf27d5425399ebf6f0fb50ebb88f18'),
    bytes.fromhex('05a955c0e0b8c6f7c6e7e0e92ce409dab907f0cc'),
]

print("\nPublic Key Hash to Bitcoin Address:")
for pk_hash in pubkey_hashes:
    addr = pubkeyhash_to_addr(pk_hash)
    print(f"Public Key Hash: {pk_hash.hex()} => Address: {addr}")

# Create public key hash from redeem script
redeem_script = '52210279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f817982103828d7e19b05b6c6e2a6a7e7e0e92ce409dab907f0cc52ae'
redeem_script_bytes = bytes.fromhex(redeem_script)
redeem_script_hash160 = hash160(to_bytes(redeem_script_bytes))
print("\nRedeem Script to Public Key Hash (hash160):")
print(f"Redeem Script: {redeem_script}\nHash160: {redeem_script_hash160.hex()}")

# DER signature conversion
der_sig = bytes.fromhex('3045022100e0b8c6f7c6e7e0e92ce409dab907f0cc05a955c0e0b8c6f7c6e7e0e92ce409da02207b15cbf27d5425399ebf6f0fb50ebb88f1862e907b15cbf27d5425399ebf6f0f')
converted_sig = convert_der_sig(der_sig)
print("\nDER Signature Conversion:")
print(f"Original DER Signature: {der_sig.hex()}\nConverted Signature: {converted_sig.hex()}")

# Integer to varbyte integer and back
integer_value = 123456
varbyte = int_to_varbyteint(integer_value)
int_back = varbyteint_to_int(varbyte)
print("\nInteger to Varbyte Integer and Back:")
print(f"Integer: {integer_value} => Varbyte: {varbyte.hex()} => Back to Int: {int_back}")

# Normalize data
data_list = ['Bitcoin', b'Litecoin', '  ethereum  ', b'  dogecoin  ']
normalized_strings = [normalize_string(d) for d in data_list]
normalized_vars = [normalize_var(d) for d in data_list]
print("\nNormalized Strings:")
for orig, norm in zip(data_list, normalized_strings):
    print(f"Original: {orig} => Normalized: {norm}")
print("\nNormalized Vars:")
for orig, norm in zip(data_list, normalized_vars):
    print(f"Original: {orig} => Normalized: {norm}")

# Bech32 address to public key hash and back
bech32_addr = 'bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kygt080'
bech32_pubkey_hash = addr_bech32_to_pubkeyhash(bech32_addr)
bech32_addr_back = pubkeyhash_to_addr_bech32(bech32_pubkey_hash, 'bc')
print("\nBech32 Address to Public Key Hash and Back:")
print(f"Bech32 Address: {bech32_addr} => Public Key Hash: {bech32_pubkey_hash.hex()}")
print(f"Public Key Hash: {bech32_pubkey_hash.hex()} => Bech32 Address: {bech32_addr_back}")
```