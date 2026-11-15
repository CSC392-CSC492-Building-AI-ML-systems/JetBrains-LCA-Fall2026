```python
from emulator_library import (
    makeEmulatorBaseWith10StubASAndHosts,
    createBlockchain,
    load_eth_nodes,
    connect_to_geth_pow,
    connect_to_geth_poa,
    setCustomGeth,
    createEmulatorAccountFromMnemonic,
    setVirtualNode,
    getVirtualNodes,
    createEmulation,
    build_emulator,
)

# Step 1: Create emulator base with 10 stub AS and hosts
emulator = makeEmulatorBaseWith10StubASAndHosts()

# Step 2: Create Ethereum service with saveState and override set to True
eth_service = {
    "saveState": True,
    "override": True
}
emulator.addService("ethereum", eth_service)

# Step 3: Create two blockchains: POW and POA
pow_blockchain = createBlockchain(type="POW", name="POW-Chain")
poa_blockchain = createBlockchain(type="POA", name="POA-Chain")

# Step 4: For each blockchain, create four nodes
pow_nodes = load_eth_nodes(pow_blockchain, count=4)
poa_nodes = load_eth_nodes(poa_blockchain, count=4)

# Step 5: Set first two nodes as bootnodes and start mining
for node in pow_nodes[:2]:
    connect_to_geth_pow(node, bootnode=True, mining=True)
for node in poa_nodes[:2]:
    connect_to_geth_poa(node, bootnode=True, mining=True)

# Step 6: For the third node, create accounts with a certain balance
createEmulatorAccountFromMnemonic(pow_nodes[2], mnemonic="test test test test test test test test test test test junk", balance=1000000000000000000)
createEmulatorAccountFromMnemonic(poa_nodes[2], mnemonic="test test test test test test test test test test test junk", balance=1000000000000000000)

# Step 7: For the fourth node, set custom geth command options
pow_nodes[3].setCustomGeth(options=["--maxpeers", "50", "--syncmode", "fast"])
poa_nodes[3].setCustomGeth(options=["--maxpeers", "25", "--syncmode", "full"])

# Step 8: Enable HTTP and WebSocket connections on certain nodes
pow_nodes[0].enable_http = True
pow_nodes[0].enable_ws = True
poa_nodes[1].enable_http = True
poa_nodes[1].enable_ws = True

# Step 9: Set custom geth binary file on one of the nodes
pow_nodes[1].setCustomGeth(binary="/usr/local/bin/custom_geth")

# Step 10: Customize display names for visualization
pow_nodes[0].display_name = "POW-Bootnode-1"
pow_nodes[1].display_name = "POW-Bootnode-2"
pow_nodes[2].display_name = "POW-Account-Node"
pow_nodes[3].display_name = "POW-Custom-Node"
poa_nodes[0].display_name = "POA-Bootnode-1"
poa_nodes[1].display_name = "POA-Bootnode-2"
poa_nodes[2].display_name = "POA-Account-Node"
poa_nodes[3].display_name = "POA-Custom-Node"

# Step 11: Bind virtual nodes to physical nodes using filters
virtual_nodes = getVirtualNodes(emulator)
for i, node in enumerate(pow_nodes + poa_nodes):
    setVirtualNode(node, virtual_nodes[i % len(virtual_nodes)], filter={"as_id": i % 10})

# Step 12: Add the Ethereum layer to the emulator
emulator.addLayer("ethereum", pow_nodes + poa_nodes)

# Step 13: Save the component to a file
emulator.saveComponent("emulator_component.json")

# Step 14: Compile the emulator with Docker and save output to a directory
build_emulator(emulator, output_dir="./emulator_output", use_docker=True)
```