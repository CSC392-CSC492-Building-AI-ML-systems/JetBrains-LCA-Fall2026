```python
from emulator_library import (
    createEmulation,
    makeEmulatorBaseWith10StubASAndHosts,
    EthereumServer,
    createBlockchain,
    addEmulatorAccounts,
    setCustomGeth,
    setVirtualNode,
    build_emulator,
    gen_emulation_files,
)

# Step 1: Create emulator base with 10 stub AS and hosts
emulator_base = makeEmulatorBaseWith10StubASAndHosts()

# Step 2: Create Ethereum service with saveState and override set to True
eth_service = EthereumServer(saveState=True, override=True)

# Step 3: Create two blockchains: POW and POA
pow_chain = createBlockchain(consensus='pow')
poa_chain = createBlockchain(consensus='poa')

# Step 4: For each blockchain, create four nodes
pow_nodes = [pow_chain.create_node() for _ in range(4)]
poa_nodes = [poa_chain.create_node() for _ in range(4)]

# Step 5: Set first two nodes as bootnodes and start mining
for node in pow_nodes[:2]:
    node.set_bootnode(True)
    node.start_mining()
for node in poa_nodes[:2]:
    node.set_bootnode(True)
    node.start_mining()

# Step 6: For third node, create accounts with a certain balance
pow_accounts = addEmulatorAccounts(pow_nodes[2], balances={"0x123...": 1000})
poa_accounts = addEmulatorAccounts(poa_nodes[2], balances={"0xabc...": 2000})

# Step 7: For fourth node, set custom geth command options
pow_nodes[3].set_geth_options(['--maxpeers', '50', '--syncmode', 'fast'])
poa_nodes[3].set_geth_options(['--maxpeers', '25', '--syncmode', 'light'])

# Step 8: Enable HTTP and WebSocket connections on certain nodes
pow_nodes[0].enable_http(True)
pow_nodes[0].enable_ws(True)
poa_nodes[1].enable_http(True)
poa_nodes[1].enable_ws(True)

# Step 9: Set custom geth binary file on one node
setCustomGeth(pow_nodes[1], '/usr/local/bin/custom_geth')

# Step 10: Customize display names for visualization
pow_nodes[0].set_display_name("POW Bootnode 1")
pow_nodes[1].set_display_name("POW Bootnode 2")
pow_nodes[2].set_display_name("POW Account Node")
pow_nodes[3].set_display_name("POW Custom Node")
poa_nodes[0].set_display_name("POA Bootnode 1")
poa_nodes[1].set_display_name("POA Bootnode 2")
poa_nodes[2].set_display_name("POA Account Node")
poa_nodes[3].set_display_name("POA Custom Node")

# Step 11: Bind virtual nodes to physical nodes using filters
for i, node in enumerate(pow_nodes + poa_nodes):
    setVirtualNode(node, filter=f"AS{i%10+1}")

# Step 12: Add Ethereum layer to the emulator
emulator_base.add_layer(eth_service)
eth_service.add_blockchain(pow_chain)
eth_service.add_blockchain(poa_chain)

# Step 13: Save the component to a file
gen_emulation_files(emulator_base, filename="emulator_component.json")

# Step 14: Compile the emulator with Docker and save output to a directory
build_emulator(emulator_base, docker=True, output_dir="./emulator_output")
```