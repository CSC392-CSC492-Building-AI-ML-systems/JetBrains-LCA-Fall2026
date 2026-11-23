```python
from emulatorlib import (
    makeEmulatorBaseWith10StubASAndHosts,
    createBlockchain,
    connect_to_geth_pow,
    connect_to_geth_poa,
    createEmulatorAccountFromMnemonic,
    setCustomGeth,
    get_eth_nodes,
    addEmulatorAccounts,
    setVirtualNode,
    getVirtualNodes,
    build_emulator,
    createEmulation
)

# Step 1: Create emulator base with 10 stub AS and hosts
emulator = makeEmulatorBaseWith10StubASAndHosts()

# Step 2: Create Ethereum service with saveState and override set to True
eth_service = createEmulation(
    emulator=emulator,
    service_type="ethereum",
    saveState=True,
    override=True
)

# Step 3: Create two blockchains: POW and POA
pow_chain = createBlockchain(type="pow", name="POW-Chain")
poa_chain = createBlockchain(type="poa", name="POA-Chain")

# Step 4: For each blockchain, create four nodes
pow_nodes = []
poa_nodes = []
for i in range(4):
    pow_node = pow_chain.create_node(name=f"POW-Node-{i+1}")
    poa_node = poa_chain.create_node(name=f"POA-Node-{i+1}")
    pow_nodes.append(pow_node)
    poa_nodes.append(poa_node)

# Step 5: Set first two nodes as bootnodes and start mining
for node in pow_nodes[:2]:
    connect_to_geth_pow(node, bootnode=True, mining=True)
for node in poa_nodes[:2]:
    connect_to_geth_poa(node, bootnode=True, mining=True)

# Step 6: For third node, create accounts with a certain balance
mnemonic = "test test test test test test test test test test test junk"
account = createEmulatorAccountFromMnemonic(mnemonic, index=0, balance=1000000000000000000)
addEmulatorAccounts(pow_nodes[2], [account])
addEmulatorAccounts(poa_nodes[2], [account])

# Step 7: For fourth node, set custom geth command options
custom_cmd_pow = "--maxpeers 50 --syncmode 'full'"
custom_cmd_poa = "--maxpeers 25 --syncmode 'light'"
pow_nodes[3].set_custom_geth_command(custom_cmd_pow)
poa_nodes[3].set_custom_geth_command(custom_cmd_poa)

# Step 8: Enable HTTP and WebSocket connections on certain nodes
pow_nodes[0].enable_http(port=8545)
pow_nodes[1].enable_ws(port=8546)
poa_nodes[0].enable_http(port=9545)
poa_nodes[1].enable_ws(port=9546)

# Step 9: Set custom geth binary file on one of the nodes
setCustomGeth(pow_nodes[3], "/usr/local/bin/custom-geth")

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
virtual_nodes = getVirtualNodes(emulator)
for i, v_node in enumerate(virtual_nodes):
    setVirtualNode(v_node, physical_node_filter=f"host{i%10+1}")

# Step 12: Add Ethereum layer to emulator
emulator.add_layer(eth_service)

# Step 13: Save the component to a file
emulator.save_component("emulator_component.json")

# Step 14: Compile the emulator with Docker and save output to a directory
build_emulator(emulator, output_dir="./emulator_output", use_docker=True)
```