```python
from emulator_library import (
    makeEmulatorBaseWith10StubASAndHosts,
    createBlockchain,
    getNodeByAsnAndName,
    registerNodes,
    enableOn,
    setCustomGeth,
    connect_to_geth_pow,
    connect_to_geth_poa,
    getBootNodes,
    getMinerNodes,
    get_eth_nodes,
    createHostsOnNetwork,
    get_balance_of_account,
    makeStubAsWithHosts,
    connectToBlockchain,
    getVirtualNodes,
    load_eth_nodes,
    __getEdgeNodes,
    saveComponentToFile,
    compileEmulatorWithDocker
)

# Step 1: Create emulator base with 10 stub AS and hosts
emulator = makeEmulatorBaseWith10StubASAndHosts()

# Step 2: Create Ethereum service with saveState and override set to True
ethereum_service = {
    "saveState": True,
    "override": True
}
emulator.addService("ethereum", ethereum_service)

# Step 3: Create two blockchains: POW and POA
blockchain_pow = createBlockchain(type="POW", name="POW-Chain")
blockchain_poa = createBlockchain(type="POA", name="POA-Chain")

# Step 4: For each blockchain, create four nodes
pow_nodes = []
poa_nodes = []
for i in range(4):
    pow_node = blockchain_pow.createNode(name=f"POW-Node-{i+1}")
    poa_node = blockchain_poa.createNode(name=f"POA-Node-{i+1}")
    pow_nodes.append(pow_node)
    poa_nodes.append(poa_node)

# Step 5: Set first two nodes as bootnodes and start mining on them
for node in pow_nodes[:2]:
    blockchain_pow.setBootNode(node)
    blockchain_pow.startMining(node)
for node in poa_nodes[:2]:
    blockchain_poa.setBootNode(node)
    blockchain_poa.startMining(node)

# Step 6: For the third node, create accounts with a certain balance
account_balance = 100 * 10**18  # 100 ETH
pow_nodes[2].createAccount(balance=account_balance)
poa_nodes[2].createAccount(balance=account_balance)

# Step 7: For the fourth node, set custom geth command options
pow_nodes[3].setGethOptions(["--syncmode", "light", "--maxpeers", "10"])
poa_nodes[3].setGethOptions(["--syncmode", "full", "--maxpeers", "5"])

# Step 8: Enable HTTP and WebSocket connections on certain nodes
pow_nodes[0].enableHTTP(port=8545)
pow_nodes[1].enableWS(port=8546)
poa_nodes[0].enableHTTP(port=8547)
poa_nodes[1].enableWS(port=8548)

# Step 9: Set custom geth binary file on one of the nodes
custom_geth_path = "/usr/local/bin/custom_geth"
pow_nodes[2].setCustomGeth(custom_geth_path)

# Step 10: Customize display names for visualization
pow_nodes[0].setDisplayName("POW Bootnode 1")
pow_nodes[1].setDisplayName("POW Bootnode 2")
pow_nodes[2].setDisplayName("POW Account Node")
pow_nodes[3].setDisplayName("POW Custom Geth Node")
poa_nodes[0].setDisplayName("POA Bootnode 1")
poa_nodes[1].setDisplayName("POA Bootnode 2")
poa_nodes[2].setDisplayName("POA Account Node")
poa_nodes[3].setDisplayName("POA Custom Geth Node")

# Step 11: Bind virtual nodes to physical nodes using filters
for i, node in enumerate(pow_nodes + poa_nodes):
    node.bindToPhysicalNode(filter={"asn": (i % 10) + 1})

# Step 12: Add the Ethereum layer to the emulator
emulator.addBlockchain(blockchain_pow)
emulator.addBlockchain(blockchain_poa)

# Step 13: Save the component to a file
saveComponentToFile(emulator, "emulator_component.json")

# Step 14: Compile the emulator with Docker and save output to a directory
compileEmulatorWithDocker(emulator, output_dir="./emulator_docker_output")
```