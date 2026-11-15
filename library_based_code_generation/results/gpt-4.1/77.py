```python
from emulator import Emulator, AutonomousSystem, Host, EthereumService, Blockchain, Node, Account, PhysicalNode, Filter

# Create emulator
emu = Emulator()

# Create 10 stub Autonomous Systems (AS) and hosts
as_list = []
host_list = []
for i in range(10):
    asys = AutonomousSystem(name=f"AS{i+1}")
    host = Host(name=f"Host{i+1}")
    asys.add_host(host)
    emu.add_autonomous_system(asys)
    as_list.append(asys)
    host_list.append(host)

# Create Ethereum service with saveState and override set to True
eth_service = EthereumService(saveState=True, override=True)

# Create two blockchains: POW and POA
pow_chain = Blockchain(name="POW_Chain", consensus="pow")
poa_chain = Blockchain(name="POA_Chain", consensus="poa")

# For each blockchain, create four nodes
pow_nodes = []
poa_nodes = []
for i in range(4):
    pow_node = Node(name=f"POW_Node{i+1}")
    poa_node = Node(name=f"POA_Node{i+1}")
    pow_chain.add_node(pow_node)
    poa_chain.add_node(poa_node)
    pow_nodes.append(pow_node)
    poa_nodes.append(poa_node)

# Set first two nodes as bootnodes and start mining
for node in pow_nodes[:2]:
    node.set_bootnode(True)
    node.start_mining()
for node in poa_nodes[:2]:
    node.set_bootnode(True)
    node.start_mining()

# For the third node, create accounts with a certain balance
pow_nodes[2].add_account(Account(balance=1000))
poa_nodes[2].add_account(Account(balance=1000))

# For the fourth node, set custom geth command options
pow_nodes[3].set_geth_options("--syncmode 'light' --cache 512")
poa_nodes[3].set_geth_options("--syncmode 'full' --cache 1024")

# Enable HTTP and WebSocket connections on certain nodes
pow_nodes[0].enable_http(True)
pow_nodes[0].enable_ws(True)
poa_nodes[1].enable_http(True)
poa_nodes[1].enable_ws(True)

# Set custom geth binary file on one of the nodes
pow_nodes[1].set_geth_binary("/usr/local/bin/custom_geth")

# Customize display names for visualization
pow_nodes[0].set_display_name("POW Bootnode 1")
pow_nodes[1].set_display_name("POW Bootnode 2")
pow_nodes[2].set_display_name("POW Account Node")
pow_nodes[3].set_display_name("POW Custom Node")
poa_nodes[0].set_display_name("POA Bootnode 1")
poa_nodes[1].set_display_name("POA Bootnode 2")
poa_nodes[2].set_display_name("POA Account Node")
poa_nodes[3].set_display_name("POA Custom Node")

# Bind virtual nodes to physical nodes using filters
for i, node in enumerate(pow_nodes + poa_nodes):
    phys_node = PhysicalNode(name=f"PhysNode{i+1}")
    node.bind_to_physical(phys_node, filter=Filter(type="location", value=f"rack{i%5}"))

# Add blockchains to Ethereum service
eth_service.add_blockchain(pow_chain)
eth_service.add_blockchain(poa_chain)

# Add Ethereum layer to emulator
emu.add_service(eth_service)

# Save the component to a file
emu.save_component("ethereum_emulator_component.yaml")

# Compile the emulator with Docker and save the output to a directory
emu.compile_with_docker(output_dir="emulator_output")
```