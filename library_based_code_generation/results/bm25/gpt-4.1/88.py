```python
from seedemu import *
from seedemu.layers import *
from seedemu.services.ethereum import *

# Create emulator base with 10 stub AS and 3 hosts per stub AS
emu = Emulator()
emu.addLayer(makeEmulatorBaseWith10StubASAndHosts(10, 3))

# Create Ethereum service layer and sub-layer for blockchain
eth_layer = Ethereum()
blockchain = eth_layer.createBlockchain(name="pos", consensus="POS")
blockchain.setTerminalTotalDifficulty(30)

# For each host in each AS, create blockchain virtual node, Docker label, enable Geth HTTP
for asn in range(1, 11):
    for host_idx in range(1, 4):
        host_name = f"h{host_idx}"
        node_name = f"as{asn}-{host_name}"
        # Create blockchain virtual node
        vnode = eth_layer.createVirtualNode(node_name)
        # Bind virtual node to physical node
        eth_layer.bindVirtualNodeToPhysicalNode(node_name, asn, host_name)
        # Add Docker container label
        eth_layer.setDockerLabel(node_name, f"ethereum-node-{node_name}")
        # Enable Geth HTTP communication
        eth_layer.connect_to_geth_pos(node_name, blockchain.name, http=True)
        # Customize display name
        eth_layer.setDisplayName(node_name, f"Ethereum Node {node_name}")

# Set specific hosts as BeaconSetupNode, BootNode, and validator nodes
# Example: as1-h1 as BeaconSetupNode, as1-h2 as BootNode, as1-h3 as validator, as2-h1 as validator, etc.
eth_layer.setNodeRole("as1-h1", blockchain.name, NodeRole.BeaconSetupNode)
eth_layer.setNodeRole("as1-h2", blockchain.name, NodeRole.BootNode)
eth_layer.setNodeRole("as1-h3", blockchain.name, NodeRole.Validator)
eth_layer.setNodeRole("as2-h1", blockchain.name, NodeRole.Validator)
eth_layer.setNodeRole("as2-h2", blockchain.name, NodeRole.Validator)
eth_layer.setNodeRole("as3-h1", blockchain.name, NodeRole.Validator)
eth_layer.setNodeRole("as4-h1", blockchain.name, NodeRole.Validator)
eth_layer.setNodeRole("as5-h1", blockchain.name, NodeRole.Validator)
eth_layer.setNodeRole("as6-h1", blockchain.name, NodeRole.Validator)
eth_layer.setNodeRole("as7-h1", blockchain.name, NodeRole.Validator)
eth_layer.setNodeRole("as8-h1", blockchain.name, NodeRole.Validator)
eth_layer.setNodeRole("as9-h1", blockchain.name, NodeRole.Validator)
eth_layer.setNodeRole("as10-h1", blockchain.name, NodeRole.Validator)

# Add Ethereum layer to emulator
emu.addLayer(eth_layer)

# Render and compile with Docker, internetMap and etherView enabled
emu.render()
emu.compile(
    './output',
    compileDocker=True,
    compileInternetMap=True,
    compileEtherView=True,
    override=True
)
```