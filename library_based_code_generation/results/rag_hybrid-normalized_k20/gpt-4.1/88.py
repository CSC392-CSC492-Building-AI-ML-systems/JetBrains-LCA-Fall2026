```python
from seedemu import *
from seedemu.layers import *
from seedemu.services.ethereum import *

# Create emulator base with 10 stub AS and 3 hosts per stub AS
emu = makeEmulatorBaseWith10StubASAndHosts(3)

# Create Ethereum service layer and sub-layer (blockchain)
eth_layer = Ethereum()
blockchain = eth_layer.createBlockchain(name="pos", consensus="POS")
blockchain.setTerminalTotalDifficulty(30)

# For each host in each stub AS, create blockchain virtual node, docker label, enable geth http, etc.
for asn in range(65001, 65011):
    for host_id in range(1, 4):
        host_name = f"h{host_id}"
        node_name = f"as{asn}-{host_name}"
        v_node = eth_layer.setVirtualNode(node_name)
        v_node.setDockerContainerLabel(f"eth-{node_name}")
        v_node.connectToBlockchain("pos")
        v_node.connect_to_geth_pos(http=True)
        v_node.setDisplayName(f"Ethereum Node {node_name}")
        v_node.bindToPhysicalNode(asn, host_name)

        # Set BeaconSetupNode for the first host of the first AS
        if asn == 65001 and host_id == 1:
            v_node.setBeaconSetupNode(True)
        # Set BootNode for the second host of the first AS
        elif asn == 65001 and host_id == 2:
            v_node.setBootNode(True)
        # Set validator nodes for all third hosts in each AS
        elif host_id == 3:
            v_node.setValidator(True)

# Add Ethereum layer to the emulator
emu.addLayer(eth_layer)

# Render and compile with Docker, internetMap and etherView enabled
emu.render()
emu.compile(
    './output',
    docker=True,
    internetMap=True,
    etherView=True,
    override=True
)
```