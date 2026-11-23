```python
from seedemu import Emulator
from seedemu.layers import Service, Blockchain, Ethereum
from seedemu.services.ethereum import ConsensusMechanism
from seedemu.services.ethereum import EthereumNodeType

# Create emulator base with 10 stub AS and 3 hosts per stub AS
emu = Emulator()
emu.makeEmulatorBaseWith10StubASAndHosts(hostsPerStub=3)

# Create Ethereum service layer
service_layer = Service()

# Create Ethereum blockchain sub-layer
eth_layer = Ethereum()
blockchain = Blockchain(
    name="pos",
    consensus=ConsensusMechanism.POS,
    terminal_total_difficulty=30
)
eth_layer.addBlockchain(blockchain)

# Add Ethereum layer as a sub-layer of the service layer
service_layer.addSubLayer(eth_layer)

# For each host in each AS, create blockchain virtual node, docker label, enable Geth HTTP, set roles, customize display name, bind virtual node
for asn in range(1, 11):
    for host in range(1, 4):
        node_id = f"as{asn}-host{host}"
        v_node = eth_layer.createVirtualNode(node_id, blockchain="pos")
        eth_layer.setDockerLabel(node_id, f"ethereum-node-{node_id}")
        eth_layer.enableGethHttp(node_id)
        eth_layer.setVirtualNode(node_id, asn, host)

        # Set roles based on conditions
        if asn == 1 and host == 1:
            eth_layer.setNodeType(node_id, EthereumNodeType.BeaconSetupNode)
            eth_layer.setDisplayName(node_id, "BeaconSetupNode")
        elif asn == 1 and host == 2:
            eth_layer.setNodeType(node_id, EthereumNodeType.BootNode)
            eth_layer.setDisplayName(node_id, "BootNode")
        elif host == 3:
            eth_layer.setNodeType(node_id, EthereumNodeType.Validator)
            eth_layer.setDisplayName(node_id, f"Validator-{asn}")
        else:
            eth_layer.setNodeType(node_id, EthereumNodeType.FullNode)
            eth_layer.setDisplayName(node_id, f"FullNode-{asn}-{host}")

# Add the Ethereum service layer to the emulator
emu.addLayer(service_layer)

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