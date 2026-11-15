```python
from seedemu import Emulator, AutonomousSystem, Host, ServiceLayer, BlockchainLayer, BlockchainVirtualNode, DockerContainer, BlockchainConsensus, BlockchainNodeRole
from seedemu.services.ethereum import EthereumLayer, EthereumBlockchain, EthereumVirtualNode, EthereumNodeRole

# Create emulator base with 10 stub AS and 3 hosts per stub AS
emu = Emulator()
num_as = 10
hosts_per_as = 3
as_list = []
host_list = []

for asn in range(1, num_as + 1):
    stub_as = AutonomousSystem(asn)
    for h in range(1, hosts_per_as + 1):
        host = Host(f"h{asn}_{h}")
        stub_as.addHost(host)
        host_list.append((asn, h, host))
    emu.addAutonomousSystem(stub_as)
    as_list.append(stub_as)

# Create Ethereum service layer and sub-layer (blockchain)
eth_layer = EthereumLayer()
blockchain = EthereumBlockchain(name="pos", consensus=BlockchainConsensus.POS)
blockchain.setTerminalTotalDifficulty(30)
eth_layer.addBlockchain(blockchain)

# For each host, create a blockchain virtual node, docker label, enable geth http, set roles, customize display name, bind to physical node
for asn, h, host in host_list:
    node_name = f"ethnode_{asn}_{h}"
    v_node = EthereumVirtualNode(node_name)
    v_node.setBlockchain("pos")
    v_node.setDockerContainer(DockerContainer(label=f"geth_{asn}_{h}"))
    v_node.enableGethHttp()
    v_node.setDisplayName(f"AS{asn}-Host{h}")

    # Set roles based on conditions
    if asn == 1 and h == 1:
        v_node.setRole(EthereumNodeRole.BeaconSetupNode)
    elif asn == 1 and h == 2:
        v_node.setRole(EthereumNodeRole.BootNode)
    elif h == 3:
        v_node.setRole(EthereumNodeRole.Validator)
    else:
        v_node.setRole(EthereumNodeRole.FullNode)

    v_node.bindPhysicalNode(host.getName())
    blockchain.addVirtualNode(v_node)

# Add Ethereum layer to emulator
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