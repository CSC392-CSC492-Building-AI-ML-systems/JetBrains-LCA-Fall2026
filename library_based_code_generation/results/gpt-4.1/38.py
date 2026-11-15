```python
from seedemu import Emulator, Layer, AutonomousSystem, Network, Router, Host, InternetExchange, Ebgp, WebService, Utility
from seedemu.layers import Physical, Application
from seedemu.services.web import WebServiceNode
from seedemu.core import saveComponentFile
import shutil

# Create the emulator
emu = Emulator()

# --- Transit AS ---
transit_asn = 100
transit_as = AutonomousSystem(transit_asn, displayName="Transit AS")

# Create two Internet Exchanges
ix1 = InternetExchange(1, displayName="IX-East")
ix2 = InternetExchange(2, displayName="IX-West")
transit_as.addInternetExchange(ix1)
transit_as.addInternetExchange(ix2)

# Create three internal networks
net1 = Network(1, displayName="Transit-Net-1")
net2 = Network(2, displayName="Transit-Net-2")
net3 = Network(3, displayName="Transit-Net-3")
transit_as.addNetwork(net1)
transit_as.addNetwork(net2)
transit_as.addNetwork(net3)

# Create four routers in a linear structure
r1 = Router(1, displayName="Transit-R1")
r2 = Router(2, displayName="Transit-R2")
r3 = Router(3, displayName="Transit-R3")
r4 = Router(4, displayName="Transit-R4")
transit_as.addRouter(r1)
transit_as.addRouter(r2)
transit_as.addRouter(r3)
transit_as.addRouter(r4)

# Connect routers linearly: r1 <-> r2 <-> r3 <-> r4
net1.addNode(r1)
net1.addNode(r2)
net2.addNode(r2)
net2.addNode(r3)
net3.addNode(r3)
net3.addNode(r4)

# Connect routers to IXes
ix1.addNode(r1)
ix2.addNode(r4)

# --- Stub ASes ---
stub1_asn = 200
stub2_asn = 201
stub3_asn = 202

# Stub 1
stub1 = AutonomousSystem(stub1_asn, displayName="Stub AS 1")
stub1_net = Network(10, displayName="Stub1-Net")
stub1.addNetwork(stub1_net)
stub1_router = Router(10, displayName="Stub1-Router")
stub1.addRouter(stub1_router)
stub1_net.addNode(stub1_router)
stub1_host1 = Host(10, displayName="Stub1-Host1")
stub1_host2 = Host(11, displayName="Stub1-Host2")
stub1.addHost(stub1_host1)
stub1.addHost(stub1_host2)
stub1_net.addNode(stub1_host1)
stub1_net.addNode(stub1_host2)
# Install software and create account on host1
stub1_host1.addSoftware("nginx")
stub1_host1.addAccount("webadmin", password="webpass")

# Stub 2
stub2 = AutonomousSystem(stub2_asn, displayName="Stub AS 2")
stub2_net = Network(20, displayName="Stub2-Net")
stub2.addNetwork(stub2_net)
stub2_router = Router(20, displayName="Stub2-Router")
stub2.addRouter(stub2_router)
stub2_net.addNode(stub2_router)
stub2_host1 = Host(20, displayName="Stub2-Host1")
stub2_host2 = Host(21, displayName="Stub2-Host2")
stub2.addHost(stub2_host1)
stub2.addHost(stub2_host2)
stub2_net.addNode(stub2_host1)
stub2_net.addNode(stub2_host2)

# Stub 3 using utility function
stub3 = Utility.createStubAs(stub3_asn, numHosts=1, displayName="Stub AS 3")
# Further customize stub3
stub3_host = Host(30, displayName="Stub3-Host1")
stub3.addHost(stub3_host)
stub3.getNetwork(0).addNode(stub3_host)

# --- Add ASes to a layer ---
as_layer = Layer("AS Layer")
as_layer.addAutonomousSystem(transit_as)
as_layer.addAutonomousSystem(stub1)
as_layer.addAutonomousSystem(stub2)
as_layer.addAutonomousSystem(stub3)

emu.addLayer(as_layer)

# --- BGP Peering ---
ebgp = Ebgp()
# Transit AS as ISP for all stub ASes
ebgp.addPeering(transit_asn, stub1_asn)
ebgp.addPeering(transit_asn, stub2_asn)
ebgp.addPeering(transit_asn, stub3_asn)
# Direct peering between stub1 and stub2
ebgp.addPeering(stub1_asn, stub2_asn)
emu.addLayer(ebgp)

# --- Web Service Layer ---
web_layer = Application("Web Service Layer")
web_node1 = WebServiceNode("web1", displayName="Web Service 1")
web_node2 = WebServiceNode("web2", displayName="Web Service 2")
web_layer.addWebServiceNode(web_node1)
web_layer.addWebServiceNode(web_node2)

# Bind web service nodes to physical hosts
web_layer.bindWebServiceNodeToPhysicalNode("web1", stub1_host1.getId())
web_layer.bindWebServiceNodeToPhysicalNode("web2", stub2_host1.getId())

emu.addLayer(web_layer)

# --- Save to component file ---
saveComponentFile(emu, "emulator.component")

# --- Render the emulator ---
emu.render()

# --- Change display names for nodes hosting web services ---
stub1_host1.setDisplayName("WebServer-Stub1")
stub2_host1.setDisplayName("WebServer-Stub2")

# --- Compile with Docker ---
docker_images = {
    "router": "mydockerhub/custom-router:latest",
    "host": "./local-images/custom-host-image.tar"
}
emu.compile(
    outputFolder="output",
    dockerImages=docker_images,
    generateDockerFiles=True
)

# Copy base container image to output folder
shutil.copy("./local-images/base-container-image.tar", "output/base-container-image.tar")
```