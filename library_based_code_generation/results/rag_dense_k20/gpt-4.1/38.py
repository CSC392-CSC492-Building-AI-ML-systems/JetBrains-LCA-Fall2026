```python
from seedemu.core import Emulator, Layer, Node, AutonomousSystem, InternetExchange, Network, Router, Host, Service, Account
from seedemu.layers import Bgp, WebService
from seedemu.services import Web
from seedemu.utility import makeEmulatorBaseWith5StubASAndHosts

# Create the emulator
emu = Emulator()

# --- Transit AS ---
transit_asn = 100
transit_as = AutonomousSystem(transit_asn, displayName="Transit AS")

# Create two Internet Exchanges
ix1 = InternetExchange(1, displayName="IX Alpha")
ix2 = InternetExchange(2, displayName="IX Beta")
transit_as.addInternetExchange(ix1)
transit_as.addInternetExchange(ix2)

# Create three internal networks
net1 = Network(1, displayName="TransitNet1")
net2 = Network(2, displayName="TransitNet2")
net3 = Network(3, displayName="TransitNet3")
transit_as.addNetwork(net1)
transit_as.addNetwork(net2)
transit_as.addNetwork(net3)

# Create four routers in a linear structure
r1 = Router(1, displayName="TransitRouter1")
r2 = Router(2, displayName="TransitRouter2")
r3 = Router(3, displayName="TransitRouter3")
r4 = Router(4, displayName="TransitRouter4")
transit_as.addRouter(r1)
transit_as.addRouter(r2)
transit_as.addRouter(r3)
transit_as.addRouter(r4)

# Link routers linearly: r1 <-> r2 <-> r3 <-> r4
net1.addNode(r1)
net1.addNode(r2)
net2.addNode(r2)
net2.addNode(r3)
net3.addNode(r3)
net3.addNode(r4)

# Add transit AS to a layer
transit_layer = Layer()
transit_layer.addAutonomousSystem(transit_as)
emu.addLayer(transit_layer)

# --- Stub ASes ---
stub_asn1 = 200
stub_asn2 = 201
stub_asn3 = 202

# Stub AS 1
stub1 = AutonomousSystem(stub_asn1, displayName="Stub AS 1")
stub1_net = Network(10, displayName="Stub1Net")
stub1_router = Router(10, displayName="Stub1Router")
stub1_host1 = Host(10, displayName="Stub1Host1")
stub1_host2 = Host(11, displayName="Stub1Host2")
stub1_net.addNode(stub1_router)
stub1_net.addNode(stub1_host1)
stub1_net.addNode(stub1_host2)
stub1.addNetwork(stub1_net)
stub1.addRouter(stub1_router)
stub1.addHost(stub1_host1)
stub1.addHost(stub1_host2)
# Add software and account to host1
stub1_host1.addSoftware("nginx")
stub1_host1.addAccount(Account("webadmin", password="webpass"))
emu.addLayer(Layer().addAutonomousSystem(stub1))

# Stub AS 2
stub2 = AutonomousSystem(stub_asn2, displayName="Stub AS 2")
stub2_net = Network(20, displayName="Stub2Net")
stub2_router = Router(20, displayName="Stub2Router")
stub2_host1 = Host(20, displayName="Stub2Host1")
stub2_host2 = Host(21, displayName="Stub2Host2")
stub2_net.addNode(stub2_router)
stub2_net.addNode(stub2_host1)
stub2_net.addNode(stub2_host2)
stub2.addNetwork(stub2_net)
stub2.addRouter(stub2_router)
stub2.addHost(stub2_host1)
stub2.addHost(stub2_host2)
emu.addLayer(Layer().addAutonomousSystem(stub2))

# Stub AS 3 using utility function and customize
utility_layer = Layer()
utility_as = makeEmulatorBaseWith5StubASAndHosts(asnBase=stub_asn3, count=1)
# The utility function returns a Layer, extract the AS and customize
for asys in utility_as.getAutonomousSystems():
    asys.setDisplayName("Stub AS 3")
    # Add custom software to first host
    hosts = asys.getHosts()
    if hosts:
        hosts[0].addSoftware("curl")
    utility_layer.addAutonomousSystem(asys)
emu.addLayer(utility_layer)

# --- BGP Peering ---
bgp_layer = Bgp()

# Set transit AS as ISP for all stub ASes
bgp_layer.addProviderCustomer(transit_asn, stub_asn1)
bgp_layer.addProviderCustomer(transit_asn, stub_asn2)
bgp_layer.addProviderCustomer(transit_asn, stub_asn3)

# Direct peering between stub AS 1 and stub AS 2
bgp_layer.addPeer(stub_asn1, stub_asn2)

emu.addLayer(bgp_layer)

# --- Web Service Layer ---
web_layer = WebService()
web_node1 = Service("WebService1", service=Web(), displayName="Web Node 1")
web_node2 = Service("WebService2", service=Web(), displayName="Web Node 2")
web_layer.addService(web_node1)
web_layer.addService(web_node2)

# Bind virtual web service nodes to physical hosts
web_layer.bindServiceToHost("WebService1", stub1_host1.getId())
web_layer.bindServiceToHost("WebService2", stub2_host1.getId())

emu.addLayer(web_layer)

# --- Save to component file ---
emu.save("emulation_component.bin")

# --- Render the emulator ---
emu.render()

# --- Change display names for nodes hosting web services ---
stub1_host1.setDisplayName("WebServer-Alpha")
stub2_host1.setDisplayName("WebServer-Beta")

# --- Compile the emulator using Docker ---
custom_images = {
    "router": "mydockerhub/custom-router:latest",
    "host": "./local-images/custom-host-image.tar"
}
emu.compile(
    output="output_folder",
    docker=True,
    dockerImages=custom_images,
    generateDockerFiles=True,
    copyBaseImage=True
)
```