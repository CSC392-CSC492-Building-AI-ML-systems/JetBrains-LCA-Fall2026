```python
import os
from seed_emulator import (
    makeEmulatorBaseWith10StubASAndHosts,
    getNodeByAsnAndName,
    Service,
    TorService,
    TorServer,
    createDirectory,
    PoAServer,
    createControlService,
    ServiceMerger,
    doesAContainB,
    shouldMerge,
    _createService,
)
from seed_emulator.dns import (
    RootDNSServer,
    TldDNSServer,
    CcTldDNSServer,
    SecondLevelZoneDNSServer,
    LocalDNSServer,
    DNSZone,
    DNSRecord,
)
from seed_emulator.compiler import DockerCompiler

# 1. Create base emulator topology
emulator = makeEmulatorBaseWith10StubASAndHosts()

# 2. Ransomware Service
# Create ransomware attacker (no botnet, no Tor)
attacker_asn = 65001
attacker_host = getNodeByAsnAndName(emulator, attacker_asn, "host1")
ransomware_attacker = Service(
    name="ransomware_attacker",
    type="ransomware",
    config={
        "role": "attacker",
        "botnet": False,
        "tor": False,
    }
)
attacker_host.add_service(ransomware_attacker)

# Create 16 ransomware victims (no botnet)
victim_hosts = []
victim_services = []
for i in range(1, 17):
    asn = 65002 + (i // 4)  # Distribute victims across ASes
    host_name = f"host{i}"
    host = getNodeByAsnAndName(emulator, asn, host_name)
    victim_service = Service(
        name=f"ransomware_victim_{i}",
        type="ransomware",
        config={
            "role": "victim",
            "botnet": False,
        }
    )
    host.add_service(victim_service)
    victim_hosts.append(host)
    victim_services.append(victim_service)

# 3. Tor Service
tor_service = TorService(name="tor_network")

# Directory Authorities
for i in range(3):
    asn = 65005 + i
    host = getNodeByAsnAndName(emulator, asn, "host1")
    dir_auth = TorServer(
        name=f"tor_dir_auth_{i+1}",
        role="directory_authority"
    )
    host.add_service(dir_auth)
    tor_service.add_node(dir_auth)

# Relays
for i in range(5):
    asn = 65008 + (i % 2)
    host = getNodeByAsnAndName(emulator, asn, f"host{i+2}")
    relay = TorServer(
        name=f"tor_relay_{i+1}",
        role="relay"
    )
    host.add_service(relay)
    tor_service.add_node(relay)

# Exits
for i in range(2):
    asn = 65003 + i
    host = getNodeByAsnAndName(emulator, asn, "host2")
    exit_node = TorServer(
        name=f"tor_exit_{i+1}",
        role="exit"
    )
    host.add_service(exit_node)
    tor_service.add_node(exit_node)

# Clients
for i in range(4):
    asn = 65004 + i
    host = getNodeByAsnAndName(emulator, asn, "host3")
    client = TorServer(
        name=f"tor_client_{i+1}",
        role="client"
    )
    host.add_service(client)
    tor_service.add_node(client)

# Hidden Service linked to ransomware attacker
hidden_service = TorServer(
    name="tor_hidden_service_ransomware",
    role="hidden_service",
    linked_service=ransomware_attacker
)
attacker_host.add_service(hidden_service)
tor_service.add_node(hidden_service)

# Add Tor service to emulator
emulator.add_service(tor_service)

# 4. DNS Layer
# Root DNS Server
root_dns = RootDNSServer(
    name="root_dns",
    zones=[DNSZone(name=".", records=[])]
)

# TLD and ccTLD Servers
com_tld = TldDNSServer(
    name="com_tld",
    zones=[DNSZone(name="com.", records=[])]
)
org_tld = TldDNSServer(
    name="org_tld",
    zones=[DNSZone(name="org.", records=[])]
)
us_cctld = CcTldDNSServer(
    name="us_cctld",
    zones=[DNSZone(name="us.", records=[])]
)

# Second-level zone servers
example_com_zone = DNSZone(
    name="example.com.",
    records=[
        DNSRecord(type="A", name="www.example.com.", value="10.0.0.10"),
        DNSRecord(type="A", name="victim1.example.com.", value="10.0.1.11"),
    ]
)
example_com_sld = SecondLevelZoneDNSServer(
    name="example_com_sld",
    zones=[example_com_zone]
)

hidden_onion_zone = DNSZone(
    name="hidden.onion.",
    records=[
        DNSRecord(type="A", name="hidden.onion.", value="127.0.0.1"),
    ]
)
hidden_onion_sld = SecondLevelZoneDNSServer(
    name="hidden_onion_sld",
    zones=[hidden_onion_zone]
)

# Local DNS Server
local_dns_zone = DNSZone(
    name="local.",
    records=[
        DNSRecord(type="A", name="attacker.local.", value="10.0.2.1"),
        DNSRecord(type="A", name="victim.local.", value="10.0.2.2"),
    ]
)
local_dns = LocalDNSServer(
    name="local_dns",
    zones=[local_dns_zone]
)

# Add DNS servers to emulator
emulator.add_service(root_dns)
emulator.add_service(com_tld)
emulator.add_service(org_tld)
emulator.add_service(us_cctld)
emulator.add_service(example_com_sld)
emulator.add_service(hidden_onion_sld)
emulator.add_service(local_dns)

# 5. Compile emulator with Docker compiler and custom base images
output_dir = "./output_emulator"
os.makedirs(output_dir, exist_ok=True)

compiler = DockerCompiler(
    base_images={
        "victim": "custom/victim-base:latest",
        "attacker": "custom/attacker-base:latest"
    }
)
compiler.compile(emulator, output_dir=output_dir)

# 6. Copy necessary files and make script executable
os.system(f"cp ./scripts/setup.sh {output_dir}/setup.sh")
os.system(f"chmod +x {output_dir}/setup.sh")
```