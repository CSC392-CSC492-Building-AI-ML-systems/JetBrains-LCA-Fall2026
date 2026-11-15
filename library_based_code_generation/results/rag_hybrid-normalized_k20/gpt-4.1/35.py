```python
import random
from sc2 import maps, run_game, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import *
from sc2.bot_ai import BotAI

class ProtossBot(BotAI):
    async def on_step(self, iteration):
        await self.distribute_workers()
        await self.build_pylons()
        await self.build_probes()
        await self.build_assimilators()
        await self.build_gateway()
        await self.build_cybercore()
        await self.research_warpgate()
        await self.morph_gateways()
        await self.build_proxy_pylon()
        await self.warp_units()
        await self.train_army()
        await self.chrono_boost()
        await self.attack()

    async def build_pylons(self):
        if self.supply_left < 5 and self.already_pending(PYLON) == 0:
            nexuses = self.townhalls.ready
            if nexuses.exists:
                await self.build(PYLON, near=nexuses.first.position.towards(self.game_info.map_center, 5))

    async def build_probes(self):
        for nexus in self.townhalls.ready.idle:
            if self.can_afford(PROBE) and self.workers.amount < self.townhalls.amount * 22:
                nexus.train(PROBE)

    async def build_assimilators(self):
        for nexus in self.townhalls.ready:
            vaspenes = self.vespene_geyser.closer_than(15.0, nexus)
            for vaspene in vaspenes:
                if not self.gas_buildings.closer_than(1.0, vaspene).exists and self.can_afford(ASSIMILATOR):
                    worker = self.select_build_worker(vaspene.position)
                    if worker:
                        worker.build(ASSIMILATOR, vaspene)
                        break

    async def build_gateway(self):
        if self.structures(PYLON).ready.exists:
            pylon = self.structures(PYLON).ready.random
            if self.structures(GATEWAY).amount + self.already_pending(GATEWAY) < 2 and self.can_afford(GATEWAY):
                await self.build(GATEWAY, near=pylon)

    async def build_cybercore(self):
        if self.structures(GATEWAY).ready.exists and not self.structures(CYBERNETICSCORE):
            pylon = self.structures(PYLON).ready.random
            if self.can_afford(CYBERNETICSCORE):
                await self.build(CYBERNETICSCORE, near=pylon)

    async def research_warpgate(self):
        for core in self.structures(CYBERNETICSCORE).ready.idle:
            if self.can_afford(RESEARCH_WARPGATE) and not self.already_pending_upgrade(WARPGATERESEARCH):
                core.research(WARPGATERESEARCH)

    async def morph_gateways(self):
        if self.structures(CYBERNETICSCORE).ready and self.structures(GATEWAY).ready.exists:
            if self.tech_ready_upgrade(WARPGATERESEARCH):
                for gateway in self.structures(GATEWAY).ready:
                    if gateway.is_idle:
                        gateway(AbilityId.MORPH_WARPGATE)

    async def build_proxy_pylon(self):
        if self.structures(PYLON).amount < 3 and self.can_afford(PYLON):
            pos = self.enemy_start_locations[0].towards(self.game_info.map_center, 20)
            if not self.structures(PYLON).closer_than(10, pos).exists:
                await self.build(PYLON, near=pos)

    async def warp_units(self):
        if self.structures(WARPGATE).ready.exists and self.structures(PYLON).ready.exists:
            for warpgate in self.structures(WARPGATE).ready.idle:
                abilities = await self.get_available_abilities(warpgate)
                if AbilityId.WARPGATETRAIN_STALKER in abilities and self.can_afford(STALKER):
                    pylon = self.structures(PYLON).ready.random
                    placement = await self.find_placement(AbilityId.WARPGATETRAIN_STALKER, pylon.position, placement_step=1)
                    if placement:
                        warpgate.warp_in(STALKER, placement)

    async def train_army(self):
        for gateway in self.structures(GATEWAY).ready.idle:
            if self.can_afford(STALKER) and self.structures(CYBERNETICSCORE).ready:
                gateway.train(STALKER)

    async def chrono_boost(self):
        for nexus in self.townhalls.ready:
            if nexus.energy >= 50:
                if self.structures(CYBERNETICSCORE).ready.exists and not self.already_pending_upgrade(WARPGATERESEARCH):
                    core = self.structures(CYBERNETICSCORE).ready.first
                    nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, core)
                else:
                    nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, nexus)

    async def attack(self):
        stalkers = self.units(STALKER).idle
        if stalkers.exists:
            targets = self.enemy_units
            if targets.exists:
                for stalker in stalkers:
                    stalker.attack(targets.closest_to(stalker))
            else:
                for stalker in stalkers:
                    stalker.attack(self.enemy_start_locations[0])

    async def on_unit_destroyed(self, unit_tag):
        if not self.townhalls.exists:
            for unit in self.units:
                unit.attack(self.enemy_start_locations[0])

run_game(
    maps.get("(2)CatalystLE"),
    [Bot(Race.Protoss, ProtossBot()), Computer(Race.Protoss, Difficulty.Easy)],
    realtime=False,
)
```