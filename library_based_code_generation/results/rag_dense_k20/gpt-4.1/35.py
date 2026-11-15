```python
import random
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Race, Difficulty
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.upgrade_id import UpgradeId
from sc2.ids.buff_id import BuffId
from sc2.player import Bot, Computer
from sc2.position import Point2
from sc2.main import run_game

class ProtossBot(BotAI):
    async def on_step(self, iteration):
        if not self.townhalls:
            # No Nexuses left, try to fight with remaining units
            for unit in self.units(UnitTypeId.STALKER):
                targets = self.enemy_units | self.enemy_structures
                if targets:
                    target = targets.closest_to(unit)
                else:
                    target = self.enemy_start_locations[0]
                if unit.weapon_cooldown == 0:
                    unit.attack(target)
            return

        await self.distribute_workers()
        await self.build_pylons()
        await self.build_assimilators()
        await self.train_probes()
        await self.build_gateway()
        await self.build_cybercore()
        await self.research_warpgate()
        await self.morph_gateways()
        await self.warp_units()
        await self.build_proxy_pylon()
        await self.chrono_boost()
        await self.stalker_attack()

    async def build_pylons(self):
        if self.supply_left < 5 and self.already_pending(UnitTypeId.PYLON) == 0:
            nexuses = self.townhalls.ready
            if nexuses.exists and self.can_afford(UnitTypeId.PYLON):
                pos = nexuses.first.position.towards(self.game_info.map_center, 5)
                await self.build(UnitTypeId.PYLON, near=pos)

    async def build_assimilators(self):
        for nexus in self.townhalls.ready:
            vgs = self.vespene_geyser.closer_than(15.0, nexus)
            for vg in vgs:
                if not self.can_afford(UnitTypeId.ASSIMILATOR):
                    break
                worker = self.select_build_worker(vg.position)
                if worker and not self.gas_buildings.closer_than(1.0, vg).exists:
                    await self.do(worker.build(UnitTypeId.ASSIMILATOR, vg))

    async def train_probes(self):
        for nexus in self.townhalls.ready.idle:
            if self.can_afford(UnitTypeId.PROBE) and self.workers.amount < self.townhalls.amount * 22:
                await self.do(nexus.train(UnitTypeId.PROBE))

    async def build_gateway(self):
        if self.structures(UnitTypeId.PYLON).ready.exists:
            if self.structures(UnitTypeId.GATEWAY).amount + self.structures(UnitTypeId.WARPGATE).amount < 2:
                pylon = self.structures(UnitTypeId.PYLON).ready.random
                if self.can_afford(UnitTypeId.GATEWAY) and self.already_pending(UnitTypeId.GATEWAY) == 0:
                    await self.build(UnitTypeId.GATEWAY, near=pylon)

    async def build_cybercore(self):
        if self.structures(UnitTypeId.GATEWAY).ready.exists and not self.structures(UnitTypeId.CYBERNETICSCORE):
            pylon = self.structures(UnitTypeId.PYLON).ready.random
            if self.can_afford(UnitTypeId.CYBERNETICSCORE) and self.already_pending(UnitTypeId.CYBERNETICSCORE) == 0:
                await self.build(UnitTypeId.CYBERNETICSCORE, near=pylon)

    async def research_warpgate(self):
        if self.structures(UnitTypeId.CYBERNETICSCORE).ready.exists:
            cybercore = self.structures(UnitTypeId.CYBERNETICSCORE).ready.first
            if self.can_afford(AbilityId.RESEARCH_WARPGATE) and not self.already_pending_upgrade(UpgradeId.WARPGATERESEARCH):
                abilities = await self.get_available_abilities(cybercore)
                if AbilityId.RESEARCH_WARPGATE in abilities:
                    await self.do(cybercore(AbilityId.RESEARCH_WARPGATE))

    async def morph_gateways(self):
        if self.structures(UnitTypeId.CYBERNETICSCORE).ready.exists and self.structures(UnitTypeId.GATEWAY).ready.exists:
            if self.state.upgrades and UpgradeId.WARPGATERESEARCH in self.state.upgrades:
                for gateway in self.structures(UnitTypeId.GATEWAY).ready:
                    abilities = await self.get_available_abilities(gateway)
                    if AbilityId.MORPH_WARPGATE in abilities:
                        await self.do(gateway(AbilityId.MORPH_WARPGATE))

    async def warp_units(self):
        if self.structures(UnitTypeId.WARPGATE).ready.exists and self.structures(UnitTypeId.PYLON).ready.exists:
            for warpgate in self.structures(UnitTypeId.WARPGATE).ready.idle:
                abilities = await self.get_available_abilities(warpgate)
                if AbilityId.WARPGATEWARP_STALKER in abilities and self.can_afford(UnitTypeId.STALKER):
                    pylon = self.structures(UnitTypeId.PYLON).ready.random
                    placement = await self.find_placement(AbilityId.WARPGATEWARP_STALKER, pylon.position, placement_step=1)
                    if placement:
                        await self.do(warpgate.warp_in(UnitTypeId.STALKER, placement))

    async def build_proxy_pylon(self):
        if self.structures(UnitTypeId.PYLON).amount < 3 and self.can_afford(UnitTypeId.PYLON):
            proxy_pos = self.enemy_start_locations[0].towards(self.game_info.map_center, 20)
            if not self.structures(UnitTypeId.PYLON).closer_than(10, proxy_pos).exists:
                worker = self.select_build_worker(proxy_pos)
                if worker:
                    await self.do(worker.build(UnitTypeId.PYLON, proxy_pos))

    async def chrono_boost(self):
        for nexus in self.townhalls.ready:
            if nexus.energy >= 50:
                # Prioritize Cybercore, then Nexus
                cybercores = self.structures(UnitTypeId.CYBERNETICSCORE).ready
                if cybercores.exists and not cybercores.first.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                    await self.do(nexus(AbilityId.EFFECT_CHRONOBOOST, cybercores.first))
                elif not nexus.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                    await self.do(nexus(AbilityId.EFFECT_CHRONOBOOST, nexus))

    async def stalker_attack(self):
        stalkers = self.units(UnitTypeId.STALKER).idle
        if stalkers.exists:
            targets = self.enemy_units | self.enemy_structures
            if targets.exists:
                target = targets.closest_to(stalkers.first)
            else:
                target = self.enemy_start_locations[0]
            for stalker in stalkers:
                await self.do(stalker.attack(target))

def main():
    run_game(
        maps.get("(2)CatalystLE"),
        [Bot(Race.Protoss, ProtossBot()), Computer(Race.Protoss, Difficulty.Easy)],
        realtime=False,
    )

if __name__ == "__main__":
    main()
```