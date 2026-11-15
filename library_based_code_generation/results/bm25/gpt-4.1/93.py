```python
import random
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Race, Difficulty
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.player import Bot, Computer
from sc2.main import run_game

class ThreeBaseThreeStargateBot(BotAI):
    async def on_step(self, iteration):
        await self.distribute_workers()
        await self.build_pylons()
        await self.train_probes()
        await self.expand()
        await self.build_gas()
        await self.build_gateway_and_cybercore()
        await self.build_stargates()
        await self.train_voidrays()
        await self.chrono_boost()
        await self.attack_with_voidrays()
        await self.all_in_if_no_nexus()

    async def build_pylons(self):
        if self.supply_left < 5 and self.already_pending(UnitTypeId.PYLON) == 0:
            nexuses = self.townhalls.ready
            if nexuses.exists:
                pos = nexuses.random.position.towards(self.game_info.map_center, 5)
                await self.build(UnitTypeId.PYLON, near=pos)

    async def train_probes(self):
        for nexus in self.townhalls.ready:
            if nexus.is_idle and self.can_afford(UnitTypeId.PROBE):
                if self.workers.amount < self.townhalls.amount * 22:
                    await self.do(nexus.train(UnitTypeId.PROBE))

    async def expand(self):
        if self.townhalls.amount + self.already_pending(UnitTypeId.NEXUS) < 3 and self.can_afford(UnitTypeId.NEXUS):
            await self.expand_now()

    async def build_gas(self):
        for nexus in self.townhalls.ready:
            vgs = self.vespene_geyser.closer_than(15.0, nexus)
            for vg in vgs:
                if not self.gas_buildings.closer_than(1.0, vg).exists and self.can_afford(UnitTypeId.ASSIMILATOR):
                    worker = self.select_build_worker(vg.position)
                    if worker:
                        await self.do(worker.build(UnitTypeId.ASSIMILATOR, vg))
                        break

    async def build_gateway_and_cybercore(self):
        if self.structures(UnitTypeId.PYLON).ready.exists:
            pylon = self.structures(UnitTypeId.PYLON).ready.random
            if self.structures(UnitTypeId.GATEWAY).amount + self.already_pending(UnitTypeId.GATEWAY) < 1:
                if self.can_afford(UnitTypeId.GATEWAY):
                    await self.build(UnitTypeId.GATEWAY, near=pylon)
            elif self.structures(UnitTypeId.GATEWAY).ready.exists and self.structures(UnitTypeId.CYBERNETICSCORE).amount + self.already_pending(UnitTypeId.CYBERNETICSCORE) < 1:
                if self.can_afford(UnitTypeId.CYBERNETICSCORE):
                    await self.build(UnitTypeId.CYBERNETICSCORE, near=pylon)

    async def build_stargates(self):
        if self.townhalls.ready.amount >= 3 and self.structures(UnitTypeId.CYBERNETICSCORE).ready.exists:
            stargates_total = self.structures(UnitTypeId.STARGATE).amount + self.already_pending(UnitTypeId.STARGATE)
            if stargates_total < 3 and self.can_afford(UnitTypeId.STARGATE):
                pylon = self.structures(UnitTypeId.PYLON).ready.random
                await self.build(UnitTypeId.STARGATE, near=pylon)

    async def train_voidrays(self):
        if self.townhalls.ready.amount >= 3:
            for stargate in self.structures(UnitTypeId.STARGATE).ready.idle:
                if self.can_afford(UnitTypeId.VOIDRAY):
                    await self.do(stargate.train(UnitTypeId.VOIDRAY))

    async def chrono_boost(self):
        for nexus in self.townhalls.ready:
            abilities = await self.get_available_abilities(nexus)
            if AbilityId.EFFECT_CHRONOBOOSTENERGYCOST in abilities:
                # Prioritize stargates, then cybercore, then gateway, then nexus
                targets = self.structures(UnitTypeId.STARGATE).ready | self.structures(UnitTypeId.CYBERNETICSCORE).ready | self.structures(UnitTypeId.GATEWAY).ready
                if targets.exists:
                    target = targets.random
                    await self.do(nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, target))
                else:
                    await self.do(nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, nexus))

    async def attack_with_voidrays(self):
        voidrays = self.units(UnitTypeId.VOIDRAY).ready
        if voidrays.amount > 5:
            target = self.enemy_start_locations[0]
            for vr in voidrays.idle:
                await self.do(vr.attack(target))

    async def all_in_if_no_nexus(self):
        if not self.townhalls.exists:
            for worker in self.workers.idle | self.workers.gathering:
                await self.do(worker.attack(self.enemy_start_locations[0]))

def main():
    run_game(
        maps.get("(2)CatalystLE"),
        [
            Bot(Race.Protoss, ThreeBaseThreeStargateBot()),
            Computer(Race.Protoss, Difficulty.Easy)
        ],
        realtime=False,
    )

if __name__ == "__main__":
    main()
```