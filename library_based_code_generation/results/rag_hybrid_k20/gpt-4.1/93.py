```python
import random
from sc2 import maps, run_game, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import *
from sc2.bot_ai import BotAI

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
        await self.attack()

    async def build_pylons(self):
        if self.supply_left < 5 and not self.already_pending(PYLON):
            nexuses = self.townhalls.ready
            if nexuses.exists:
                pos = nexuses.random.position.towards(self.game_info.map_center, random.randrange(5, 8))
                await self.build(PYLON, near=pos)

    async def train_probes(self):
        for nexus in self.townhalls.ready.idle:
            if self.can_afford(PROBE) and nexus.assigned_harvesters < nexus.ideal_harvesters:
                nexus.train(PROBE)

    async def expand(self):
        if self.townhalls.amount < 3 and self.can_afford(NEXUS) and not self.already_pending(NEXUS):
            await self.expand_now()

    async def build_gas(self):
        for nexus in self.townhalls.ready:
            vgs = self.vespene_geyser.closer_than(15.0, nexus)
            for vg in vgs:
                if not self.gas_buildings.closer_than(1.0, vg).exists and self.can_afford(ASSIMILATOR):
                    worker = self.select_build_worker(vg.position)
                    if worker:
                        worker.build(ASSIMILATOR, vg)
                        break

    async def build_gateway_and_cybercore(self):
        if self.structures(PYLON).ready.exists:
            pylon = self.structures(PYLON).ready.random
            if self.structures(GATEWAY).amount < 1 and self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
                await self.build(GATEWAY, near=pylon)
            elif self.structures(GATEWAY).ready.exists and self.structures(CYBERNETICSCORE).amount < 1 and self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                await self.build(CYBERNETICSCORE, near=pylon)

    async def build_stargates(self):
        if self.structures(CYBERNETICSCORE).ready.exists and self.townhalls.amount >= 3:
            if self.structures(STARGATE).amount + self.already_pending(STARGATE) < 3 and self.can_afford(STARGATE):
                pylon = self.structures(PYLON).ready.random
                await self.build(STARGATE, near=pylon)

    async def train_voidrays(self):
        if self.structures(STARGATE).ready.exists and self.townhalls.amount >= 3:
            for stargate in self.structures(STARGATE).ready.idle:
                if self.can_afford(VOIDRAY):
                    stargate.train(VOIDRAY)

    async def chrono_boost(self):
        for nexus in self.townhalls.ready:
            abilities = await self.get_available_abilities(nexus)
            if AbilityId.EFFECT_CHRONOBOOSTENERGYCOST in abilities:
                targets = self.structures(STARGATE).ready | self.structures(CYBERNETICSCORE).ready | self.structures(GATEWAY).ready
                if targets.exists:
                    target = targets.random
                    nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, target)

    async def attack(self):
        voidrays = self.units(VOIDRAY).ready
        if voidrays.amount > 5:
            for vr in voidrays.idle:
                vr.attack(self.enemy_start_locations[0])
        if self.townhalls.amount == 0 and self.units(PROBE).exists:
            for worker in self.units(PROBE).idle:
                worker.attack(self.enemy_start_locations[0])

run_game(
    maps.get("(2)CatalystLE"),
    [Bot(Race.Protoss, ThreeBaseThreeStargateBot()), Computer(Race.Protoss, Difficulty.Easy)],
    realtime=False,
)
```