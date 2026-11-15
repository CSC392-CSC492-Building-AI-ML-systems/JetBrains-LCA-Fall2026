```python
import random
import asyncio
from sc2 import maps, run_game, Race, Difficulty, position
from sc2.player import Bot, Computer
from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.upgrade_id import UpgradeId
from sc2.ids.buff_id import BuffId

class ZergRushBot(BotAI):
    async def on_start(self):
        await self.client.chat_send("GL HF! Zerg Rush incoming!", team_only=False)
        await self.client.game_step(2)
        self.rushed = False

    async def on_step(self, iteration):
        # Send chat message at the start
        if iteration == 2:
            await self.client.chat_send("You better be ready!", team_only=False)

        # Draw creep pixelmap for debugging
        await self.draw_creep_pixelmap()

        # Inject hatcheries with larva
        await self.inject_hatcheries()

        # Manage resources: build overlords, extractors, drones
        await self.manage_resources()

        # Research metabolic boost (zergling speed)
        await self.research_upgrades()

        # Train zerglings
        await self.train_units()

        # Build spawning pool if not exists
        await self.build_spawning_pool()

        # Build more hatcheries if enough minerals
        await self.expand()

        # Attack enemy structures with all zerglings
        await self.rush_attack()

    async def on_end(self, result):
        print(f"Game ended with result: {result}")

    async def draw_creep_pixelmap(self):
        for y in range(self.game_info.map_size[1]):
            for x in range(self.game_info.map_size[0]):
                pos = position.Point2((x, y))
                if self.state.creep[x, y]:
                    self._client.debug_text_world("C", pos, size=8, color=(255, 0, 255))

    async def inject_hatcheries(self):
        for queen in self.units(UnitTypeId.QUEEN).idle:
            abilities = await self.get_available_abilities(queen)
            if AbilityId.EFFECT_INJECTLARVA in abilities:
                hatch = self.townhalls.ready.random_or(None)
                if hatch:
                    queen(AbilityId.EFFECT_INJECTLARVA, hatch)

    async def manage_resources(self):
        # Build overlords if supply low
        if self.supply_left < 3 and self.already_pending(UnitTypeId.OVERLORD) == 0:
            for larva in self.larva:
                larva.train(UnitTypeId.OVERLORD)
                break

        # Build extractors if spawning pool is ready and less than 2 extractors
        if self.structures(UnitTypeId.SPAWNINGPOOL).ready.exists:
            if self.gas_buildings.amount < 2:
                for drone in self.workers.idle:
                    vgs = self.vespene_geyser.closer_than(10, self.townhalls.first)
                    for vg in vgs:
                        if not self.gas_buildings.closer_than(1, vg).exists:
                            drone.build_gas(vg)
                            return

        # Build more drones if less than 16 per base
        if self.workers.amount < self.townhalls.amount * 16 and self.can_afford(UnitTypeId.DRONE):
            for larva in self.larva:
                larva.train(UnitTypeId.DRONE)
                break

    async def research_upgrades(self):
        if self.structures(UnitTypeId.SPAWNINGPOOL).ready.exists:
            pool = self.structures(UnitTypeId.SPAWNINGPOOL).ready.first
            if self.already_pending_upgrade(UpgradeId.ZERGLINGMOVEMENTSPEED) == 0:
                abilities = await self.get_available_abilities(pool)
                if AbilityId.RESEARCH_ZERGLINGMETABOLICBOOST in abilities and self.can_afford(UpgradeId.ZERGLINGMOVEMENTSPEED):
                    pool(AbilityId.RESEARCH_ZERGLINGMETABOLICBOOST)

    async def train_units(self):
        # Train zerglings if spawning pool is ready
        if self.structures(UnitTypeId.SPAWNINGPOOL).ready.exists:
            for larva in self.larva:
                if self.can_afford(UnitTypeId.ZERGLING):
                    larva.train(UnitTypeId.ZERGLING)

        # Train queens if less than 2 per hatchery
        for hatch in self.townhalls.ready:
            if self.units(UnitTypeId.QUEEN).closer_than(10, hatch).amount < 2:
                if self.can_afford(UnitTypeId.QUEEN) and hatch.is_idle:
                    hatch.train(UnitTypeId.QUEEN)

    async def build_spawning_pool(self):
        if not self.structures(UnitTypeId.SPAWNINGPOOL).exists and self.can_afford(UnitTypeId.SPAWNINGPOOL):
            pos = await self.get_next_expansion()
            await self.build(UnitTypeId.SPAWNINGPOOL, near=pos)

    async def expand(self):
        if self.townhalls.amount < 2 and self.can_afford(UnitTypeId.HATCHERY):
            pos = await self.get_next_expansion()
            await self.build(UnitTypeId.HATCHERY, near=pos)

    async def rush_attack(self):
        if self.units(UnitTypeId.ZERGLING).amount > 20 or self.rushed:
            self.rushed = True
            targets = self.enemy_structures
            if targets.exists:
                target = targets.random.position
            else:
                target = self.enemy_start_locations[0]
            for zergling in self.units(UnitTypeId.ZERGLING).idle:
                zergling.attack(target)

if __name__ == "__main__":
    run_game(
        maps.get("AbyssalReefLE"),
        [
            Bot(Race.Zerg, ZergRushBot()),
            Computer(Race.Terran, Difficulty.Medium)
        ],
        realtime=False,
        save_replay_as="zerg_rush_bot.SC2Replay"
    )
```