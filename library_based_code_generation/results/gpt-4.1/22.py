```python
import asyncio
import logging
from sc2 import maps, Race, Difficulty, run_game, position
from sc2.player import Bot, Computer
from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.upgrade_id import UpgradeId

logging.basicConfig(level=logging.INFO)

class ZergRushBot(BotAI):
    async def on_start(self):
        await self.client.game_step(2)
        self.rushed = False
        self.chat_sent = False
        self.inject_targets = set()
        self.logger = logging.getLogger("ZergRushBot")
        self.logger.info("Game started.")

    async def on_step(self, iteration):
        # Send chat message once
        if not self.chat_sent:
            await self.chat_send("GL HF! Prepare for the Zerg rush!")
            self.chat_sent = True

        await self.manage_resources()
        await self.inject_hatcheries()
        await self.train_units()
        await self.build_structures()
        await self.research_upgrades()
        await self.attack_enemy()
        await self.draw_creep_pixelmap()

    async def on_end(self, result):
        self.logger.info(f"Game ended with result: {result}")

    async def manage_resources(self):
        # Build more drones if less than 16 per base and less than 30 total
        if self.supply_workers < 30 and self.can_afford(UnitTypeId.DRONE):
            for hatch in self.townhalls.ready.idle:
                if self.can_afford(UnitTypeId.DRONE):
                    hatch.train(UnitTypeId.DRONE)

        # Build Extractors if we have spawning pool and less than 2 extractors
        if self.structures(UnitTypeId.SPAWNINGPOOL).ready.exists:
            for geyser in self.vespene_geyser.closer_than(10, self.townhalls.first):
                if self.can_afford(UnitTypeId.EXTRACTOR) and not self.gas_buildings.closer_than(1, geyser).exists:
                    drone = self.workers.closer_than(10, geyser).random_or(None)
                    if drone:
                        drone.build_gas(geyser)

    async def inject_hatcheries(self):
        for queen in self.units(UnitTypeId.QUEEN).idle:
            abilities = await self.get_available_abilities(queen)
            if AbilityId.EFFECT_INJECTLARVA in abilities:
                hatch = self.townhalls.ready.random_or(None)
                if hatch:
                    queen(AbilityId.EFFECT_INJECTLARVA, hatch)

    async def train_units(self):
        # Train Queens if we have Spawning Pool and less than 2 queens
        if self.structures(UnitTypeId.SPAWNINGPOOL).ready.exists:
            if self.units(UnitTypeId.QUEEN).amount < 2:
                for hatch in self.townhalls.ready.idle:
                    if self.can_afford(UnitTypeId.QUEEN):
                        hatch.train(UnitTypeId.QUEEN)

        # Train Zerglings if we have Spawning Pool
        if self.structures(UnitTypeId.SPAWNINGPOOL).ready.exists:
            for hatch in self.townhalls.ready:
                if self.can_afford(UnitTypeId.ZERGLING):
                    hatch.train(UnitTypeId.ZERGLING)

        # Morph to Banelings if we have Baneling Nest
        if self.structures(UnitTypeId.BANELINGNEST).ready.exists:
            for zergling in self.units(UnitTypeId.ZERGLING).idle:
                if self.can_afford(UnitTypeId.BANELING):
                    zergling(AbilityId.MORPHZERGLINGTOBANELING)

    async def build_structures(self):
        # Build Overlords if supply is low
        if self.supply_left < 4 and self.can_afford(UnitTypeId.OVERLORD):
            for hatch in self.townhalls.ready.idle:
                hatch.train(UnitTypeId.OVERLORD)

        # Build Spawning Pool if not exists
        if not self.structures(UnitTypeId.SPAWNINGPOOL).exists and self.can_afford(UnitTypeId.SPAWNINGPOOL):
            await self.build(UnitTypeId.SPAWNINGPOOL, near=self.townhalls.first.position.towards(self.game_info.map_center, 5))

        # Build Baneling Nest if not exists and we have Spawning Pool
        if self.structures(UnitTypeId.SPAWNINGPOOL).ready.exists and not self.structures(UnitTypeId.BANELINGNEST).exists:
            if self.can_afford(UnitTypeId.BANELINGNEST):
                await self.build(UnitTypeId.BANELINGNEST, near=self.townhalls.first.position.towards(self.game_info.map_center, 7))

        # Expand if we have enough minerals and less than 2 bases
        if self.townhalls.amount < 2 and self.can_afford(UnitTypeId.HATCHERY):
            await self.expand_now()

    async def research_upgrades(self):
        # Research Zergling Speed
        if self.structures(UnitTypeId.SPAWNINGPOOL).ready.exists:
            pool = self.structures(UnitTypeId.SPAWNINGPOOL).ready.first
            if self.can_afford(UpgradeId.ZERGLINGMOVEMENTSPEED) and not self.already_pending_upgrade(UpgradeId.ZERGLINGMOVEMENTSPEED):
                abilities = await self.get_available_abilities(pool)
                if AbilityId.RESEARCH_ZERGLINGMETABOLICBOOST in abilities:
                    pool(AbilityId.RESEARCH_ZERGLINGMETABOLICBOOST)

        # Research Baneling Speed
        if self.structures(UnitTypeId.BANELINGNEST).ready.exists:
            nest = self.structures(UnitTypeId.BANELINGNEST).ready.first
            if self.can_afford(UpgradeId.BANELINGSPEED) and not self.already_pending_upgrade(UpgradeId.BANELINGSPEED):
                abilities = await self.get_available_abilities(nest)
                if AbilityId.RESEARCH_CENTRIFUGALHOOKS in abilities:
                    nest(AbilityId.RESEARCH_CENTRIFUGALHOOKS)

    async def attack_enemy(self):
        # Attack with all Zerglings and Banelings if we have more than 20 Zerglings
        if self.units(UnitTypeId.ZERGLING).amount > 20 or self.units(UnitTypeId.BANELING).amount > 5:
            target = self.enemy_structures.random_or(self.enemy_start_locations[0])
            for unit in self.units(UnitTypeId.ZERGLING).idle | self.units(UnitTypeId.BANELING).idle:
                unit.attack(target.position)

    async def draw_creep_pixelmap(self):
        # Draw creep as purple pixels for debugging
        for y in range(self.game_info.map_size[1]):
            for x in range(self.game_info.map_size[0]):
                pos = position.Point2((x + 0.5, y + 0.5))
                if self.state.creep[x, y]:
                    await self._client.debug_text_world("•", pos, color=(200, 0, 200), size=8)

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(
        run_game(
            maps.get("AcropolisLE"),
            [
                Bot(Race.Zerg, ZergRushBot()),
                Computer(Race.Terran, Difficulty.Medium)
            ],
            realtime=False,
            save_replay_as="zerg_rush_bot.SC2Replay"
        )
    )
```