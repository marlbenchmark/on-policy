from onpolicy.envs.overcooked.script_agent.base import BaseScriptPeriod
from onpolicy.envs.overcooked.overcooked_ai_py.mdp.actions import Direction, Action
import onpolicy.envs.overcooked.script_agent.utils as utils
import numpy as np

class Pickup_Object(BaseScriptPeriod):
    def __init__(self, obj, terrain_type="XPODS", random_put=True, random_pos=True):
        """Pickup some object at specific terrains 
        obj: str
            "onion", "dish", "soup"
        terrain_type: str
            example "XPOD"
        random_put: bool
            if True, put the irrelevant obj at random position
        random_pos: bool
            if True, find a random obj, otherwise the closest one 
        """
        super().__init__(("random_" if random_pos else "") + "pickup_" + obj)

        self.__put_pos = None
        self.__obj_pos = None
        self.__random_pos = None

        self.random_put = random_put
        self.random_pos = random_pos
        self.target_obj = obj
        self.terrain_type = terrain_type
    
    def reset(self, mdp, state, player_idx):
        self.__put_pos = None
        self.__obj_pos = None
        self.__random_pos = None
    
    def step(self, mdp, state, player_idx):
        player = state.players[player_idx]

        if player.has_object() and player.get_object().name != self.target_obj:
             # not target obj, place in random position
            action, self.__put_pos = utils.interact(mdp, state, player_idx, pre_goal = self.__put_pos, random=self.random_put, terrain_type="XOPDS", obj=["can_put"])
            return action
        
        if not player.has_object():
            # find target obj
            action, self.__obj_pos = utils.interact(mdp, state, player_idx, pre_goal = self.__obj_pos, random=self.random_pos, terrain_type=self.terrain_type, obj=[self.target_obj])
            return action

        action, self.__random_pos = utils.random_move(mdp, state, player_idx, pre_goal = self.__random_pos)
        return action
    
    def done(self, mdp, state, player_idx):
        player = state.players[player_idx]
        return player.has_object() and player.get_object().name == self.target_obj

class Put_Object(BaseScriptPeriod):
    def __init__(self, terrain_type="XPODS", random_put=True, obj="can_put", pos_mask=None, move_mask=None):
        """Pickup some object at specific terrains 
        terrain_type: str
            example "XPODS"
        random_put: bool
            if True, put the irrelevant obj at random position
        """
        super().__init__(("random_" if random_put else "") + "put")

        self.__put_pos = None
        self.__random_pos = None
        self.__obj = obj

        self.random_put = random_put
        self.terrain_type = terrain_type
        self.pos_mask = pos_mask
        self.move_mask = move_mask
    
    def reset(self, mdp, state, player_idx):
        self.__put_pos = None
        self.__random_pos = None
    
    def step(self, mdp, state, player_idx):
        player = state.players[player_idx]

        if player.has_object():
             # not target obj, place in random position
            action, self.__put_pos = utils.interact(mdp, state, player_idx, pre_goal = self.__put_pos, random=self.random_put, terrain_type=self.terrain_type, obj=[self.__obj] if type(self.__obj) == str else self.__obj, pos_mask=self.pos_mask, move_mask=self.move_mask)
            return action

        action, self.__random_pos = utils.random_move(mdp, state, player_idx, pre_goal = self.__random_pos, move_mask=self.move_mask)
        return action
    
    def done(self, mdp, state, player_idx):
        player = state.players[player_idx]
        return not player.has_object()

class Pickup_Onion_and_Place_in_Pot(BaseScriptPeriod):
    def __init__(self, random_put=True, random_pot=True, random_onion=True):
        """
        random_put: bool
            if True, place the object to random position when the player starts with 
        random_pot: bool
            if True, find a random pot to place onion
        random_onion: bool
            if True, take a random onion
        """
        super().__init__(period_name="Pickup_Onion_and_Place_in_Pot")

        
        self.random_put = random_put
        self.random_pot = random_pot
        self.random_onion = random_onion

        self.__stage = 1
        self.__current_period = Pickup_Object(obj="onion", terrain_type="OX", random_put=self.random_put, random_pos=self.random_onion)
    
    def reset(self, mdp, state, player_idx):
        self.__stage = 1
        self.__current_period = Pickup_Object(obj="onion", terrain_type="OX", random_put=self.random_put, random_pos=self.random_onion)
    
    def step(self, mdp, state, player_idx):
        player = state.players[player_idx]

        if self.__stage == 1:
            if self.__current_period.done(mdp, state, player_idx):
                assert player.has_object() and player.get_object().name == "onion"
                self.__stage = 2
                self.__current_period = Put_Object(terrain_type="P", random_put=self.random_pot)
            else:
                return self.__current_period.step(mdp, state, player_idx)
        return self.__current_period.step(mdp, state, player_idx)

    def done(self, mdp, state, player_idx):
        player = state.players[player_idx]
        return self.__stage == 2 and not player.has_object()


class Pickup_Onion_and_Place_Random(BaseScriptPeriod):
    def __init__(self, random_put=True, random_onion=True):
        """
        random_put: bool
            if True, place the object to random position when the player starts with 
        random_onion: bool
            if True, take a random onion
        """
        super().__init__(period_name="Pickup_Onion_and_Place_Random")

        
        self.random_put = random_put
        self.random_onion = random_onion

        self.__stage = 1
        self.__current_period = Pickup_Object(obj="onion", terrain_type="OX", random_put=self.random_put, random_pos=self.random_onion)
    
    def reset(self, mdp, state, player_idx):
        self.__stage = 1
        self.__current_period = Pickup_Object(obj="onion", terrain_type="OX", random_put=self.random_put, random_pos=self.random_onion)
    
    def step(self, mdp, state, player_idx):
        player = state.players[player_idx]

        if self.__stage == 1:
            if self.__current_period.done(mdp, state, player_idx):
                assert player.has_object() and player.get_object().name == "onion"
                self.__stage = 2
                self.__current_period = Put_Object(terrain_type="XOPDS", random_put=True)
            else:
                return self.__current_period.step(mdp, state, player_idx)
        return self.__current_period.step(mdp, state, player_idx)

    def done(self, mdp, state, player_idx):
        player = state.players[player_idx]
        return self.__stage == 2 and not player.has_object()


class Put_Onion_Everywhere(BaseScriptPeriod):
    def __init__(self, random_put=True, random_onion=True):
        """
        random_put: bool
            if True, place the object to random position when the player starts with 
        random_onion: bool
            if True, take a random onion
        """
        super().__init__(period_name="Put_Onion_Everywhere")

        
        self.random_put = random_put
        self.random_onion = random_onion

        self.__stage = 1
        self.__current_period = Pickup_Object(obj="onion", terrain_type="O", random_put=self.random_put, random_pos=self.random_onion)
    
    def reset(self, mdp, state, player_idx):
        self.__stage = 1
        self.__current_period = Pickup_Object(obj="onion", terrain_type="O", random_put=self.random_put, random_pos=self.random_onion)
    
    def step(self, mdp, state, player_idx):
        player = state.players[player_idx]

        if self.__stage == 1:
            if self.__current_period.done(mdp, state, player_idx):
                assert player.has_object() and player.get_object().name == "onion"
                self.__stage = 2
                self.__current_period = Put_Object(terrain_type="X", random_put=True)
            else:
                return self.__current_period.step(mdp, state, player_idx)
        return self.__current_period.step(mdp, state, player_idx)

    def done(self, mdp, state, player_idx):
        player = state.players[player_idx]
        return self.__stage == 2 and not player.has_object()

class Pickup_Dish_and_Place_Random(BaseScriptPeriod):
    def __init__(self, random_put=True, random_dish=True):
        """
        random_put: bool
            if True, place the object to random position when the player starts with 
        random_dish: bool
            if True, take a random dish
        """
        super().__init__(period_name="Pickup_Dish_and_Place_Random")

        
        self.random_put = random_put
        self.random_dish = random_dish

        self.__stage = 1
        self.__current_period = Pickup_Object(obj="dish", terrain_type="XOPDS", random_put=self.random_put, random_pos=self.random_dish)
    
    def reset(self, mdp, state, player_idx):
        self.__stage = 1
        self.__current_period = Pickup_Object(obj="dish", terrain_type="XOPDS", random_put=self.random_put, random_pos=self.random_dish)
    
    def step(self, mdp, state, player_idx):
        player = state.players[player_idx]

        if self.__stage == 1:
            if self.__current_period.done(mdp, state, player_idx):
                self.__stage = 2
                self.__current_period = Put_Object(terrain_type="XOPDS", random_put=True)
            else:
                return self.__current_period.step(mdp, state, player_idx)
        return self.__current_period.step(mdp, state, player_idx)

    def done(self, mdp, state, player_idx):
        player = state.players[player_idx]
        return self.__stage == 2 and not player.has_object()

class Put_Dish_Everywhere(BaseScriptPeriod):
    def __init__(self, random_put=True, random_dish=True):
        """
        random_put: bool
            if True, place the object to random position when the player starts with 
        random_dish: bool
            if True, take a random dish
        """
        super().__init__(period_name="Put_Dish_Everywhere")

        
        self.random_put = random_put
        self.random_dish = random_dish

        self.__stage = 1
        self.__current_period = Pickup_Object(obj="dish", terrain_type="D", random_put=self.random_put, random_pos=self.random_dish)
    
    def reset(self, mdp, state, player_idx):
        self.__stage = 1
        self.__current_period = Pickup_Object(obj="dish", terrain_type="D", random_put=self.random_put, random_pos=self.random_dish)
    
    def step(self, mdp, state, player_idx):
        player = state.players[player_idx]

        if self.__stage == 1:
            if self.__current_period.done(mdp, state, player_idx):
                self.__stage = 2
                self.__current_period = Put_Object(terrain_type="X", random_put=True)
            else:
                return self.__current_period.step(mdp, state, player_idx)
        return self.__current_period.step(mdp, state, player_idx)

    def done(self, mdp, state, player_idx):
        player = state.players[player_idx]
        return self.__stage == 2 and not player.has_object()


class Pickup_Soup(BaseScriptPeriod):
    def __init__(self, random_dish=True, random_soup=True):
        super().__init__(period_name="Pickup_Soup")

        self.random_dish = random_dish
        self.random_soup = random_soup

        self.__stage = 1
        self.__current_period = Pickup_Object(obj="dish", terrain_type="XOPDS", random_put=True, random_pos=self.random_dish)
    
    def reset(self, mdp, state, player_idx):
        self.__stage = 1
        if utils.exists(mdp, state, player_idx, terrain_type="X", obj="soup"):
            #  if there are soups on table, take that on table
            self.__current_period = Pickup_Object(obj="soup", terrain_type="XP", random_put=True, random_pos=self.random_soup)
        else:
            self.__current_period = Pickup_Object(obj="dish", terrain_type="XOPDS", random_put=True, random_pos=self.random_dish)
    
    def step(self, mdp, state, player_idx):
        player = state.players[player_idx]

        if self.__stage == 1:
            if self.__current_period.done(mdp, state, player_idx):
                assert player.has_object() and player.get_object().name == "dish"
                self.__stage = 2
                # this is a quick hack to use put as pickup soup
                self.__current_period = Put_Object(terrain_type="P", random_put=self.random_soup, obj=["soup", "cooking_soup"])
            else:
                return self.__current_period.step(mdp, state, player_idx)
        return self.__current_period.step(mdp, state, player_idx)

    def done(self, mdp, state, player_idx):
        player = state.players[player_idx]
        return player.has_object() and player.get_object().name == "soup"
    
class Pickup_Soup_and_Deliver(BaseScriptPeriod):
    def __init__(self, random_dish=True, random_soup=True):
        super().__init__(period_name="Pickup_Soup_and_Deliver")

        self.random_dish = random_dish
        self.random_soup = random_soup

        self.__stage = 1
        self.__current_period = Pickup_Soup(random_dish=self.random_dish, random_soup=self.random_soup)
    
    def reset(self, mdp, state, player_idx):
        self.__stage = 1
        self.__current_period = Pickup_Soup(random_dish=self.random_dish, random_soup=self.random_soup)
    
    def step(self, mdp, state, player_idx):
        player = state.players[player_idx]

        if self.__stage == 1:
            if self.__current_period.done(mdp, state, player_idx):
                assert player.has_object() and player.get_object().name == "soup"
                self.__stage = 2
                # this is a quick hack to use put as deliver
                self.__current_period = Put_Object(terrain_type="S", random_put=False)
            else:
                return self.__current_period.step(mdp, state, player_idx)
        return self.__current_period.step(mdp, state, player_idx)

    def done(self, mdp, state, player_idx):
        player = state.players[player_idx]
        return self.__stage == 2 and self.__current_period.done(mdp, state, player_idx)

class Pickup_Soup_and_Place_Random(BaseScriptPeriod):
    def __init__(self, random_dish=True, random_soup=True):
        super().__init__(period_name="Pickup_Soup_and_Place_Random")

        self.random_dish = random_dish
        self.random_soup = random_soup

        self.__stage = 1
        self.__current_period = Pickup_Soup(random_dish=self.random_dish, random_soup=self.random_soup)
    
    def reset(self, mdp, state, player_idx):
        self.__stage = 1
        self.__current_period = Pickup_Soup(random_dish=self.random_dish, random_soup=self.random_soup)
    
    def step(self, mdp, state, player_idx):
        player = state.players[player_idx]

        if self.__stage == 1:
            if self.__current_period.done(mdp, state, player_idx):
                assert player.has_object() and player.get_object().name == "soup"
                self.__stage = 2
                # this is a quick hack to use put as deliver
                self.__current_period = Put_Object(terrain_type="XOPDS", random_put=True)
            else:
                return self.__current_period.step(mdp, state, player_idx)
        return self.__current_period.step(mdp, state, player_idx)

    def done(self, mdp, state, player_idx):
        player = state.players[player_idx]
        return self.__stage == 2 and self.__current_period.done(mdp, state, player_idx)

class Random3_Only_Onion_to_Middle(BaseScriptPeriod):
    # only works for random3
    def __init__(self):
        super().__init__(Random3_Only_Onion_to_Middle)

        self.__pos_mask = np.zeros((5, 8), dtype=np.int32)
        self.__pos_mask[2, 2:6] = 1
        self.__move_mask = np.zeros((5, 8), dtype=np.int32)
        self.__move_mask[2:, :] = 1

        self.__stage = 1
        self.__current_period = Pickup_Object(obj="onion", terrain_type="O", random_put=True, random_pos=True)

    def reset(self, mdp, state, player_idx):
        self.__stage = 1
        self.__current_period = Pickup_Object(obj="onion", terrain_type="O", random_put=True, random_pos=True)
    
    def step(self, mdp, state, player_idx):
        player = state.players[player_idx]

        if self.__stage == 1:
            if self.__current_period.done(mdp, state, player_idx):
                self.__stage = 2
                self.__current_period = Put_Object(terrain_type="X", random_put=True, obj="can_put", pos_mask=self.__pos_mask, move_mask=self.__move_mask)
            else:
                return self.__current_period.step(mdp, state, player_idx)
        return self.__current_period.step(mdp, state, player_idx)
    
    def done(self, mdp, state, player_idx):
        return self.__stage == 2 and self.__current_period.done(mdp, state, player_idx)

SCRIPT_PERIODS_CLASSES={
    "pickup_object": Pickup_Object,
    "put_object": Put_Object,
    "pickup_onion_and_place_in_pot": Pickup_Onion_and_Place_in_Pot,
    "pickup_onion_and_place_random": Pickup_Onion_and_Place_Random,
    "pickup_soup": Pickup_Soup,
    "pickup_soup_and_deliver": Pickup_Soup_and_Deliver,
    "pickup_soup_and_place_random": Pickup_Soup_and_Place_Random,
    "random3_only_onion_to_middle": Random3_Only_Onion_to_Middle,
    "pickup_dish_and_place_random": Pickup_Dish_and_Place_Random,
    "put_onion_everywhere": Put_Onion_Everywhere,
    "put_dish_everywhere": Put_Dish_Everywhere
}

