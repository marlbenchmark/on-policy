import numpy as np
import random as rd
from onpolicy.envs.overcooked.overcooked_ai_py.mdp.actions import Direction, Action

def compute_valid_map(mdp, state, player_idx, terrain_type, obj_lst):
    player = state.players[player_idx]
    valid_map = np.zeros((len(mdp.terrain_mtx), len(mdp.terrain_mtx[0])), dtype=np.int32)
    for terrain in terrain_type:
        positions = list(mdp.terrain_pos_dict[terrain])
        for pos in positions:
            x, y = pos
            for obj in obj_lst:
                if obj == "onion":
                    valid_map[y, x] += (terrain == "O") or (terrain == "X" and state.has_object(pos) and state.get_object(pos).name == "onion")
                elif obj == "dish":
                    valid_map[y, x] += (terrain == "D") or (terrain == "X" and state.has_object(pos) and state.get_object(pos).name == "dish")
                elif obj == "cooking_soup":
                    if terrain == "P" and state.has_object(pos) and state.get_object(pos).name == "soup":
                        obj = state.get_object(pos)
                        soup_type, num_items, cook_time = obj.state
                        if num_items == mdp.num_items_for_soup and cook_time < mdp.soup_cooking_time:
                            valid_map[y, x] += 1
                elif obj == "soup":
                    if terrain == "P":
                        if state.has_object(pos) and state.get_object(pos).name == "soup":
                            obj = state.get_object(pos)
                            soup_type, num_items, cook_time = obj.state
                            if num_items == mdp.num_items_for_soup and cook_time >= mdp.soup_cooking_time:
                                valid_map[y, x] += 1
                    elif terrain == "X":
                        valid_map[y, x] += (state.has_object(pos) and state.get_object(pos).name == "soup")
                elif obj == "empty":
                    valid_map[y, x] += (terrain in "XP" and not state.has_object(pos))
                elif obj == "unfull_soup":
                    if terrain == "P":
                        if state.has_object(pos) and state.get_object(pos).name == "soup":
                            obj = state.get_object(pos)
                            soup_type, num_items, cook_time = obj.state
                            if num_items < mdp.num_items_for_soup:
                                valid_map[y, x] += 1
                elif obj == "can_put":
                    if terrain == "X":
                        valid_map[y, x] += (not state.has_object(pos) and player.has_object())
                    elif terrain == "P":
                        if player.has_object() and player.get_object().name == "onion":
                            if not state.has_object(pos):
                                valid_map[y, x] += 1
                            else:
                                obj = state.get_object(pos)
                                assert obj.name == "soup"
                                soup_type, num_items, cook_time = obj.state
                                valid_map[y, x] += (num_items < mdp.num_items_for_soup)
                    elif terrain == "S":
                        if player.has_object() and player.get_object().name == "soup":
                            valid_map[y, x] += 1
                else:
                    raise NotImplementedError(f"Object {obj} not implemented.")
            valid_map[y, x] = min(1, valid_map[y, x])
    return valid_map

def bfs(mdp, state, player_idx, move_mask=None):
    player = state.players[player_idx]
    other_player = state.players[1 - player_idx]
    dist = -np.ones((len(mdp.terrain_mtx), len(mdp.terrain_mtx[0])), dtype=np.int32)
    path = [[[None, None] for x in range(dist.shape[1])] for y in range(dist.shape[0])]

    x, y = player.position
    o_x, o_y = other_player.position
    dist[y, x] = 0

    q = [(x, y)]
    Head = 0
    Tail = 1
    while Head < Tail:
        x, y = pos = q[Head]
        Head += 1
        for d in Direction.ALL_DIRECTIONS:
            x1, y1 = adj_pos = Action.move_in_direction(pos, d)
            if y1 >= 0 and y1 < dist.shape[0] and x1 >= 0 and x1 < dist.shape[1] and dist[y1, x1] == -1 and (x1 != o_x or y1 != o_y) and (move_mask is None or move_mask[y1, x1] == 1):
                dist[y1, x1] = dist[y, x] + 1
                path[y1][x1] = (pos, d)
                if mdp.terrain_mtx[y1][x1] == " ":
                    q.append((x1, y1))
                    Tail += 1
    return dist, path
    

def interact(mdp, state, player_idx, pre_goal, random, terrain_type, obj, pos_mask=None, move_mask=None):
    """
    obj: List[str]
        "onion", "cooking_soup", "dish", "soup"(ready) or "empty", "can_put", "can_interact"
    """
    player = state.players[player_idx]
    pos, o = player.position, player.orientation
    i_pos = Action.move_in_direction(pos, o)

    valid_map = compute_valid_map(mdp, state, player_idx, terrain_type, obj)

    if pos_mask is not None:
        valid_map = valid_map * pos_mask
    
    dist, path = bfs(mdp, state, player_idx, move_mask=move_mask)

    # print("requirement: ", terrain_type, obj)
    # print("valid_map\n", valid_map)
    # print("dist\n", dist)
    # print("path")
    # for row in path:
    #     print(row)
    # print("pos", pos)
    # print("other_pos", state.players[1 - player_idx].position)
    # print("pre_goal", pre_goal)

    goal = None
    if pre_goal is not None:
        # assert mdp.get_terrain_type_at_pos(pre_goal) in terrain_type
        if valid_map[pre_goal[1], pre_goal[0]] and dist[pre_goal[1], pre_goal[0]] != -1:
            goal = pre_goal

    if goal is None:
        candidates = []
        for y in range(valid_map.shape[0]):
            for x in range(valid_map.shape[1]):
                if valid_map[y, x] and dist[y, x] != -1:
                    candidates.append((x, y))
        if len(candidates) == 0:
            candidates = mdp.get_valid_player_positions()
        candidates = [(x, y) for x, y in candidates if dist[y, x] != -1 and (move_mask is None or move_mask[y, x] == 1)]
        if len(candidates) == 0:
            candidates = mdp.get_valid_player_positions()
        candidates = [(x, y) for x, y in candidates if dist[y, x] != -1]
        if random:
            goal = rd.choice(candidates)
        else:
            for x, y in candidates:
                if goal is None or dist[y, x] < dist[goal[1], goal[0]]:
                    goal = (x, y)
    
    # print("goal", goal)

    if i_pos[1] == goal[1] and i_pos[0] == goal[0] and mdp.get_terrain_type_at_pos(goal) in terrain_type and valid_map[goal[1], goal[0]]:
        # print("INTERACT at goal")
        return Action.INTERACT, goal
    
    x, y = goal
    action = rd.choice(Direction.ALL_DIRECTIONS)
    history = []
    while x!=pos[0] or y!=pos[1]:
        history.append((x, y, action))
        try:
            (x, y), action = path[y][x]
        except TypeError as e:
            print(history)
            raise e
    history.append((x, y, action))
    # print("history", history)
    return action, goal

def random_move(mdp, state, player_idx, pre_goal, move_mask=None):
    player = state.players[player_idx]
    pos, o = player.position, player.orientation
    i_pos = Action.move_in_direction(pos, o)
    
    dist, path = bfs(mdp, state, player_idx, move_mask=move_mask)

    goal = None
    if pre_goal is not None:
        # assert mdp.get_terrain_type_at_pos(pre_goal) in terrain_type
        if dist[pre_goal[1], pre_goal[0]] != -1:
            goal = pre_goal

    if goal is None:
        candidates = mdp.get_valid_player_positions()
        candidates = [(x, y) for x, y in candidates if dist[y, x] != -1 and (move_mask is None or move_mask[y, x] == 1)]
        goal = rd.choice(candidates)
    
    x, y = goal
    action = rd.choice(Direction.ALL_DIRECTIONS)
    history = []
    while x!=pos[0] or y!=pos[1]:
        history.append((x, y, action))
        try:
            (x, y), action = path[y][x]
        except TypeError as e:
            print(history)
            raise e
        if len(history) > 10:
            print(history)
            raise RuntimeError
    history.append((x, y, action))
    return action, goal


def exists(mdp, state, player_idx, terrain_type, obj):
    
    player = state.players[player_idx]
    pos, o = player.position, player.orientation
    i_pos = Action.move_in_direction(pos, o)

    valid_map = compute_valid_map(mdp, state, player_idx, terrain_type, obj)
    
    dist, path = bfs(mdp, state, player_idx)

    # print("valid_map\n", valid_map)
    # print("dist\n", dist)

    candidates = []
    for y in range(valid_map.shape[0]):
        for x in range(valid_map.shape[1]):
            if valid_map[y, x] and dist[y, x] != -1:
                candidates.append((x, y))
    
    return len(candidates) > 0