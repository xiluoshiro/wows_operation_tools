"""从三格血开始规划整场行动；只读取输入，不修改单位或敌人文件。"""
import argparse
import itertools
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

ATTRS = ('firepower', 'armor', 'maneuverability')
ABILITIES = {'smoke_screen', 'large_caliber', 'anti_aircraft_fire',
             'rapid_fire', 'attack_aircraft', 'detectability'}
OFFSET = {'left': -1, 'right': 1, 'front_left': -1, 'front': 0, 'front_right': 1}
LABEL = dict(zip(ATTRS, ('火力', '存活性', '机动性')),
             smoke_screen='烟幕', large_caliber='大口径', anti_aircraft_fire='防空火力',
             rapid_fire='速射', attack_aircraft='攻击机', detectability='被侦察性')
REGULAR_OPTIONS = (*ATTRS, 'double_buffs')
SPECIAL_OPTIONS = ('stats_40_disable_effects', 'double_debuffs', 'ignore_weakness', 'health_plus_2')
OPTION_LABEL = dict(LABEL, double_buffs='增益效果 ×2',
                    stats_40_disable_effects='能力失效、三维 +40%', double_debuffs='减益效果 ×2',
                    ignore_weakness='忽略敌方弱点', health_plus_2='生命 +2')


def read_inputs(units_path, enemies_path):
    units = json.loads(units_path.read_text(encoding='utf-8-sig'))
    enemies = json.loads(enemies_path.read_text(encoding='utf-8-sig'))
    if not isinstance(units, list) or not 4 <= len(units) <= 63:
        raise ValueError('单位数量须为 4–63 张（位掩码上限），每关使用四张。')
    ids = [u['id'] for u in units]
    if len(set(ids)) != len(ids):
        raise ValueError('单位 ID 重复。')
    for u in units:
        if u['type'] not in {'aircraft', 'destroyer', 'cruiser', 'battleship'}:
            raise ValueError(f'{u["id"]} 的单位类型无效。')
        for a in ATTRS:
            if type(u[a]) is not int or u[a] < 0:
                raise ValueError(f'{u["id"]}.{a} 须为非负整数满血属性。')
        for field, names, directions in (
            ('buffs', set(ATTRS), {'left', 'right'}),
            ('debuffs', ABILITIES, {'front_left', 'front', 'front_right'}),
        ):
            for effect in u[field]:
                if effect['name'] not in names or effect['direction'] not in directions:
                    raise ValueError(f'{u["id"]}.{field} 的能力或方向无效。')
    def number(key, prefix):
        match = re.fullmatch(prefix + r'([1-9]\d*)', key)
        if not match:
            raise ValueError(f'键名应为 {prefix} 加正整数：{key}')
        return int(match[1])
    stages = []
    for stage in sorted(enemies, key=lambda k: number(k, 'stage')):
        levels = []
        for name in sorted(enemies[stage], key=lambda k: number(k, 'level')):
            level = enemies[stage][name]
            if type(level.get('disable_buffs', False)) is not bool:
                raise ValueError(f'{stage}/{name} 的 disable_buffs 须为布尔值。')
            reduction = level.get('outgoing_debuff_effect_reduction_percent', 0)
            if type(reduction) is not int or not 0 <= reduction <= 100:
                raise ValueError(f'{stage}/{name} 的对敌减益效果降低幅度须为 0–100 的整数百分数。')
            reinforcement = level.get('reinforcement', {})
            if not isinstance(reinforcement, dict):
                raise ValueError(f'{stage}/{name} 的 reinforcement 须为对象。')
            if reinforcement:
                if type(reinforcement.get('percent')) is not int or not 1 <= reinforcement['percent'] <= 100:
                    raise ValueError(f'{stage}/{name} 的补强幅度须为 1–100 的整数百分数。')
                if type(reinforcement.get('max_units')) is not int or not 1 <= reinforcement['max_units'] <= 4:
                    raise ValueError(f'{stage}/{name} 每关补强单位数须为 1–4。')
                attributes = reinforcement.get('attributes', [])
                choices = reinforcement.get('options', attributes)
                if not choices or len(set(choices)) != len(choices) or not set(choices) <= set(REGULAR_OPTIONS):
                    raise ValueError(f'{stage}/{name} 的补强可选属性无效。')
                if not set(attributes) <= set(ATTRS):
                    raise ValueError(f'{stage}/{name} 的单属性补强标识无效。')
                positions = reinforcement.get('positions', [1, 2, 3, 4])
                if (not positions or any(type(p) is not int or not 1 <= p <= 4 for p in positions)
                        or len(set(positions)) != len(positions)):
                    raise ValueError(f'{stage}/{name} 的补强位置须为不重复的 1–4。')
                if reinforcement['max_units'] > len(positions):
                    raise ValueError(f'{stage}/{name} 的补强单位数不能超过允许补强的位置数。')
            special = level.get('special_reinforcement', {})
            if not isinstance(special, dict):
                raise ValueError(f'{stage}/{name} 的 special_reinforcement 须为对象。')
            if special:
                options = special.get('options', [])
                if not options or len(set(options)) != len(options) or not set(options) <= set(SPECIAL_OPTIONS):
                    raise ValueError(f'{stage}/{name} 的特殊补强选项无效。')
                positions = special.get('positions', [1, 2, 3, 4])
                if (not positions or any(type(p) is not int or not 1 <= p <= 4 for p in positions)
                        or len(set(positions)) != len(positions)):
                    raise ValueError(f'{stage}/{name} 的特殊补强位置须为不重复的 1–4。')
                if type(special.get('max_units')) is not int or not 1 <= special['max_units'] <= len(positions):
                    raise ValueError(f'{stage}/{name} 的特殊补强数量无效。')
            foes = sorted(level['enemies'], key=lambda e: e['position'])
            if [e['position'] for e in foes] != [1, 2, 3, 4]:
                raise ValueError(f'{stage}/{name} 须填写位置 1–4 的四个敌人。')
            for e in foes:
                if type(e['power']) is not int or e['power'] < 0:
                    raise ValueError(f'{stage}/{name} 敌方战力须为非负整数。')
                if type(e.get('buff_percent', 0)) is not int or e.get('buff_percent', 0) < 0:
                    raise ValueError(f'{stage}/{name} 敌方 buff_percent 须为非负整数百分数。')
                if not set(e['weaknesses']) <= ABILITIES:
                    raise ValueError(f'{stage}/{name} 存在未知弱点标识。')
            debuffs = level['debuff']
            percents = level.get('debuff_percent', {})
            if not set(debuffs) <= set(ATTRS) or set(percents) != set(debuffs):
                raise ValueError(f'{stage}/{name} 的属性减益或减益幅度缺失/不匹配。')
            if any(type(p) is not int or not 0 <= p <= 100 for p in percents.values()):
                raise ValueError(f'{stage}/{name} 的减益百分数须为 0–100 的整数。')
            if not set(level['forbidden']) <= {'aircraft', 'destroyer', 'cruiser', 'battleship'}:
                raise ValueError(f'{stage}/{name} 存在未知禁用类型。')
            levels.append((stage, name, {**level, 'enemies': foes}))
        if not levels:
            raise ValueError(f'{stage} 没有据点。')
        stages.append(levels)
    if not stages:
        raise ValueError('敌人文件为空。')
    return units, stages


def make_table(units, level, formations, order):
    """每个接收单位的属性只依赖自身血量；来源单位血量不改变效果比例。"""
    if order == 'both':
        before, before_valid, enemy = make_table(units, level, formations, 'before-buffs')
        if not level['debuff']:
            return before, before_valid, enemy
        after, after_valid, _ = make_table(units, level, formations, 'after-buffs')
        return np.minimum(before, after), before_valid & after_valid, enemy
    greens = np.zeros((len(formations), 4, 3), dtype=bool)
    reds = np.zeros((len(formations), 4), dtype=bool)
    valid = np.ones(len(formations), dtype=bool)
    for src in range(4):
        for i, u in enumerate(units):
            selected = formations[:, src] == i
            if u['type'] in level['forbidden']:
                valid[selected] = False
            for field in ('buffs', 'debuffs'):
                if field == 'buffs' and level.get('disable_buffs', False):
                    continue
                for effect in u[field]:
                    dst = src + OFFSET[effect['direction']]
                    if not 0 <= dst < 4:
                        continue
                    if field == 'buffs':
                        greens[selected, dst, ATTRS.index(effect['name'])] = True
                    elif effect['name'] in level['enemies'][dst]['weaknesses']:
                        reds[selected, dst] = True
    # 叠加未确认：多次命中仅计一次已知收益。
    powers = np.array([e['power'] for e in level['enemies']])
    buff_factors = 100 + np.array([e.get('buff_percent', 0) for e in level['enemies']])
    # power 已包含敌方增益；还原基础战力后，增益与红色减益按百分点相加。
    # 中间值不取整，仅将最终敌方有效战力向上取整。
    reduction = level.get('outgoing_debuff_effect_reduction_percent', 0)
    red_numerator = 20 * (100 - reduction)
    denominators = buff_factors[None, :] * 100
    enemy = (powers[None, :] * (denominators - reds * red_numerator)
             + denominators - 1) // denominators
    base = np.array([[u[a] for a in ATTRS] for u in units], dtype=np.int64)
    margins = []
    for health in (1, 2, 3):
        stats = (base * (health + 7) // 10)[formations]
        if order != 'after-buffs':
            for a, percent in level.get('debuff_percent', {}).items():
                j = ATTRS.index(a)
                stats[:, :, j] = stats[:, :, j] * (100 - percent) // 100
        stats = stats + stats * greens * 3 // 10
        if order == 'after-buffs':
            for a, percent in level.get('debuff_percent', {}).items():
                j = ATTRS.index(a)
                stats[:, :, j] = stats[:, :, j] * (100 - percent) // 100
        margins.append(stats.sum(axis=2) - 10 - enemy)
    if order is None:
        # 顺序未确认时，只使用受条件影响的属性没有同时收到绿色增益的排阵。
        for a in level['debuff']:
            valid &= ~greens[:, :, ATTRS.index(a)].any(axis=1)
    return np.array(margins), valid, enemy


def reinforcement_table(units, level, formations):
    """单卡只选一个属性；不计绿色增益对道具收益的额外放大。"""
    gains = np.zeros((3, len(formations), 4), dtype=np.int64)
    attributes = np.full(gains.shape, -1, dtype=np.int8)
    config = level.get('reinforcement', {})
    if not config:
        return gains, attributes
    allowed = np.array([p in config.get('positions', [1, 2, 3, 4]) for p in range(1, 5)])
    for health in (1, 2, 3):
        for a in config['attributes']:
            raw = np.array([u[a] * (health + 7) // 10 for u in units])[formations]
            factor = 100 - level.get('debuff_percent', {}).get(a, 0)
            # 道具与属性条件的顺序未确认：两种折算顺序取较小增量。
            gain = np.minimum((raw * factor // 100) * config['percent'] // 100,
                              (raw * config['percent'] // 100) * factor // 100)
            better = (gain > gains[health - 1]) & allowed[None, :]
            gains[health - 1][better] = gain[better]
            attributes[health - 1][better] = ATTRS.index(a)
    return gains, attributes


def apply_reinforcements(margins, gains, minimum, max_units):
    """先补齐最低余量，再在限额内改善其他非正余量；每卡最多一次。"""
    required = margins < minimum
    valid = ((~required | (margins + gains >= minimum)).all(axis=1)
             & (required.sum(axis=1) <= max_units))
    optional = (~required & (margins <= 0) & (gains > 0))
    remaining = np.maximum(0, max_units - required.sum(axis=1))
    selected = required | (optional & (optional.cumsum(axis=1) <= remaining[:, None]))
    return margins + gains * selected, selected, valid


def evaluate_choices(units, level, formation, health, regular, special, order):
    """固定阵容的补强组合；0 表示不使用，普通/特殊代码均为 1–4。"""
    regular = np.asarray(regular, dtype=np.int8).reshape(-1, 4)
    special = np.asarray(special, dtype=np.int8).reshape(-1, 4)
    if regular.shape != special.shape or not np.all((regular >= 0) & (regular <= 4)) or not np.all((special >= 0) & (special <= 4)):
        raise ValueError('每个补强组合须分别给出四个 0–4 选项。')
    count = len(regular)
    valid = np.full(count, len(set(formation)) == 4 and all(1 <= h <= 3 for h in health))
    for config, selections, names in ((level.get('reinforcement', {}), regular, REGULAR_OPTIONS),
                                       (level.get('special_reinforcement', {}), special, SPECIAL_OPTIONS)):
        options = config.get('options', config.get('attributes', []))
        positions = config.get('positions', [1, 2, 3, 4])
        for p in range(4):
            allowed = [0] + [i + 1 for i, name in enumerate(names) if name in options and p + 1 in positions]
            valid &= np.isin(selections[:, p], allowed)
        valid &= (selections > 0).sum(axis=1) <= config.get('max_units', 0)
    if any(units[i]['type'] in level['forbidden'] for i in formation):
        valid[:] = False
    healed = np.minimum(3, np.array(health)[None, :] + (special == 4) * 2)
    base = np.array([[units[i][a] for a in ATTRS] for i in formation])
    raw = base[None, :, :] * (healed[:, :, None] + 7) // 10
    greens = np.zeros((count, 4, 3), dtype=np.int64)
    reds = np.zeros((count, 4), dtype=np.int64)
    reduction = level.get('outgoing_debuff_effect_reduction_percent', 0)
    for src, index in enumerate(formation):
        source = units[index]
        enabled = special[:, src] != 1
        for effect in source['buffs']:
            dst = src + OFFSET[effect['direction']]
            if 0 <= dst < 4 and not level.get('disable_buffs', False):
                rate = np.where(enabled, np.where(regular[:, src] == 4, 60, 30), 0)
                attribute = ATTRS.index(effect['name'])
                greens[:, dst, attribute] = np.maximum(greens[:, dst, attribute], rate)
        for effect in source['debuffs']:
            dst = src + OFFSET[effect['direction']]
            if 0 <= dst < 4:
                matches = effect['name'] in level['enemies'][dst]['weaknesses']
                applies = enabled & (matches | (special[:, src] == 3))
                rate = np.where(applies, 20 * np.where(special[:, src] == 2, 2, 1) * (100 - reduction), 0)
                reds[:, dst] = np.maximum(reds[:, dst], rate)
    powers = np.array([e['power'] for e in level['enemies']])
    denominators = np.array([(100 + e.get('buff_percent', 0)) * 100 for e in level['enemies']])
    enemy = (powers[None, :] * (denominators[None, :] - reds) + denominators[None, :] - 1) // denominators[None, :]
    conditions = np.array([level.get('debuff_percent', {}).get(a, 0) for a in ATTRS])
    conditioned = raw * (100 - conditions) // 100
    before = conditioned + conditioned * greens // 100
    after = (raw + raw * greens // 100) * (100 - conditions) // 100
    if order == 'after-buffs':
        totals = after.sum(axis=2)
    elif order == 'both':
        totals = np.minimum(before.sum(axis=2), after.sum(axis=2))
    else:
        totals = before.sum(axis=2)
    # +40% 与关卡减益按百分点相加；保留邻卡增益，不放大其额外收益。
    isolated = raw * (140 - conditions) // 100
    isolated_green = np.minimum(conditioned * greens // 100,
                                (raw * greens // 100) * (100 - conditions) // 100)
    totals = np.where(special == 1, (isolated + isolated_green).sum(axis=2), totals)
    percent = level.get('reinforcement', {}).get('percent', 0)
    item_gains = np.minimum(conditioned * percent // 100,
                            (raw * percent // 100) * (100 - conditions) // 100)
    chosen_attribute = np.maximum(regular - 1, 0)
    gain = np.take_along_axis(item_gains, np.minimum(chosen_attribute, 2)[:, :, None], axis=2)[:, :, 0]
    gain = np.where((regular >= 1) & (regular <= 3), gain, 0)
    low = totals + gain - 10
    if order is None:
        valid &= ~((greens > 0) & (conditions > 0)).any(axis=(1, 2))
    return {'valid': valid, 'low': low, 'enemy': enemy, 'margins': low - enemy,
            'health': healed, 'next_health': healed - 1, 'greens': greens, 'reds': reds,
            'item_gain': gain, 'regular': regular, 'special': special}


def position_choices(units, level, formation, health):
    """只保留单属性最大增量；它不改变能力或后续血量，不会丢掉更优选择。"""
    regular_config = level.get('reinforcement', {})
    special_config = level.get('special_reinforcement', {})
    regular_options = regular_config.get('options', regular_config.get('attributes', []))
    per_position = []
    for p, index in enumerate(formation):
        special_options = [0]
        if p + 1 in special_config.get('positions', [1, 2, 3, 4]):
            for code, option in enumerate(SPECIAL_OPTIONS, 1):
                if option not in special_config.get('options', []):
                    continue
                if code in (2, 3) and not units[index]['debuffs']:
                    continue
                if code == 4 and health[p] >= 3:
                    continue
                special_options.append(code)
        choices = []
        for special in special_options:
            regular_options_here = [0]
            if p + 1 in regular_config.get('positions', [1, 2, 3, 4]):
                effective_health = min(3, health[p] + (2 if special == 4 else 0))
                gains = []
                for a in ATTRS:
                    raw = units[index][a] * (effective_health + 7) // 10
                    factor = 100 - level.get('debuff_percent', {}).get(a, 0)
                    percent = regular_config.get('percent', 0)
                    gains.append(min((raw * factor // 100) * percent // 100,
                                     (raw * percent // 100) * factor // 100) if a in regular_options else -1)
                if max(gains) > 0:
                    regular_options_here.append(int(np.argmax(gains)) + 1)
                if 'double_buffs' in regular_options and units[index]['buffs'] and special != 1:
                    regular_options_here.append(4)
            choices.extend((regular, special) for regular in regular_options_here)
        per_position.append(choices)
    return per_position


def read_seed_plan(path, units, stages):
    """只读取旧报告中的阵容，不继承血量、补强或旧胜负结论。"""
    ids = {u['id']: i for i, u in enumerate(units)}
    plan = {}
    key = None
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if line.startswith('## stage'):
            key = tuple(line[3:].split('/'))
        if re.match(r'^\| [1-4] /', line):
            match = re.search(r'（([a-z0-9_]+)）', line)
            if key is None or match is None or match[1] not in ids:
                raise ValueError('阵容参考文件的单位 ID 缺失或不在当前卡池。')
            plan.setdefault(key, []).append(ids[match[1]])
    expected = {(stage, name) for group in stages for stage, name, _ in group}
    if set(plan) != expected:
        raise ValueError('阵容参考文件须包含当前全部关卡。')
    for group in stages:
        used = set()
        for stage, name, _ in group:
            formation = plan[stage, name]
            if len(formation) != 4 or len(set(formation)) != 4 or used.intersection(formation):
                raise ValueError('阵容参考文件须每关四个不同单位，且同阶段不得重复。')
            used.update(formation)
    return plan


class OptimizedPlan(dict):
    """报告携带优化范围，不把固定阵容最优误称为全卡池最优。"""


def level_choices_dp(units, level, formation, health, minimum, max_risk, order, check_time=lambda: None):
    """沿四个位置推进 DP；一个对位只依赖左邻、自己、右邻。"""
    if min(health) <= 0 or any(units[i]['type'] in level['forbidden'] for i in formation):
        return []
    options = position_choices(units, level, formation, health)
    zero = (0, 0)
    conditions = np.array([level.get('debuff_percent', {}).get(a, 0) for a in ATTRS])

    def run(per_position):
        # 缓存三张相邻卡的局部结果，不构造四位置补强的笛卡尔积。
        local = []
        for p in range(4):
            check_time()
            left = per_position[p-1] if p else [zero]
            right = per_position[p+1] if p < 3 else [zero]
            triples = [(a, b, c) for a in left for b in per_position[p] for c in right]
            regular = np.zeros((len(triples), 4), dtype=np.int8)
            special = np.zeros_like(regular)
            for row, triple in enumerate(triples):
                for source, action in zip((p-1, p, p+1), triple):
                    if 0 <= source < 4:
                        regular[row, source], special[row, source] = action
            values = evaluate_choices(units, level, formation, health, regular, special, order)
            margins = values['margins'][:, p]
            valid = margins >= minimum
            if order is None:
                valid &= ~((values['greens'][:, p] > 0) & (conditions > 0)).any(axis=1)
            local.append({triple: int(margin) for triple, margin, allowed in zip(triples, margins, valid) if allowed})

        # key: 最后两个动作、战后血量前缀、普通补强数、风险数。
        # 同状态只保留更少特殊补强；已关闭的对位不会影响后续位置。
        states = {(zero, zero, (), 0, 0): (0, (), (), ())}
        for p in range(4):
            check_time()
            next_states = {}
            for (left, own, hp_prefix, normal_count, risks), (special_count, regular_path, special_path, margins) in states.items():
                for action in per_position[p]:
                    ordinary, extra = action
                    nc, sc = normal_count + (ordinary > 0), special_count + (extra > 0)
                    if nc > level.get('reinforcement', {}).get('max_units', 0) or sc > level.get('special_reinforcement', {}).get('max_units', 0):
                        continue
                    new_margins = margins
                    new_risks = risks
                    if p:
                        margin = local[p-1].get((left, own, action))
                        if margin is None:
                            continue
                        new_risks += margin <= 0
                        new_margins += (margin,)
                    if new_risks > max_risk:
                        continue
                    after_hp = min(3, health[p] + (2 if extra == 4 else 0)) - 1
                    key = (own, action, hp_prefix + (after_hp,), nc, new_risks)
                    value = (sc, regular_path + (ordinary,), special_path + (extra,), new_margins)
                    previous = next_states.get(key)
                    quality = (sc, -min(new_margins, default=0), -sum(new_margins))
                    if previous is None or quality < (previous[0], -min(previous[3], default=0), -sum(previous[3])):
                        next_states[key] = value
            states = next_states
        results = {}
        for (left, own, next_hp, nc, risks), (sc, regular, special, margins) in states.items():
            margin = local[3].get((left, own, zero))
            if margin is None or risks + (margin <= 0) > max_risk:
                continue
            risks += margin <= 0
            margins += (margin,)
            signature = (next_hp, risks)
            value = {'next_health': next_hp, 'risks': risks, 'cost': (sc, nc),
                     'regular': regular, 'special': special, 'margin': min(margins)}
            previous = results.get(signature)
            if previous is None or (sc, nc, -value['margin']) < (*previous['cost'], -previous['margin']):
                results[signature] = value
        return list(results.values())

    ordinary_only = run([[action for action in choices if action[1] == 0] for choices in options])
    if any(choice['risks'] == 0 for choice in ordinary_only):
        # 普通补强已能赢时，非回血特殊补强不改变下一关状态且成本更高，全部支配淘汰。
        options = [[action for action in choices if action[1] in (0, 4)] for choices in options]
        if all(all(action[1] == 0 for action in choices) for choices in options):
            return ordinary_only
    return run(options)


def solve_choices(units, stages, seed, seconds, minimum, max_risk, order, caches=None):
    """全关卡血量 DP：特殊次数第一目标，普通次数第二目标。"""
    started = time.monotonic()
    levels = [entry for group in stages for entry in group]
    future_units = [sorted({u for stage, name, _ in levels[i:] for u in seed[stage, name]}) for i in range(len(levels)+1)]
    caches = caches if caches is not None else {}
    memo, cache = caches.setdefault('global', {}), caches.setdefault('local', {})
    suffix = [tuple(tuple(seed[stage,name]) for stage,name,_ in levels[i:]) for i in range(len(levels)+1)]
    nodes = 0
    incumbent = None
    def check_time():
        if seconds and time.monotonic() - started >= seconds:
            raise TimeoutError
    def visit(index, hp, budget):
        nonlocal nodes, incumbent
        check_time()
        if index == len(levels):
            return (0, 0), {}
        state = (index, suffix[index], tuple(hp[u] for u in future_units[index]), budget)
        if state in memo:
            return memo[state]
        nodes += 1
        stage, name, level = levels[index]
        formation = seed[stage, name]
        health = tuple(hp[i] for i in formation)
        key = (index, tuple(formation), health)
        if key not in cache:
            cache[key] = level_choices_dp(units, level, formation, health, minimum, max_risk, order, check_time)
        # 下一关看不到的回血差异立即合并；相同状态只保留字典序成本最小的转移。
        transitions = {}
        for choice in cache[key]:
            if choice['risks'] > budget:
                continue
            next_hp = list(hp)
            for p, unit in enumerate(formation):
                next_hp[unit] = choice['next_health'][p]
            signature = (tuple(next_hp[u] for u in future_units[index+1]), choice['risks'])
            prior = transitions.get(signature)
            if prior is None or choice['cost'] < prior[0]['cost']:
                transitions[signature] = (choice, tuple(next_hp))
        best = None
        for choice, next_hp in sorted(transitions.values(), key=lambda item: item[0]['cost']):
            if best is not None and choice['cost'] >= best[0]:
                continue
            tail = visit(index+1, next_hp, budget-choice['risks'])
            if tail is None:
                continue
            cost = tuple(a+b for a,b in zip(choice['cost'], tail[0]))
            if best is None or cost < best[0]:
                entry = {'formation': formation, 'regular': list(choice['regular']), 'special': list(choice['special'])}
                best = cost, {(stage, name): entry} | tail[1]
                if index == 0:
                    incumbent = best
        memo[state] = best
        return best
    complete = True
    try:
        result = visit(0, (3,) * len(units), max_risk)
    except TimeoutError:
        complete, result = False, incumbent
    plan = OptimizedPlan(result[1] if result else {})
    plan.special_optimal = complete and result is not None
    plan.optimization_scope = '当前参考阵容'
    status = 'found' if result else ('no_plan_with_seed' if complete else 'timeout')
    return status, plan, nodes, time.monotonic() - started


def optimize_formations(units, stages, seed, seconds, minimum, max_risk, order):
    """换位/换卡邻域搜索；每个候选的补强及回血成本由完整 DP 求最优。"""
    started = time.monotonic()
    caches = {}
    levels = [entry for group in stages for entry in group]
    keys = [(stage,name) for stage,name,_ in levels]
    seen = set()
    nodes = 0
    def remaining():
        return max(0.001, seconds-(time.monotonic()-started)) if seconds else 0
    def signature(candidate):
        return tuple(tuple(candidate[key]) for key in keys)
    def measure(plan):
        return (sum(sum(code>0 for code in choice['special']) for choice in plan.values()),
                sum(sum(code>0 for code in choice['regular']) for choice in plan.values()))
    def strength(plan):
        # 同特殊次数的过渡候选允许多用普通补强；保留较好的后续血量以跨过局部停滞。
        hp=[3]*len(units)
        total=0
        for key in keys:
            choice=plan[key]
            for p,i in enumerate(choice['formation']):
                battle=min(3,hp[i]+(2 if choice['special'][p]==4 else 0))
                total+=sum(units[i][a] for a in ATTRS)*(battle+7)
                hp[i]=battle-1
        return total
    status, best, count, _ = solve_choices(units, stages, seed, remaining(), minimum, max_risk, order, caches)
    nodes+=count
    if not best:
        return status,best,nodes,time.monotonic()-started
    best_cost=measure(best)
    beam=[(seed,best)]
    seen.add(signature(seed))
    rounds=0
    while beam and (not seconds or time.monotonic()-started<seconds):
        rounds+=1
        frontier=[]
        for source,source_plan in beam:
            # 有特殊补强的关卡先处理，其余关卡仍可为后续保留血量。
            ordered=sorted(keys,key=lambda key:-sum(code>0 for code in source_plan[key]['special']))
            used={stage:{u for s,n in keys if s==stage for u in source[s,n]} for stage in {s for s,n in keys}}
            def candidates():
                for key in ordered:
                    stage,name=key
                    f=source[key]
                    allowed=next(level['forbidden'] for s,n,level in levels if (s,n)==key)
                    for a in range(4):
                        for b in range(a+1,4):
                            candidate={k:list(v) for k,v in source.items()}
                            candidate[key][a],candidate[key][b]=candidate[key][b],candidate[key][a]
                            yield candidate
                        for u in range(len(units)):
                            if u in used[stage] or units[u]['type'] in allowed:
                                continue
                            candidate={k:list(v) for k,v in source.items()}
                            candidate[key][a]=u
                            yield candidate
                for a,left in enumerate(ordered):
                    for right in ordered[a+1:]:
                        if left[0]!=right[0]:
                            continue
                        for p in range(4):
                            for q in range(4):
                                candidate={k:list(v) for k,v in source.items()}
                                candidate[left][p],candidate[right][q]=candidate[right][q],candidate[left][p]
                                yield candidate
            for candidate in candidates():
                if seconds and time.monotonic()-started>=seconds:
                    break
                sig=signature(candidate)
                if sig in seen:
                    continue
                seen.add(sig)
                _,plan,count,_=solve_choices(units,stages,candidate,remaining(),minimum,max_risk,order,caches)
                nodes+=count
                if not plan:
                    continue
                cost=measure(plan)
                if cost<best_cost:
                    best,best_cost=plan,cost
                    print(f'优化进展：特殊 {cost[0]} 次，普通 {cost[1]} 次。',file=sys.stderr,flush=True)
                if cost[0]<=best_cost[0]:
                    frontier.append((candidate,plan,cost,strength(plan)))
            if seconds and time.monotonic()-started>=seconds:
                break
        # 同一阵容不重复展开。保留普通次数较少和后续血量较好的不同候选。
        frontier=[entry for entry in frontier if entry[2][0]<=best_cost[0]]
        by_cost=sorted(frontier,key=lambda entry:(entry[2],-entry[3]))[:2]
        by_health=sorted(frontier,key=lambda entry:(entry[2][0],-entry[3],entry[2][1]))[:2]
        unique={signature(entry[0]):(entry[0],entry[1]) for entry in by_cost+by_health}
        beam=list(unique.values())
        if rounds>=4:
            break
    best.fixed_special_optimal=best.special_optimal
    best.optimization_scope='阵容邻域搜索；每个候选使用全关卡成本 DP'
    best.special_optimal=False
    return 'found',best,nodes,time.monotonic()-started


def solve(units, stages, seconds, minimum, max_risk, order):
    started = time.monotonic()
    def check_time():
        if seconds and time.monotonic() - started >= seconds:
            raise TimeoutError
    formations = None
    nodes = 0
    failed = set()
    try:
        check_time()
        formations = np.array(list(itertools.permutations(range(len(units)), 4)))
        masks = np.bitwise_or.reduce(np.left_shift(np.uint64(1), formations.astype(np.uint64)), axis=1)
        cost = np.array([sum(u[a] for a in ATTRS) for u in units])[formations].sum(axis=1)
        tables = {}
        for levels in stages:
            for stage, name, level in levels:
                check_time()
                table, valid, enemy = make_table(units, level, formations, order)
                gains, _ = reinforcement_table(units, level, formations)
                tables[stage, name] = (table, valid, enemy, gains, level.get('reinforcement', {}).get('max_units', 0))
        check_time()
        def candidates(key, hp):
            check_time()
            table, valid, _, gains, max_units = tables[key]
            health = np.array(hp)[formations]
            margins = table[np.maximum(health, 1) - 1, np.arange(len(formations))[:, None], np.arange(4)]
            boosts = gains[np.maximum(health, 1) - 1, np.arange(len(formations))[:, None], np.arange(4)]
            margins, selected, reinforcement_valid = apply_reinforcements(margins, boosts, minimum, max_units)
            item_counts = selected.sum(axis=1)
            risks = (margins <= 0).sum(axis=1)
            ids = np.flatnonzero(valid & reinforcement_valid & (health > 0).all(axis=1)
                                 & (margins.min(axis=1) >= minimum) & (risks <= max_risk))
            # 相同四张卡的排列只保留风险最少者，未来血量和可用卡相同。
            ordered = ids[np.lexsort((-margins[ids].sum(axis=1), -margins[ids].min(axis=1), item_counts[ids],
                                      risks[ids], masks[ids]))]
            _, first = np.unique(masks[ordered], return_index=True)
            ids = ordered[first]
            ids = ids[np.lexsort((-margins[ids].sum(axis=1), item_counts[ids], cost[ids], risks[ids]))]
            return ids, risks
        def visit(stage_index, hp, budget, path):
            nonlocal nodes
            check_time()
            if stage_index == len(stages):
                return path
            state = (stage_index, hp, budget)
            if state in failed:
                return None
            levels = stages[stage_index]
            if sum(h > 0 for h in hp) < 4 * len(levels):
                failed.add(state)
                return None
            groups = {}
            lower_risk = 0
            # 后续血量只会下降；当前血量下已不可行的后续关卡可提前剪枝。
            for future in stages[stage_index:]:
                for stage, name, _ in future:
                    key = (stage, name)
                    ids, risks = candidates(key, hp)
                    if not len(ids):
                        failed.add(state)
                        return None
                    lower_risk += int(risks[ids].min())
                    if future is levels:
                        groups[key] = (ids, risks)
            if lower_risk > budget:
                failed.add(state)
                return None
            seen_ends = {}
            def assign(remaining, used, spent, chosen):
                nonlocal nodes
                check_time()
                if not remaining:
                    if seen_ends.get(used, math.inf) <= spent:
                        return None
                    seen_ends[used] = spent
                    next_hp = tuple(h - ((used >> i) & 1) for i, h in enumerate(hp))
                    return visit(stage_index + 1, next_hp, budget - spent, path | chosen)
                choices = []
                for key in remaining:
                    ids, risks = groups[key]
                    ids = ids[((masks[ids] & np.uint64(used)) == 0) & (risks[ids] + spent <= budget)]
                    if not len(ids):
                        return None
                    choices.append((len(ids), key, ids))
                _, key, ids = min(choices, key=lambda c: c[0])
                for f in ids:
                    nodes += 1
                    result = assign([k for k in remaining if k != key], used | int(masks[f]),
                                    spent + int(groups[key][1][f]), chosen | {key: int(f)})
                    if result is not None:
                        return result
                return None
            result = assign(list(groups), 0, 0, {})
            if result is None:
                failed.add(state)
            return result
        result = visit(0, (3,) * len(units), max_risk, {})
        status = 'found' if result is not None else 'no_solution_in_model'
        plan = {key: formations[index].tolist() for key, index in (result or {}).items()}
    except TimeoutError:
        status, plan = 'timeout', {}
    return status, plan, nodes, time.monotonic() - started


def report(units, stages, status, plan, nodes, elapsed, args):
    if getattr(args, 'advanced', False) or (plan and isinstance(next(iter(plan.values())), dict)):
        return report_choices(units, stages, status, plan, nodes, elapsed, args)
    texts = {'found': '找到满足指定约束的完整方案（未证明最优，未实测游戏）',
             'timeout': '搜索达到时限，尚未找到完整方案；不能判定无解',
             'no_solution_in_model': '穷举完成：指定保守模型和风险约束下无解'}
    lines = ['# 战斗规划', '', texts[status], '',
             f'输入：{args.enemies.name}；单位 {len(units)} 张，起始均为三格血。',
             f'搜索用时 {elapsed:.2f} 秒，分支 {nodes}；最低保守余量要求 {args.min_margin}，允许风险对位 {args.max_risk}。',
             '模型：血量系数 100% / 90% / 80%；绿色属性 +30%；匹配红色能力通常 −20%，再按各关对敌减益效果降低幅度折算；战力为三维之和 ±10。',
             '敌方 power 为输入战力；已按用户要求除去增益的值不再次还原。仍记录 b% 增益且红色命中时，按 power × (100+b−r)/(100+b) 计算，r 为该关实际红色减益百分数，最终向上取整。',
             '同属性绿色、同敌人红色多次命中只计一次收益。我方逐步向下取整，敌方有效战力向上取整；表中是保守边界，不是游戏读数。',
             '允许补强的关卡：每张卡只选一个属性；按当前血量基础属性计算道具增量，不计绿色能力的额外放大，属性条件与道具按较差顺序折算。',
             '条件顺序：' + ('两种顺序分别计算，逐位取较差保守边界。' if args.condition_order == 'both'
                             else args.condition_order or '未确认；排除条件减益与同属性绿色增益同时生效的排阵。'), '']
    hp = [3] * len(units)
    total_items = 0
    all_margins = []
    for levels in stages if plan else []:
        used = set()
        for stage, name, level in levels:
            f = plan[stage, name]
            assert not used.intersection(f)
            used.update(f)
            margins, valid, enemy = make_table(units, level, np.array([f]), args.condition_order)
            assert valid[0]
            gains, attributes = reinforcement_table(units, level, np.array([f]))
            health_indices = np.array([hp[i] - 1 for i in f])
            original_margins = margins[health_indices, 0, np.arange(4)][None, :]
            available_gains = gains[health_indices, 0, np.arange(4)][None, :]
            current_margins, selected, boost_valid = apply_reinforcements(
                original_margins, available_gains, args.min_margin,
                level.get('reinforcement', {}).get('max_units', 0))
            assert boost_valid[0]
            total_items += int(selected.sum())
            all_margins.extend(current_margins[0].tolist())
            lines += [f'## {stage}/{name}', '']
            if level.get('disable_buffs', False):
                lines += ['特殊条件：所有绿色增益无效；匹配的红色减益仍生效。', '']
            reduction = level.get('outgoing_debuff_effect_reduction_percent', 0)
            red_percent = 20 * (100 - reduction) / 100
            if reduction:
                lines += [f'特殊条件：我方对敌减益效果降低 {reduction}%；匹配红色减益由 20% 降为 {red_percent:g}%。', '']
            if level['debuff']:
                lines += ['我方属性条件：' + '；'.join(
                    f'{LABEL[a]} −{level["debuff_percent"][a]}%' for a in level['debuff']) + '。', '']
            if level.get('reinforcement'):
                config = level['reinforcement']
                positions = '、'.join(str(p) for p in config.get('positions', [1, 2, 3, 4]))
                lines += [f'允许单属性 +{config["percent"]}% 补强，限 {positions} 号位，每关最多 {config["max_units"]} 张卡；本关使用 {int(selected.sum())} 次。', '']
            if level.get('notes'):
                lines += [level['notes'], '']
            lines += ['| 位置 / 单位 / 血量 | 单属性补强 | 绿色增益来源 | 保守战力范围 | 敌方输入 / 截图 / 增益 → 保守有效值 | 红色减益来源 | 保守余量 |',
                      '| --- | --- | --- | --- | --- | --- | --- |']
            for dst, i in enumerate(f):
                green, red, credited = [], [], set()
                for src, j in enumerate(f):
                    for field, target in (('buffs', green), ('debuffs', red)):
                        if field == 'buffs' and level.get('disable_buffs', False):
                            continue
                        for e in units[j][field]:
                            if src + OFFSET[e['direction']] == dst and (
                                field == 'buffs' or e['name'] in level['enemies'][dst]['weaknesses']
                            ):
                                detail = f'{src+1}号{units[j]["name"]}：{LABEL[e["name"]]}/{e["direction"]}'
                                if field == 'buffs':
                                    value = units[i][e['name']] * (hp[i] + 7) // 10
                                    if args.condition_order == 'both' and e['name'] in level['debuff']:
                                        before_value = value * (100 - level['debuff_percent'][e['name']]) // 100
                                        detail += (f'，条件先算时 +{before_value * 3 // 10}；增益先算时 +{value * 3 // 10} 后再扣条件'
                                                   if e['name'] not in credited else '（重复，不另计）')
                                    else:
                                        if args.condition_order != 'after-buffs':
                                            value = value * (100 - level.get('debuff_percent', {}).get(e['name'], 0)) // 100
                                        detail += f'，+{value * 3 // 10}' if e['name'] not in credited else '（重复，不另计）'
                                    credited.add(e['name'])
                                else:
                                    detail += f'，减益 {red_percent:g}%（多次命中只计一次）'
                                target.append(detail)
                margin = int(current_margins[0, dst])
                reinforcement = '无'
                if selected[0, dst]:
                    a = ATTRS[attributes[hp[i]-1, 0, dst]]
                    reinforcement = f'{LABEL[a]} +{level["reinforcement"]["percent"]}%（保守 +{int(available_gains[0, dst])}）'
                effective = int(enemy[0, dst])
                low = margin + effective
                foe = level['enemies'][dst]
                enemy_reading = str(foe['power'])
                if 'display_power' in foe:
                    enemy_reading += f'（截图 {foe["display_power"]}，已除去指定增益）'
                if foe.get('buff_percent', 0):
                    enemy_reading += f'（增益 +{foe["buff_percent"]}%）'
                lines.append(f'| {dst+1} / {units[i]["name"]}（{units[i]["id"]}） / {hp[i]}血 | {reinforcement} | {"；".join(green) or "无"} | '
                             f'{low}–{low+20} | {enemy_reading} → {effective} | '
                             f'{"；".join(red) or "无"} | {margin}{"（风险）" if margin <= 0 else ""} |')
            for i in f:
                hp[i] -= 1
            lines += ['', '本关推演结算后：' + '；'.join(f'{units[i]["name"]} {hp[i]}血' for i in f) + '。', '']
        lines += ['本阶段未出战：' + '、'.join(u['name'] for i, u in enumerate(units) if i not in used) + '。', '']
    if all_margins:
        lines += [f'全场最低保守余量：{min(all_margins)}；非正余量对位：{sum(m <= 0 for m in all_margins)}。', '']
    lines += [f'方案道具消耗：{total_items} 次单属性补强。' if plan else '未生成完整方案，道具消耗未确定。',
              '输入文件不被修改；识图正确性仍需人工核对。']
    return '\n'.join(lines) + '\n'


def report_choices(units, stages, status, plan, nodes, elapsed, args):
    texts = {'found': '找到满足指定约束的完整方案（未证明最优，未实测游戏）',
             'timeout': '搜索达到时限，尚未找到完整方案；不能判定无解',
             'no_plan_with_seed': '当前参考阵容未找到满足约束的补强组合；不能判定其他阵容无解',
             'no_seed_plan': '尚未构造完整参考阵容；可用 --seed-plan 指定已有报告，不能判定新补强模型无解'}
    lines = ['# 战斗规划', '', texts[status], '',
             f'输入：{args.enemies.name}；单位 {len(units)} 张，起始均为三格血。',
             f'补强联合搜索用时 {elapsed:.2f} 秒，分支 {nodes}；最低保守余量要求 {args.min_margin}，允许风险对位 {args.max_risk}。',
             '每卡每关最多一个普通补强和一个特殊补强。翻倍按能力来源卡计算；多个同属性绿色或同敌人红色效果只取最强一次，不额外叠加。',
             '特殊补强 1 关闭该卡发出的绿色、红色能力，可接收邻卡绿色增益；三维 +40% 与关卡减益按百分点相加。',
             '特殊补强 3 只取消弱点要求，不取消方向要求；生命 +2 在战前回血，上限三格，不用于复活零血单位。',
             '我方逐步向下取整，敌方最终向上取整。普通三维道具按基础属性计算，不放大绿色收益；特殊 +40% 不放大邻卡增益的额外收益。',
             '属性条件与绿色增益按 ' + ('两种顺序取较差结果。' if args.condition_order == 'both'
                                        else args.condition_order or '未确认时排除同属性重合。'),
             '优化目标：先最少特殊补强，再最少普通补强；所有对位仍须满足最低余量和风险限制。',
             '单关 DP 先检验普通补强；整场 DP 合并等价未来血量状态，不继承旧报告状态。', '']
    if plan:
        lines += ['优化范围：' + getattr(plan, 'optimization_scope', '当前参考阵容') + '。', '']
    hp = [3] * len(units)
    all_margins = []
    regular_count = special_count = 0
    for group in stages if plan else []:
        used = set()
        for stage, name, level in group:
            choice = plan[stage, name]
            f = choice['formation']
            regular, special = choice['regular'], choice['special']
            assert not used.intersection(f)
            used.update(f)
            initial_health = [hp[i] for i in f]
            values = evaluate_choices(units, level, f, initial_health, [regular], [special], args.condition_order)
            assert values['valid'][0]
            regular_count += sum(code > 0 for code in regular)
            special_count += sum(code > 0 for code in special)
            all_margins.extend(values['margins'][0].tolist())
            lines += [f'## {stage}/{name}', '']
            if level.get('notes'):
                lines += [level['notes'], '']
            lines += ['| 位置 / 单位 / 战前血量 | 普通补强 | 特殊补强 | 绿色增益来源 | 保守战力范围 | 敌方输入 → 保守有效值 | 红色减益来源 | 保守余量 |',
                      '| --- | --- | --- | --- | --- | --- | --- | --- |']
            for dst, index in enumerate(f):
                green, red = [], []
                credited = set()
                for src, source_index in enumerate(f):
                    if special[src] == 1:
                        continue
                    source = units[source_index]
                    for effect in source['buffs']:
                        if src + OFFSET[effect['direction']] != dst or level.get('disable_buffs', False):
                            continue
                        a = effect['name']
                        rate = 60 if regular[src] == 4 else 30
                        best_rate = int(values['greens'][0, dst, ATTRS.index(a)])
                        detail = f'{src+1}号{source["name"]}：{LABEL[a]}/{effect["direction"]}，+{rate}%'
                        if a in credited or rate < best_rate:
                            detail += '（不另叠加）'
                        else:
                            raw = units[index][a] * (int(values['health'][0, dst]) + 7) // 10
                            factor = 100 - level.get('debuff_percent', {}).get(a, 0)
                            before_gain = (raw * factor // 100) * rate // 100
                            after_gain = (raw + raw * rate // 100) * factor // 100 - raw * factor // 100
                            if special[dst] == 1:
                                gain = min(before_gain, (raw * rate // 100) * factor // 100)
                                detail += f'（保守 +{gain}）'
                            elif before_gain == after_gain:
                                detail += f'（+{before_gain}）'
                            else:
                                detail += f'（条件先算 +{before_gain}；增益先算 +{after_gain}）'
                            credited.add(a)
                        green.append(detail)
                    for effect in source['debuffs']:
                        if src + OFFSET[effect['direction']] != dst:
                            continue
                        matches = effect['name'] in level['enemies'][dst]['weaknesses']
                        if not matches and special[src] != 3:
                            continue
                        rate = 20 * (2 if special[src] == 2 else 1) * (100 - level.get('outgoing_debuff_effect_reduction_percent', 0)) / 100
                        detail = f'{src+1}号{source["name"]}：{LABEL[effect["name"]]}/{effect["direction"]}，减益 {rate:g}%'
                        if not matches:
                            detail += '（特殊补强忽略弱点）'
                        if rate * 100 < values['reds'][0, dst]:
                            detail += '（按更强来源计算，不叠加）'
                        red.append(detail)
                ordinary = '无' if regular[dst] == 0 else OPTION_LABEL[REGULAR_OPTIONS[regular[dst]-1]]
                if 1 <= regular[dst] <= 3:
                    ordinary += f' +{level["reinforcement"]["percent"]}%（保守 +{int(values["item_gain"][0,dst])}）'
                extra = '无' if special[dst] == 0 else OPTION_LABEL[SPECIAL_OPTIONS[special[dst]-1]]
                health_text = str(initial_health[dst])
                if special[dst] == 4:
                    health_text += f'→{int(values["health"][0,dst])}'
                low, enemy, margin = (int(values[field][0,dst]) for field in ('low', 'enemy', 'margins'))
                foe = level['enemies'][dst]
                reading = str(foe['power'])
                if 'display_power' in foe:
                    reading += f'（截图 {foe["display_power"]}，已除去指定增益）'
                if foe.get('buff_percent', 0):
                    reading += f'（仍含增益 +{foe["buff_percent"]}%）'
                lines.append(f'| {dst+1} / {units[index]["name"]}（{units[index]["id"]}） / {health_text}血 | {ordinary} | {extra} | '
                             f'{"；".join(green) or "无"} | {low}–{low+20} | {reading} → {enemy} | '
                             f'{"；".join(red) or "无"} | {margin}{"（风险）" if margin<=0 else ""} |')
            for p, index in enumerate(f):
                hp[index] = int(values['next_health'][0,p])
            lines += ['', '本关推演结算后：' + '；'.join(f'{units[i]["name"]} {hp[i]}血' for i in f) + '。', '']
        lines += ['本阶段未出战：' + '、'.join(u['name'] for i,u in enumerate(units) if i not in used) + '。', '']
    if all_margins:
        lines += [f'全场最低保守余量：{min(all_margins)}；非正余量对位：{sum(m<=0 for m in all_margins)}。',
                  f'补强使用：普通 {regular_count} 次，特殊 {special_count} 次。', '']
    lines += ['此结果只验证当前输入及保守计算，不代表游戏实测或全局最优；输入文件不被修改。']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('enemies', type=Path, help='本轮 enemies JSON')
    parser.add_argument('--units', type=Path, default=Path(__file__).with_name('units.json'))
    parser.add_argument('--seconds', type=float, default=60, help='总搜索时限，默认60秒；0为不限时')
    parser.add_argument('--min-margin', type=int, default=1, help='最低保守余量；负数允许风险')
    parser.add_argument('--max-risk', type=int, default=0, help='允许多少个非正余量对位，默认0')
    parser.add_argument('--condition-order', choices=('before-buffs', 'after-buffs', 'both'),
                        help='已确认的据点减益顺序；both 分别计算两种顺序并取较差结果')
    parser.add_argument('--output', type=Path, help='可选：保存 Markdown；默认只输出到终端')
    parser.add_argument('--seed-plan', type=Path, help='可选：以已有 Markdown 报告的阵容联合搜索新补强，不继承旧血量')
    parser.add_argument('--fixed-formations', action='store_true', help='仅对参考阵容做精确成本 DP，跳过阵容邻域搜索')
    args = parser.parse_args()
    try:
        if not math.isfinite(args.seconds) or args.seconds < 0 or args.max_risk < 0:
            raise ValueError('时限和风险数量须为非负数。')
        if args.output and args.output.resolve() in {args.units.resolve(), args.enemies.resolve(), Path(__file__).resolve()}:
            raise ValueError('输出路径不能覆盖输入文件或脚本。')
        units, stages = read_inputs(args.units, args.enemies)
        args.advanced = any(level.get('special_reinforcement') or 'double_buffs' in level.get('reinforcement', {}).get('options', [])
                            for group in stages for _,_,level in group)
        if args.advanced:
            advanced_started = time.monotonic()
            base_nodes = 0
            if args.seed_plan is None and args.enemies.stem.startswith('enemies_'):
                candidate = args.enemies.with_name('plan_' + args.enemies.stem[len('enemies_'):] + '.md')
                if candidate.exists():
                    try:
                        seed = read_seed_plan(candidate, units, stages)
                    except (OSError, ValueError):
                        seed = None
                    if seed:
                        args.seed_plan = candidate
            if args.seed_plan:
                seed = read_seed_plan(args.seed_plan, units, stages)
            else:
                _, seed, base_nodes, _ = solve(units, stages, args.seconds / 2, min(-1000, args.min_margin),
                                               4 * sum(len(group) for group in stages), args.condition_order)
            remaining = args.seconds - (time.monotonic() - advanced_started) if args.seconds else 0
            if not seed:
                status, plan, nodes = 'no_seed_plan', {}, base_nodes
            elif args.seconds and remaining <= 0:
                status, plan, nodes = 'timeout', {}, base_nodes
            else:
                optimize = solve_choices if args.fixed_formations else optimize_formations
                status, plan, nodes, _ = optimize(units, stages, seed, remaining, args.min_margin, args.max_risk, args.condition_order)
                nodes += base_nodes
            elapsed = time.monotonic() - advanced_started
        else:
            status, plan, nodes, elapsed = solve(units, stages, args.seconds, args.min_margin, args.max_risk, args.condition_order)
        text = report(units, stages, status, plan, nodes, elapsed, args)
        if args.output:
            args.output.write_text(text, encoding='utf-8')
        else:
            print(text, end='')
        return {'found': 0, 'no_solution_in_model': 3, 'timeout': 4,
                'no_plan_with_seed': 5, 'no_seed_plan': 5}[status]
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(main())
