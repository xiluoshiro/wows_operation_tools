"""全卡池 CP-SAT 行动规划：先最少特殊增强，再最少普通增强。"""
import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

local_deps = Path(__file__).with_name('.solver_deps')
if local_deps.exists():
    sys.path.insert(0, str(local_deps))
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


def format_power(value):
    """Keep fractional combat margins visible without floating-point tails."""
    return f'{float(value):.10g}'

def read_inputs(units_path, enemies_path):
    units = json.loads(units_path.read_text(encoding='utf-8-sig'))
    enemies = json.loads(enemies_path.read_text(encoding='utf-8-sig'))
    if not isinstance(units, list) or len(units) < 4:
        raise ValueError('卡池至少须有四张单位卡，每关使用四张。')
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
    enemy = powers[None, :] * (denominators[None, :] - reds) / denominators[None, :]
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

def cost(plan):
    return tuple(sum(sum(code > 0 for code in entry[field]) for entry in plan.values())
                 for field in ('special', 'regular'))

def validate_plan(units, stages, plan, minimum, max_risk, order):
    expected = {(s, n) for group in stages for s, n, _ in group}
    if set(plan) != expected:
        raise ValueError('方案必须覆盖全部关卡。')
    hp = [3] * len(units)
    risks = 0
    minimum_seen = None
    snapshots = {}
    for group in stages:
        used = set()
        for s, n, level in group:
            entry = plan[s, n]
            f = entry['formation']
            if len(f) != 4 or len(set(f)) != 4 or used.intersection(f):
                raise ValueError('方案违反每关四张卡或同阶段不能重复的限制。')
            if any(type(i) is not int or not 0 <= i < len(units) for i in f):
                raise ValueError('方案单位索引超出卡池。')
            if any(len(entry[field]) != 4 or any(type(c) is not int or not 0 <= c <= 4 for c in entry[field])
                   for field in ('regular', 'special')):
                raise ValueError('每关普通、特殊补强各须四个 0–4 整数选项。')
            used.update(f)
            health = [hp[i] for i in f]
            values = evaluate_choices(units, level, f, health, entry['regular'], entry['special'], order)
            margins = values['margins'][0]
            if not values['valid'][0] or margins.min() < minimum:
                raise ValueError(f'{s}/{n} 的血量、补强限制或最低余量不满足要求。')
            risks += int((margins <= 0).sum())
            minimum_seen = min(minimum_seen, float(margins.min())) if minimum_seen is not None else float(margins.min())
            snapshots[s, n] = (tuple(health), tuple(int(h) for h in values['next_health'][0]))
            for p, i in enumerate(f):
                hp[i] = int(values['next_health'][0, p])
    if risks > max_risk:
        raise ValueError('方案风险对位超过限制。')
    return snapshots, minimum_seen

def read_seed_plan(path, units, stages):
    """只读取旧报告中的阵容，不继承血量、补强或旧胜负结论。"""
    ids = {u['id']: i for i, u in enumerate(units)}
    plan = {}
    if path.suffix.lower() == '.json':
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        for group in stages:
            for stage, name, _ in group:
                plan[stage, name] = [ids[i] for i in data[stage + '_' + name]['units']]
    key = None
    for line in path.read_text(encoding='utf-8-sig').splitlines() if path.suffix.lower() != '.json' else []:
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

def read_seed_choices(path, units, stages, minimum, max_risk, order):
    """读取已选补强但从三格血重新验证；限时搜索不会丢失可行基线。"""
    formations = read_seed_plan(path, units, stages)
    plan = {}
    if path.suffix.lower() == '.json':
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        for key, formation in formations.items():
            entry = data['_'.join(key)]
            plan[key] = dict(formation=formation, regular=entry['regular'], special=entry['special'])
    else:
        key = None
        advanced = False
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if line.startswith('## stage'):
                key = tuple(line[3:].split('/'))
            if line.startswith('| 位置 /'):
                advanced = '特殊补强' in line
            if not re.match(r'^\| [1-4] /', line):
                continue
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            ordinary = cells[1]
            regular = 0 if ordinary == '无' else 4 if '增益效果' in ordinary else next(
                (i + 1 for i, a in enumerate(ATTRS) if ordinary.startswith(LABEL[a])), -1)
            extra = cells[2] if advanced else '无'
            special = 0 if extra == '无' else next(
                (i + 1 for i, option in enumerate(SPECIAL_OPTIONS) if extra.startswith(OPTION_LABEL[option])), -1)
            if regular < 0 or special < 0:
                raise ValueError('参考报告的补强选项无法识别。')
            entry = plan.setdefault(key, dict(formation=formations[key], regular=[], special=[]))
            entry['regular'].append(regular)
            entry['special'].append(special)
    validate_plan(units, stages, plan, minimum, max_risk, order)
    return plan

def report_choices(units, stages, status, plan, nodes, elapsed, args):
    texts = {'OPTIMAL': '已证明当前输入及规则模型下的全局最优；未实测游戏。',
             'FEASIBLE': '已找到可行方案，尚未证明最优。',
             'UNKNOWN': '达到时限；保留已核验方案，尚未证明最优。' if plan else '达到时限，尚未找到方案；不能判定无解。',
             'INFEASIBLE': '已证明当前输入和指定约束下无解。'}
    summary = args.solver_summary
    lines = ['# 战斗规划', '', texts[status], '',
             f'输入：{args.enemies.name}；单位 {len(units)} 张，起始均为三格血。',
             f'全卡池 CP-SAT 求解用时 {elapsed:.2f} 秒；最低余量要求 {args.min_margin}，允许风险对位 {args.max_risk}。',
             '每卡每关最多一个普通补强和一个特殊补强。翻倍按能力来源卡计算；多个同属性绿色或同敌人红色效果只取最强一次，不额外叠加。',
             '特殊补强 1 关闭该卡发出的绿色、红色能力，可接收邻卡绿色增益；三维 +40% 与关卡减益按百分点相加。',
             '特殊补强 3 只取消弱点要求，不取消方向要求；生命 +2 在战前回血，上限三格，不用于复活零血单位。',
             '我方属性仍逐步向下取整，敌方百分比结果保留小数。默认要求我方最低战力严格高于敌方计算值，真正相等仍列为风险。普通三维道具按基础属性计算，不放大绿色收益；特殊 +40% 不放大邻卡增益的额外收益。',
             '属性条件与绿色增益按 ' + ('两种顺序取较差结果。' if args.condition_order == 'both'
                                        else args.condition_order or '未确认时排除同属性重合。'),
             '优化目标：先最少特殊补强，再最少普通补强；所有对位仍须满足最低余量和风险限制。',
             '全部单位、关卡、血量和增强同时求解；不截取候选卡池，不固定参考阵容。',
             f'求解状态：{status}；目标值：{summary["objective"]}；已证明下界：{format_power(summary["best_objective_bound"])}。', '']
    if args.special_limit is not None:
        lines += [f'本次额外限制：全场特殊增强不超过 {args.special_limit} 次。', '']
    if plan:
        weight = summary['objective_weight']
        sc, nc = cost(plan)
        lines += [f'优化目标 Z = {weight}×特殊次数＋普通次数 = {weight}×{sc}＋{nc} = {summary["objective"]}。',
                  f'全场普通增强最多 {weight-1} 次，权重 {weight} 保证先最少特殊、再最少普通。', '']
        if summary['optimal']:
            lines += [f'最优性依据：完整模型证明 Z ≥ {summary["objective"]}，此方案达到该下界，故在本次约束下全局最优。', '']
        elif not any(any(entry['special']) for entry in plan.values()):
            lines += ['特殊补强为零，已达到该项理论下界；普通补强次数未证明全局最少。', '']
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
            lines += ['| 位置 / 单位 / 战前血量 | 普通补强 | 特殊补强 | 绿色增益来源 | 保守战力范围 | 敌方输入 → 计算有效值 | 红色减益来源 | 保守余量 |',
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
                low = int(values['low'][0,dst])
                enemy, margin = (float(values[field][0,dst]) for field in ('enemy', 'margins'))
                foe = level['enemies'][dst]
                reading = str(foe['power'])
                if 'display_power' in foe:
                    reading += f'（截图 {foe["display_power"]}，已除去指定增益）'
                if foe.get('buff_percent', 0):
                    reading += f'（仍含增益 +{foe["buff_percent"]}%）'
                lines.append(f'| {dst+1} / {units[index]["name"]}（{units[index]["id"]}） / {health_text}血 | {ordinary} | {extra} | '
                             f'{"；".join(green) or "无"} | {low}–{low+20} | {reading} → {format_power(enemy)} | '
                             f'{"；".join(red) or "无"} | {format_power(margin)}{"（风险）" if margin<=0 else ""} |')
            for p, index in enumerate(f):
                hp[index] = int(values['next_health'][0,p])
            lines += ['', '本关推演结算后：' + '；'.join(f'{units[i]["name"]} {hp[i]}血' for i in f) + '。', '']
        lines += ['本阶段未出战：' + '、'.join(u['name'] for i,u in enumerate(units) if i not in used) + '。', '']
    if all_margins:
        lines += [f'全场最低保守余量：{format_power(min(all_margins))}；非正余量对位：{sum(m<=0 for m in all_margins)}。',
                  f'补强使用：普通 {regular_count} 次，特殊 {special_count} 次。', '']
    lines += ['结论适用于当前输入与上述保守规则；未提供的其他条件按输入处理，计算核验不代替原始卡面核对或游戏实测。输入文件不被修改。']
    return '\n'.join(lines) + '\n'


def write_json(path, payload):
    """Replace outputs atomically, including intermediate complete solutions."""
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    temporary.replace(path)


def export_plan(plan, units):
    return {s + '_' + n: dict(units=[units[i]['id'] for i in entry['formation']],
                             regular=entry['regular'], special=entry['special'])
            for (s, n), entry in plan.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('enemies', type=Path, help='本轮敌人 JSON')
    parser.add_argument('--units', type=Path, default=Path(__file__).with_name('units.json'))
    parser.add_argument('--seed-plan', type=Path, help='可选 JSON/Markdown 方案提示；不锁定阵容')
    parser.add_argument('--seconds', type=float, default=300, help='求解时限，默认300秒；0为不限时')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--min-margin', type=int, default=0)
    parser.add_argument('--max-risk', type=int, default=0)
    parser.add_argument('--condition-order', choices=('both', 'before-buffs', 'after-buffs'), default='both')
    parser.add_argument('--special-limit', type=int, help='可选的全场特殊增强次数上限')
    parser.add_argument('--output', type=Path, help='逐位 Markdown 报告；省略时输出到终端')
    parser.add_argument('--plan-json', type=Path, help='可复用的完整方案')
    parser.add_argument('--certificate', type=Path, help='状态、目标、下界和输入/代码哈希')
    parser.add_argument('--audit', type=Path, help='独立 Fraction 核验结果')
    parser.add_argument('--model', type=Path, help='可选导出 CP-SAT 模型')
    parser.add_argument('--log', type=Path, help='可选详细求解日志')
    args = parser.parse_args()
    try:
        if not math.isfinite(args.seconds) or args.seconds < 0 or args.workers < 1 or args.max_risk < 0:
            raise ValueError('时限和风险数须非负，线程数须为正整数。')
        if args.special_limit is not None and args.special_limit < 0:
            raise ValueError('特殊增强上限须非负。')
        source_paths = [Path(__file__), Path(__file__).with_name('battle_exact.py'),
                        Path(__file__).with_name('battle_verify.py')]
        protected = {p.resolve() for p in [args.units, args.enemies, *source_paths]}
        outputs = [p.resolve() for p in (args.output, args.plan_json, args.certificate, args.audit, args.model, args.log) if p]
        if len(outputs) != len(set(outputs)) or any(p in protected for p in outputs):
            raise ValueError('输出路径须互不相同，不能覆盖输入或源代码。')
        units, stages = read_inputs(args.units, args.enemies)
        explicit_seed = args.seed_plan is not None
        if not explicit_seed and args.enemies.stem.startswith('enemies_'):
            suffix = args.enemies.stem[len('enemies_'):]
            args.seed_plan = next((p for p in (args.enemies.with_name('formation_' + suffix + '.json'),
                                               args.enemies.with_name('plan_' + suffix + '.md')) if p.exists()), None)
        seed = None
        if args.seed_plan:
            try:
                seed = read_seed_choices(args.seed_plan, units, stages, args.min_margin, args.max_risk, args.condition_order)
            except (OSError, ValueError, KeyError, TypeError) as error:
                if explicit_seed:
                    raise ValueError(f'参考方案无效：{error}') from error
                print(f'自动参考方案无效，将从全卡池求解：{error}', file=sys.stderr)
        from battle_verify import audit_plan, summary_markdown
        if seed:
            audit_plan(units, stages, seed, args.min_margin, args.max_risk, args.condition_order)
        hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [args.units, args.enemies, *source_paths]}
        seed_hash = hashlib.sha256(args.seed_plan.read_bytes()).hexdigest() if seed else None
        from battle_exact import solve_exact
        def checkpoint(plan):
            if args.plan_json:
                write_json(args.plan_json, export_plan(plan, units))
        plan, summary = solve_exact(units, stages, seed, args.seconds, args.workers,
                                    args.special_limit, args.min_margin, args.max_risk, args.condition_order,
                                    args.model, args.log, checkpoint)
        audit = audit_plan(units, stages, plan, args.min_margin, args.max_risk, args.condition_order) if plan else None
        args.solver_summary = summary
        text = report_choices(units, stages, summary['status'], plan, summary['branches'], summary['elapsed'], args)
        if plan:
            audit_plan(units, stages, plan, args.min_margin, args.max_risk, args.condition_order, text)
            text = text.replace('# 战斗规划\n', '# 战斗规划\n\n' + summary_markdown(audit) + '\n', 1)
            checkpoint(plan)
        if args.output:
            args.output.write_text(text, encoding='utf-8', newline='\n')
        else:
            print(text, end='')
        if args.audit:
            write_json(args.audit, audit if audit else dict(status=summary['status'], verified=False))
        if args.certificate:
            from ortools import __version__
            summary.update(hashes=hashes, seed_sha256=seed_hash,
                           solver_version='OR-Tools CP-SAT ' + __version__,
                           fraction_verified=bool(audit),
                           minimum_margin_exact=audit['minimum_margin_exact'] if audit else None)
            if args.model:
                summary['model_sha256'] = hashlib.sha256(args.model.read_bytes()).hexdigest()
            if plan:
                encoded = (json.dumps(export_plan(plan, units), ensure_ascii=False, indent=2) + '\n').encode('utf-8')
                summary['plan_sha256'] = hashlib.sha256(encoded).hexdigest()
            write_json(args.certificate, summary)
        print(f'状态 {summary["status"]}；目标 {summary["objective"]}；下界 {summary["best_objective_bound"]:g}', file=sys.stderr)
        return {'OPTIMAL': 0, 'FEASIBLE': 0, 'INFEASIBLE': 3, 'UNKNOWN': 4}[summary['status']]
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(main())
