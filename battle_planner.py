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
            foes = sorted(level['enemies'], key=lambda e: e['position'])
            if [e['position'] for e in foes] != [1, 2, 3, 4]:
                raise ValueError(f'{stage}/{name} 须填写位置 1–4 的四个敌人。')
            for e in foes:
                if type(e['power']) is not int or e['power'] < 0:
                    raise ValueError(f'{stage}/{name} 敌方战力须为非负整数。')
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
    greens = np.zeros((len(formations), 4, 3), dtype=bool)
    reds = np.zeros((len(formations), 4), dtype=bool)
    valid = np.ones(len(formations), dtype=bool)
    for src in range(4):
        for i, u in enumerate(units):
            selected = formations[:, src] == i
            if u['type'] in level['forbidden']:
                valid[selected] = False
            for field in ('buffs', 'debuffs'):
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
    enemy = (powers[None, :] * np.where(reds, 8, 10) + 9) // 10
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
                tables[stage, name] = make_table(units, level, formations, order)
        check_time()
        def candidates(key, hp):
            check_time()
            table, valid, _ = tables[key]
            health = np.array(hp)[formations]
            margins = table[np.maximum(health, 1) - 1, np.arange(len(formations))[:, None], np.arange(4)]
            risks = (margins <= 0).sum(axis=1)
            ids = np.flatnonzero(valid & (health > 0).all(axis=1)
                                 & (margins.min(axis=1) >= minimum) & (risks <= max_risk))
            # 相同四张卡的排列只保留风险最少者，未来血量和可用卡相同。
            ordered = ids[np.lexsort((-margins[ids].sum(axis=1), -margins[ids].min(axis=1),
                                      risks[ids], masks[ids]))]
            _, first = np.unique(masks[ordered], return_index=True)
            ids = ordered[first]
            ids = ids[np.lexsort((-margins[ids].sum(axis=1), cost[ids], risks[ids]))]
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
    texts = {'found': '找到满足指定约束的完整方案（未证明最优，未实测游戏）',
             'timeout': '搜索达到时限，尚未找到完整方案；不能判定无解',
             'no_solution_in_model': '穷举完成：指定保守模型和风险约束下无解'}
    lines = ['# 战斗规划', '', texts[status], '',
             f'输入：{args.enemies.name}；单位 {len(units)} 张，起始均为三格血。',
             f'搜索用时 {elapsed:.2f} 秒，分支 {nodes}；最低保守余量要求 {args.min_margin}，允许风险对位 {args.max_risk}。',
             '模型：血量系数 100% / 90% / 80%；绿色属性 +30%；匹配红色能力 −20%；战力为三维之和 ±10。',
             '同属性绿色、同敌人红色多次命中只计一次收益。我方逐步向下取整，敌方有效战力向上取整；表中是保守边界，不是游戏读数。',
             '条件顺序：' + (args.condition_order or '未确认；排除条件减益与同属性绿色增益同时生效的排阵。'), '']
    hp = [3] * len(units)
    for levels in stages if plan else []:
        used = set()
        for stage, name, level in levels:
            f = plan[stage, name]
            assert not used.intersection(f)
            used.update(f)
            margins, valid, enemy = make_table(units, level, np.array([f]), args.condition_order)
            assert valid[0]
            lines += [f'## {stage}/{name}', '',
                      '| 位置 / 单位 / 血量 | 绿色增益来源 | 保守战力范围 | 敌方原值 → 保守有效值 | 红色减益来源 | 保守余量 |',
                      '| --- | --- | --- | --- | --- | --- |']
            for dst, i in enumerate(f):
                green, red, credited = [], [], set()
                for src, j in enumerate(f):
                    for field, target in (('buffs', green), ('debuffs', red)):
                        for e in units[j][field]:
                            if src + OFFSET[e['direction']] == dst and (
                                field == 'buffs' or e['name'] in level['enemies'][dst]['weaknesses']
                            ):
                                detail = f'{src+1}号{units[j]["name"]}：{LABEL[e["name"]]}/{e["direction"]}'
                                if field == 'buffs':
                                    value = units[i][e['name']] * (hp[i] + 7) // 10
                                    if args.condition_order != 'after-buffs':
                                        value = value * (100 - level.get('debuff_percent', {}).get(e['name'], 0)) // 100
                                    detail += f'，+{value * 3 // 10}' if e['name'] not in credited else '（重复，不另计）'
                                    credited.add(e['name'])
                                target.append(detail)
                margin = int(margins[hp[i]-1, 0, dst])
                effective = int(enemy[0, dst])
                low = margin + effective
                lines.append(f'| {dst+1} / {units[i]["name"]} / {hp[i]}血 | {"；".join(green) or "无"} | '
                             f'{low}–{low+20} | {level["enemies"][dst]["power"]} → {effective} | '
                             f'{"；".join(red) or "无"} | {margin}{"（风险）" if margin <= 0 else ""} |')
            for i in f:
                hp[i] -= 1
            lines += ['', '本关推演结算后：' + '；'.join(f'{units[i]["name"]} {hp[i]}血' for i in f) + '。', '']
        lines += ['本阶段未出战：' + '、'.join(u['name'] for i, u in enumerate(units) if i not in used) + '。', '']
    lines += ['道具消耗：0。输入文件不被修改；识图正确性仍需人工核对。']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('enemies', type=Path, help='本轮 enemies JSON')
    parser.add_argument('--units', type=Path, default=Path(__file__).with_name('units.json'))
    parser.add_argument('--seconds', type=float, default=60, help='总搜索时限，默认60秒；0为不限时')
    parser.add_argument('--min-margin', type=int, default=1, help='最低保守余量；负数允许风险')
    parser.add_argument('--max-risk', type=int, default=0, help='允许多少个非正余量对位，默认0')
    parser.add_argument('--condition-order', choices=('before-buffs', 'after-buffs'), help='已确认的据点减益顺序')
    parser.add_argument('--output', type=Path, help='可选：保存 Markdown；默认只输出到终端')
    args = parser.parse_args()
    try:
        if not math.isfinite(args.seconds) or args.seconds < 0 or args.max_risk < 0:
            raise ValueError('时限和风险数量须为非负数。')
        if args.output and args.output.resolve() in {args.units.resolve(), args.enemies.resolve(), Path(__file__).resolve()}:
            raise ValueError('输出路径不能覆盖输入文件或脚本。')
        units, stages = read_inputs(args.units, args.enemies)
        status, plan, nodes, elapsed = solve(units, stages, args.seconds, args.min_margin, args.max_risk, args.condition_order)
        text = report(units, stages, status, plan, nodes, elapsed, args)
        if args.output:
            args.output.write_text(text, encoding='utf-8')
        else:
            print(text, end='')
        return {'found': 0, 'no_solution_in_model': 3, 'timeout': 4}[status]
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(main())
