"""Independent Fraction audit of an actual plan; no solver arithmetic is reused."""
import math
import re
from fractions import Fraction as Q

ATTRS = ('firepower', 'armor', 'maneuverability')
REGULAR = (*ATTRS, 'double_buffs')
SPECIAL = ('stats_40_disable_effects', 'double_debuffs', 'ignore_weakness', 'health_plus_2')
OFFSET = {'left': -1, 'right': 1, 'front_left': -1, 'front': 0, 'front_right': 1}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def display(value):
    return f'{float(value):.10g}'


def audit_plan(units, stages, plan, minimum=0, max_risk=0, order='both', report=None):
    """Recompute all battles, restrictions and HP, returning exact rational evidence."""
    expected = {(s, n) for group in stages for s, n, _ in group}
    require(set(plan) == expected, '方案必须覆盖且仅覆盖当前全部关卡。')
    hp = [3] * len(units)
    positions, rationals, summaries = [], [], []
    regular_count = special_count = risks = 0
    minimum_seen = None
    exported = report_rows(report) if report is not None else None
    if exported is not None:
        require(len(exported) == 4 * len(expected), '报告对位数量不一致。')
    for group in stages:
        used = set()
        for stage, name, level in group:
            entry = plan[stage, name]
            ids = entry['formation']
            regular, special = entry['regular'], entry['special']
            tag = f'{stage}/{name}'
            require(len(ids) == 4 and len(set(ids)) == 4 and not used.intersection(ids), tag + ' 单位重复或数量不符。')
            require(all(type(i) is int and 0 <= i < len(units) for i in ids), tag + ' 单位索引无效。')
            used.update(ids)
            for field, codes, options in (('reinforcement', regular, REGULAR),
                                           ('special_reinforcement', special, SPECIAL)):
                config = level.get(field, {})
                require(len(codes) == 4 and all(type(c) is int and 0 <= c <= 4 for c in codes), tag + ' 增强代码无效。')
                require(sum(bool(c) for c in codes) <= config.get('max_units', 0), tag + ' 增强数量超限。')
                for p, code in enumerate(codes):
                    if code:
                        allowed = config.get('options', config.get('attributes', []))
                        require(p + 1 in config.get('positions', [1, 2, 3, 4])
                                and options[code - 1] in allowed, tag + f' {p+1} 号增强不允许。')
            greens, reds = [{} for _ in range(4)], [0] * 4
            formation = [units[i] for i in ids]
            for src, unit in enumerate(formation):
                if special[src] == 1:
                    continue
                for effect in unit['buffs']:
                    dst = src + OFFSET[effect['direction']]
                    if 0 <= dst < 4 and not level.get('disable_buffs', False):
                        greens[dst][effect['name']] = max(greens[dst].get(effect['name'], 0),
                                                         60 if regular[src] == 4 else 30)
                for effect in unit['debuffs']:
                    dst = src + OFFSET[effect['direction']]
                    if 0 <= dst < 4 and (effect['name'] in level['enemies'][dst]['weaknesses'] or special[src] == 3):
                        reds[dst] = max(reds[dst], 40 if special[src] == 2 else 20)
            margins, descriptions = [], []
            for p, unit in enumerate(formation):
                index = ids[p]
                require(hp[index] > 0 and unit['type'] not in level['forbidden'], tag + ' 零血或禁用单位出战。')
                before_hp = hp[index]
                effective_hp = min(3, before_hp + (2 if special[p] == 4 else 0))
                before, after, isolated, gains = [], [], [], []
                config = level.get('reinforcement', {})
                for attribute in ATTRS:
                    raw = unit[attribute] * (effective_hp + 7) // 10
                    factor = Q(100 - level.get('debuff_percent', {}).get(attribute, 0), 100)
                    conditioned = math.floor(raw * factor)
                    rate = Q(greens[p].get(attribute, 0), 100)
                    before.append(conditioned + math.floor(conditioned * rate))
                    after.append(math.floor((raw + math.floor(raw * rate)) * factor))
                    isolated.append(math.floor(raw * (factor + Q(40, 100)))
                                    + min(math.floor(conditioned * rate), math.floor(math.floor(raw * rate) * factor)))
                    gains.append(min(math.floor(Q(conditioned * config.get('percent', 0), 100)),
                                     math.floor(math.floor(Q(raw * config.get('percent', 0), 100)) * factor)))
                total = min(sum(before), sum(after)) if order == 'both' else sum(after if order == 'after-buffs' else before)
                if special[p] == 1:
                    total = sum(isolated)
                low = total + (gains[regular[p] - 1] if regular[p] in (1, 2, 3) else 0) - 10
                foe = level['enemies'][p]
                denominator = 100 + foe.get('buff_percent', 0)
                reduction = Q(reds[p] * (100 - level.get('outgoing_debuff_effect_reduction_percent', 0)), 100)
                enemy = Q(foe['power']) * (denominator - reduction) / denominator
                margin = low - enemy
                require(margin >= minimum, tag + f' {p+1} 号余量低于要求：{margin}。')
                risks += margin <= 0
                minimum_seen = margin if minimum_seen is None else min(minimum_seen, margin)
                health_text = str(before_hp) if effective_hp == before_hp else f'{before_hp}→{effective_hp}'
                positions.append([stage, name, p + 1, unit['id'], before_hp, low, float(enemy), float(margin)])
                rationals.append(dict(stage=stage, level=name, position=p + 1,
                                      enemy=str(enemy), margin=str(margin), effective_health=effective_hp))
                if report is not None:
                    # Compare exported values as well as the selected plan.
                    row = exported[len(positions) - 1]
                    require(f'（{unit["id"]}） / {health_text}血' in row[0]
                            and row[4] == f'{low}–{low+20}'
                            and row[5].endswith('→ ' + display(enemy))
                            and row[7] == display(margin) + ('（风险）' if margin <= 0 else ''),
                            tag + f' {p+1} 号报告数值不一致。')
                suffix = ''
                if regular[p]:
                    choice = ['火力', '存活性', '机动性', '增益翻倍'][regular[p] - 1]
                    suffix += f'〔{choice}' + (f' +{config["percent"]}%' if regular[p] < 4 else '') + '〕'
                if special[p]:
                    suffix += '〔特殊：' + ['能力失效、三维 +40%', '红色减益翻倍', '忽略弱点', '生命 +2'][special[p] - 1] + '〕'
                descriptions.append(unit['name'] + suffix)
                hp[index] = effective_hp - 1
                margins.append(display(margin))
            regular_count += sum(bool(c) for c in regular)
            special_count += sum(bool(c) for c in special)
            summaries.append(dict(stage=stage, level=name, units=descriptions, margins=margins))
    require(risks <= max_risk, '非正余量对位超过风险上限。')
    return dict(positions=positions, exact_rationals=rationals, position_count=len(positions),
                minimum_margin=float(minimum_seen), minimum_margin_exact=str(minimum_seen),
                ordinary_count=regular_count, special_count=special_count, risk_count=risks,
                final_health={u['id']: hp[i] for i, u in enumerate(units)}, summary=summaries)


def report_rows(report):
    return [[cell.strip() for cell in line.split('|')[1:-1]] for line in report.splitlines()
            if re.match(r'^\| [1-4] / ', line)]


def summary_markdown(audit):
    lines = [f'独立 Fraction 核验：{audit["position_count"]}/{audit["position_count"]} 个对位通过；'
             f'最低保守余量 {display(audit["minimum_margin"])}；'
             f'普通增强 {audit["ordinary_count"]} 次，特殊增强 {audit["special_count"]} 次。', '',
             '| 关卡 | 1 号位 | 2 号位 | 3 号位 | 4 号位 | 保守余量 |',
             '| --- | --- | --- | --- | --- | --- |']
    for row in audit['summary']:
        label = row['stage'][5:] + '-' + row['level'][5:]
        lines.append('| ' + label + ' | ' + ' | '.join(row['units']) + ' | '
                     + ' / '.join(row['margins']) + ' |')
    return '\n'.join(lines)
