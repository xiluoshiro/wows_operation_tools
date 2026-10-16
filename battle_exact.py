"""Full-card-pool integer constraint optimization (no candidate truncation).

All combat arithmetic is tabulated with Python integers. Only dominated ordinary
attribute choices are merged: equal price and identical effects/HP, largest gain.
An OPTIMAL result proves the optimum for the input and conservative rule model.
"""
import time
from contextlib import nullcontext

import battle_planner as api
from ortools.sat.python import cp_model


def build(units, stages, seed=None, special_limit=None, minimum=0, max_risk=0, order='both'):
    m = cp_model.CpModel()
    slots = {}
    regular_flags, special_flags, risk_flags = [], [], []
    count = len(units)
    power_bound = 2 * max(sum(u[a] for a in api.ATTRS) for u in units)
    before_hp = [m.new_constant(3) for _ in units]
    hint_hp = [3] * count

    def integer(name, lo, hi, hint=None):
        v = m.new_int_var(lo, hi, name)
        if hint is not None:
            m.add_hint(v, int(hint))
        return v

    def equals(v, value, name, hint=None):
        b = m.new_bool_var(name)
        m.add(v == value).only_enforce_if(b)
        m.add(v != value).only_enforce_if(b.Not())
        if hint is not None:
            m.add_hint(b, int(hint))
        return b

    def element(index, values, name):
        out = integer(name, min(values), max(values))
        m.add_element(index, values, out)
        return out

    for si, group in enumerate(stages):
        stage_slots = []
        for s, n, level in group:
            entry = seed[s, n] if seed else None
            reg_config = level.get('reinforcement', {})
            sp_config = level.get('special_reinforcement', {})
            options = reg_config.get('options', reg_config.get('attributes', []))
            factors = [100 - level.get('debuff_percent', {}).get(a, 0) for a in api.ATTRS]
            percent = reg_config.get('percent', 0)
            raw_table, conditioned_table, isolated_table, gains, attr_codes = [], [], [], [], []
            for unit in units:
                for h in (1, 2, 3):
                    raw = [unit[a] * (h + 7) // 10 for a in api.ATTRS]
                    cond = [r * f // 100 for r, f in zip(raw, factors)]
                    isolated = [r * (f + 40) // 100 for r, f in zip(raw, factors)]
                    gain = [min(c * percent // 100, (r * percent // 100) * f // 100)
                            if a in options else -1
                            for r, c, f, a in zip(raw, cond, factors, api.ATTRS)]
                    raw_table.append(raw)
                    conditioned_table.append(sum(cond))
                    isolated_table.append(sum(isolated))
                    gains.append(max(0, max(gain)))
                    attr_codes.append(gain.index(max(gain)) + 1)
            local = []
            for p in range(4):
                tag = f'{s}_{n}_{p}'
                hinted_u = entry['formation'][p] if entry else None
                hinted_r = entry['regular'][p] if entry else None
                hinted_sp = entry['special'][p] if entry else None
                hinted_h = hint_hp[hinted_u] if entry else None
                def hinted(value):
                    return value if entry else None
                u = integer(tag + '_unit', 0, count - 1, hinted_u)
                allowed_units = [(i,) for i, unit in enumerate(units) if unit['type'] not in level['forbidden']]
                m.add_allowed_assignments([u], allowed_units)
                h = integer(tag + '_hp', 1, 3, hinted_h)
                m.add_element(u, before_hp, h)
                sp = integer(tag + '_special', 0, 4, hinted_sp)
                allowed_sp = [0] + [i + 1 for i, name in enumerate(api.SPECIAL_OPTIONS)
                                  if name in sp_config.get('options', [])
                                  and p + 1 in sp_config.get('positions', [1, 2, 3, 4])]
                m.add_allowed_assignments([sp], [(v,) for v in allowed_sp])
                none_sp = equals(sp, 0, tag + '_no_special', hinted(hinted_sp == 0))
                heal = equals(sp, 4, tag + '_heal', hinted(hinted_sp == 4))
                disabled = equals(sp, 1, tag + '_disabled', hinted(hinted_sp == 1))
                special_flags.append(none_sp.Not())
                effective_h = integer(tag + '_effective_hp', 1, 3,
                                      3 if hinted_sp == 4 else hinted_h)
                m.add(effective_h == 3).only_enforce_if(heal)
                m.add(effective_h == h).only_enforce_if(heal.Not())
                state = integer(tag + '_state', 0, count * 3 - 1)
                m.add(state == u * 3 + effective_h - 1)
                # 0 none; 1 largest permitted attribute; 2 double outgoing green.
                mode = integer(tag + '_ordinary', 0, 2,
                               (2 if hinted_r == 4 else int(hinted_r > 0)) if entry else None)
                attr = equals(mode, 1, tag + '_attribute', (0 < hinted_r < 4) if entry else None)
                double = equals(mode, 2, tag + '_double_green', hinted(hinted_r == 4))
                no_reg = equals(mode, 0, tag + '_no_ordinary', hinted(hinted_r == 0))
                regular_flags.append(no_reg.Not())
                if p + 1 not in reg_config.get('positions', [1, 2, 3, 4]) or not reg_config:
                    m.add(mode == 0)
                if not any(a in options for a in api.ATTRS):
                    m.add(attr == 0)
                if 'double_buffs' not in options or level.get('disable_buffs', False):
                    m.add(double == 0)
                m.add(double + disabled <= 1)
                # Doubling a source with no in-board green effect is dominated.
                m.add_allowed_assignments([u, mode], [(i, v) for i, unit in enumerate(units)
                    for v in (0, 1, 2) if v != 2 or any(0 <= p + api.OFFSET[e['direction']] < 4 for e in unit['buffs'])])
                gain = element(state, gains, tag + '_gain')
                item = integer(tag + '_item_gain', 0, max(gains))
                m.add(item == gain).only_enforce_if(attr)
                m.add(item == 0).only_enforce_if(attr.Not())
                effect_index = integer(tag + '_effect', 0, count * 15 - 1)
                m.add(effect_index == u * 15 + sp * 3 + mode)
                red_index = integer(tag + '_red_index', 0, count * 5 - 1)
                m.add(red_index == u * 5 + sp)
                record = dict(unit=u, hp=h, special=sp, heal=heal, disabled=disabled,
                              mode=mode, ordinary=no_reg.Not(), special_used=none_sp.Not(),
                              state=state, effect=effect_index, red_index=red_index,
                              item=item, attrs=attr_codes, before_table=conditioned_table,
                              isolated_table=isolated_table, tag=tag)
                local.append(record)
                stage_slots.append(record)
                slots[s, n, p] = record
                if entry:
                    hint_hp[hinted_u] = (3 if hinted_sp == 4 else hinted_h) - 1
            m.add(sum(r['ordinary'] for r in local) <= reg_config.get('max_units', 0))
            m.add(sum(r['special_used'] for r in local) <= sp_config.get('max_units', 0))
            for p, record in enumerate(local):
                tag = record['tag']
                before_extras, after_extras, isolated_extras = [], [], []
                for ai, attribute in enumerate(api.ATTRS):
                    sources = []
                    for src in (p - 1, p + 1):
                        if not 0 <= src < 4 or level.get('disable_buffs', False):
                            continue
                        values = []
                        for unit in units:
                            match = any(e['name'] == attribute and src + api.OFFSET[e['direction']] == p for e in unit['buffs'])
                            for sp in range(5):
                                for mode in range(3):
                                    values.append((2 if mode == 2 else 1) if match and sp != 1 else 0)
                        if max(values):
                            sources.append(element(local[src]['effect'], values, tag + f'_green_{ai}_from_{src}'))
                    rate = integer(tag + f'_green_{ai}', 0, 2)
                    m.add_max_equality(rate, sources or [0])
                    table_index = integer(tag + f'_green_index_{ai}', 0, count * 9 - 1)
                    m.add(table_index == record['state'] * 3 + rate)
                    b_values, a_values, iso_values = [], [], []
                    for raw in raw_table:
                        r, f = raw[ai], factors[ai]
                        c = r * f // 100
                        for v in (0, 30, 60):
                            b_values.append(c * v // 100)
                            a_values.append((r + r * v // 100) * f // 100 - c)
                            iso_values.append(min(c * v // 100, (r * v // 100) * f // 100))
                    before_extras.append(element(table_index, b_values, tag + f'_before_green_{ai}'))
                    after_extras.append(element(table_index, a_values, tag + f'_after_green_{ai}'))
                    isolated_extras.append(element(table_index, iso_values, tag + f'_isolated_green_{ai}'))
                base = element(record['state'], conditioned_table, tag + '_base')
                iso_base = element(record['state'], isolated_table, tag + '_iso_base')
                total = integer(tag + '_total', 0, power_bound)
                worst = integer(tag + '_worst_order', 0, power_bound)
                if order == 'both':
                    m.add_min_equality(worst, [base + sum(before_extras), base + sum(after_extras)])
                else:
                    extras = after_extras if order == 'after-buffs' else before_extras
                    m.add(worst == base + sum(extras))
                m.add(total == iso_base + sum(isolated_extras)).only_enforce_if(record['disabled'])
                m.add(total == worst).only_enforce_if(record['disabled'].Not())
                low = total + record['item'] - 10
                red_sources = []
                enemy = level['enemies'][p]
                for src in (p - 1, p, p + 1):
                    if not 0 <= src < 4:
                        continue
                    values = []
                    for unit in units:
                        directional = [e for e in unit['debuffs'] if src + api.OFFSET[e['direction']] == p]
                        match = any(e['name'] in enemy['weaknesses'] for e in directional)
                        for sp in range(5):
                            applies = sp != 1 and (match or (sp == 3 and directional))
                            values.append((40 if sp == 2 else 20) if applies else 0)
                    if max(values):
                        red_sources.append(element(local[src]['red_index'], values, tag + f'_red_from_{src}'))
                red = integer(tag + '_red', 0, 40)
                m.add_max_equality(red, red_sources or [0])
                denom = (100 + enemy.get('buff_percent', 0)) * 100
                effectiveness = 100 - level.get('outgoing_debuff_effect_reduction_percent', 0)
                difference = low * denom - enemy['power'] * (denom - red * effectiveness)
                m.add(difference >= minimum * denom)
                if max_risk:
                    risk = m.new_bool_var(tag + '_risk')
                    m.add(difference <= 0).only_enforce_if(risk)
                    m.add(difference >= 1).only_enforce_if(risk.Not())
                    risk_flags.append(risk)
                else:
                    # Strict win, without float tolerance or an arbitrary +1 power.
                    m.add(difference >= 1)
        m.add_all_different([r['unit'] for r in stage_slots])
        next_hp = []
        for i in range(count):
            selected, healed = [], []
            for r in stage_slots:
                chosen = equals(r['unit'], i, r['tag'] + f'_is_{i}')
                selected.append(chosen)
                both = m.new_bool_var(r['tag'] + f'_heals_{i}')
                m.add(both <= chosen)
                m.add(both <= r['heal'])
                m.add(both >= chosen + r['heal'] - 1)
                healed.append(both)
            used = m.new_bool_var(f'stage{si}_uses_{i}')
            m.add(used == sum(selected))
            healed_unit = m.new_bool_var(f'stage{si}_heals_{i}')
            m.add(healed_unit == sum(healed))
            hp = integer(f'stage{si}_next_hp_{i}', 0, 3, hint_hp[i] if seed else None)
            m.add(hp == 2).only_enforce_if(healed_unit)
            m.add(hp == before_hp[i] - used).only_enforce_if(healed_unit.Not())
            next_hp.append(hp)
        before_hp = next_hp
    total_special = integer('total_special', 0, len(slots))
    total_regular = integer('total_regular', 0, len(slots))
    m.add(total_special == sum(special_flags))
    m.add(total_regular == sum(regular_flags))
    if special_limit is not None:
        m.add(total_special <= special_limit)
    m.add(sum(risk_flags) <= max_risk)
    weight = len(slots) + 1
    if seed:
        incumbent_cost = api.cost(seed)
        m.add(weight * total_special + total_regular <= weight * incumbent_cost[0] + incumbent_cost[1])
    m.minimize(weight * total_special + total_regular)
    return m, slots, total_special, total_regular, weight


def extract(solver, slots, units, stages, minimum=0, max_risk=0, order='both'):
    result = {}
    for group in stages:
        for s, n, _ in group:
            rows = [slots[s, n, p] for p in range(4)]
            result[s, n] = dict(
                formation=[solver.value(r['unit']) for r in rows],
                regular=[(r['attrs'][solver.value(r['state'])] if solver.value(r['mode']) == 1
                          else 4 if solver.value(r['mode']) == 2 else 0) for r in rows],
                special=[solver.value(r['special']) for r in rows])
    api.validate_plan(units, stages, result, minimum, max_risk, order)
    return result



def solve_exact(units, stages, seed=None, seconds=300, workers=8,
                special_limit=None, minimum=0, max_risk=0, order='both',
                model_path=None, log_path=None, on_solution=None):
    """Return the best plan and explicit proof status; limits never imply optimality."""
    start = time.monotonic()
    model, slots, sp, reg, weight = build(units, stages, seed, special_limit,
                                        minimum, max_risk, order)
    error = model.validate()
    if error:
        raise ValueError(error)
    if model_path:
        model.export_to_file(str(model_path))
    eligible_seed = seed if seed and (special_limit is None or api.cost(seed)[0] <= special_limit) else None
    best = eligible_seed

    class Save(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            nonlocal best
            best = extract(self, slots, units, stages, minimum, max_risk, order)
            if on_solution:
                on_solution(best)
            print(f'方案：特殊 {self.value(sp)}，普通 {self.value(reg)}；'
                  f'目标下界 {self.best_objective_bound:g}；{self.wall_time:.2f} 秒', flush=True)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = workers
    if seconds:
        solver.parameters.max_time_in_seconds = seconds
    solver.parameters.log_search_progress = bool(log_path)
    solver.parameters.log_to_stdout = False
    with log_path.open('w', encoding='utf-8') if log_path else nullcontext() as log:
        if log:
            log.write(model.model_stats() + '\n')
            def record(message):
                log.write(message + '\n')
                log.flush()
            solver.log_callback = record
        status = solver.solve(model, Save())
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            best = extract(solver, slots, units, stages, minimum, max_risk, order)
        elif status == cp_model.INFEASIBLE:
            best = None
        elif status == cp_model.MODEL_INVALID:
            raise ValueError(solver.response_stats())
        if log:
            log.write(solver.response_stats())
    result_cost = list(api.cost(best)) if best else None
    objective = weight * result_cost[0] + result_cost[1] if best else None
    summary = dict(status=solver.status_name(status), complete_card_pool=True,
                   dominance='Only dominated ordinary choices removed: smaller attribute gains or ineffective green doubling.',
                   condition_order=order, minimum_margin=minimum, max_risk=max_risk,
                   strict_positive_margin=max_risk == 0,
                   objective_weight=weight, objective=objective, cost=result_cost,
                   best_objective_bound=solver.best_objective_bound,
                   optimal=status == cp_model.OPTIMAL,
                   retained_seed=status == cp_model.UNKNOWN and best is not None,
                   special_limit=special_limit, seconds=seconds, workers=workers,
                   elapsed=time.monotonic() - start,
                   branches=solver.num_branches, response_stats=solver.response_stats())
    return best, summary
