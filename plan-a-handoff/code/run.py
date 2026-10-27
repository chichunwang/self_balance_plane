"""命令列入口。MuJoCo 相關指令要用專案的 .venv 執行（.venv/Scripts/python run.py ...）。

  python run.py catch              從空中丟下 → 緩衝接住 → 平衡
  python run.py juggle             丟下 → 連續拍球 → 緩衝 → 平衡
  python run.py compare            有無壓電片、DS3218 vs MG996R、能拍多高的對照表
  python run.py draw               機構圖，兩套控制（樹莓派、Teensy）的系統架構圖與接線圖
  python run.py mjcheck            MuJoCo 機構動力學驗證（報告、3D 動畫、互動模型）
  python run.py tosscmp            收合改版前後的側拋對照（第一版臂朝內 Ø220 vs 改版臂朝外 Ø198）
  python run.py mjview             匯出 MuJoCo 互動模型並顯示開啟方式
  python run.py live               即時控制模式：控制器在 MuJoCo 視窗裡即時接球／拍球，可按鍵丟球、側拋，
                                   也能用滑鼠推球（--juggle 一開始就拍球、--speed 0.5 慢動作）

常用選項：--plant mj  --servo MG996R  --no-piezo  --apex 0.2  --e 0.85  --trim 1 -1 0.5  --seed 3  --gif
輸出都在 out/。
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from ppsim.params import SERVOS, Config, PlantErrors
from ppsim.sim import Scenario, simulate

OUT = Path(__file__).parent / 'out'
WORKERS = max(1, (os.cpu_count() or 2) - 1)


def make_cfg(servo='DS3218', piezo=True, apex=None, e=None, **over):
    """over：'區塊.欄位' → 值，例如 {'mech.backlash_deg': 1.0}"""
    cfg = Config()
    cfg.servo = replace(SERVOS[servo])
    cfg.piezo.enabled = piezo
    if apex is not None:
        cfg.control.apex = apex
    if e is not None:
        cfg.surface.e = e
    for key, val in over.items():
        part, attr = key.split('.')
        setattr(getattr(cfg, part), attr, val)
    return cfg


def random_scenario(kind, seed, juggle_time=8.0):
    """丟球高度 20–35 cm、位置 ±4 cm、水平初速 ±8 cm/s 隨機"""
    rng = np.random.default_rng(100 + seed)
    xy = tuple(rng.uniform(-0.04, 0.04, 2))
    vxy = tuple(rng.uniform(-0.08, 0.08, 2))
    h = rng.uniform(0.20, 0.35)
    if kind == 'catch':
        return Scenario('catch', duration=4.0, drop_height=h, drop_xy=xy, drop_vxy=vxy, seed=seed)
    return Scenario('juggle', duration=juggle_time + 4.0, drop_height=h, drop_xy=xy, drop_vxy=vxy,
                    juggle_until=juggle_time, seed=seed)


TOSS_DISTS = (0.3, 0.45, 0.6)


def toss_scenario(cfg, seed, dist):
    """從旁邊拋弧線：從正面或背面（90° 或 270° 左右各 45° 以內；0°、180° 是盪鞦韆相機架的腳），距離 dist，
    最高點離板面 35 cm，落點在中心 ±2 cm。2026-10-26 以前是 360° 隨機方向，數字不能跟舊報告直接比。"""
    from ppsim.sim import toss
    rng = np.random.default_rng(500 + seed)
    land = rng.uniform(-0.02, 0.02, 2)
    az = (90.0 if rng.uniform() < 0.5 else 270.0) + rng.uniform(-45.0, 45.0)
    return toss(cfg, land, az, dist, apex=0.35, launch_h=0.25, duration=5.0, seed=seed)


def _job(args):
    cfg_kw, kind, seed, juggle_time, plant = args
    cfg = make_cfg(**cfg_kw)
    if kind.startswith('toss'):
        sc = toss_scenario(cfg, seed, float(kind.split(':')[1]))
    else:
        sc = random_scenario(kind, seed, juggle_time)
    res = simulate(cfg, sc, plant=plant)
    m = res.metrics
    m['caught'] = m['fell'] is None and m.get('t_balance') is not None and m.get('final_r', 1.0) < 0.02
    strong = [i for i in res.impacts if -i['vn'] > 0.1]
    dts = [abs(min((i['t'] for i in strong), key=lambda x: abs(x - h['t_hit'])) - h['t_hit'])
           for h in res.hits if strong]
    m['dt_max_ms'] = max(dts) * 1e3 if dts else None
    return m


def _run_jobs(jobs):
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(_job, [j[0] for j in jobs]))
    groups = {}
    for (_, tag), m in zip(jobs, results):
        groups.setdefault(tag, []).append(m)
    return groups


def _fmt(v, f='{:.2f}'):
    return '—' if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v)


def _ok(ms, need_balance=True):
    good = [m for m in ms if m['fell'] is None and (not need_balance or 't_balance' in m)]
    return f'{len(good)}/{len(ms)}'


def _vals(ms, key, scale=1.0):
    return [m[key] * scale for m in ms if m.get(key) is not None and m['fell'] is None]


def _row(name, c, j, extra=()):
    tb, ap, lr = _vals(c, 't_balance'), _vals(j, 'apex_mean', 100), _vals(j, 'land_r_max', 1000)
    dt = [m['dt_max_ms'] for m in c + j if m.get('dt_max_ms') is not None]
    cells = [name, _ok(c), _fmt(np.mean(tb) if tb else None, '{:.2f} s'), _ok(j, need_balance=False),
             _fmt(np.mean(ap) if ap else None, '{:.1f} cm'), _fmt(max(lr) if lr else None, '{:.0f} mm'),
             _fmt(max(dt) if dt else None, '{:.0f} ms')] + list(extra)
    return '| ' + ' | '.join(cells) + ' |'


def cmd_compare(n_seeds):
    variants = [('DS3218＋壓電片', dict(servo='DS3218', piezo=True)),
                ('DS3218、沒有壓電片', dict(servo='DS3218', piezo=False)),
                ('MG996R＋壓電片', dict(servo='MG996R', piezo=True)),
                ('MG996R、沒有壓電片', dict(servo='MG996R', piezo=False))]
    apexes = [0.15, 0.20, 0.25, 0.30]
    jobs = [((kw, kind, s, 8.0, 'py'), (name, kind)) for name, kw in variants for kind in ('catch', 'juggle')
            for s in range(1, n_seeds + 1)]
    jobs += [((dict(apex=ap), 'juggle', s, 6.0, 'py'), (ap, 'juggle')) for ap in apexes for s in range(1, n_seeds + 1)]
    lens = [('100° 廣角鏡頭（預設）', {}), ('60° 鏡頭', {'camera.hfov_deg': 60.0})]
    jobs += _toss_jobs(lens, 'py', n_seeds)
    t0 = time.time()
    g = _run_jobs(jobs)
    lines = [f'# 對照實驗（純 Python 模擬，每格 {n_seeds} 組隨機丟球；耗時 {time.time() - t0:.0f} s）', '',
             '| 設定 | 接球成功 | 接到平衡耗時 | 拍球成功 | 拍球高度（目標 15 cm） | 落點離中心最多 | 撞擊時間誤差最大 | 伺服最高 |',
             '|---|---|---|---|---|---|---|---|']
    for name, _ in variants:
        c, j = g[(name, 'catch')], g[(name, 'juggle')]
        lines.append(_row(name, c, j, [f"{max(m['servo_peak'] for m in c + j) * 100:.0f}%"]))
    lines += ['', _apex_table(g, apexes, '純 Python'), '', _toss_table(g, lens, '純 Python')]
    _write('compare.md', lines)


def _toss_jobs(variants, plant, n_seeds):
    return [((kw, f'toss:{d}', s, 0.0, plant), ('toss', name, d)) for name, kw in variants for d in TOSS_DISTS
            for s in range(1, n_seeds + 1)]


def _toss_table(g, variants, plant):
    cfg = Config()
    vh = [float(np.hypot(*toss_scenario(cfg, 1, d).start_v[:2])) for d in TOSS_DISTS]
    lines = [f'## 側拋（{plant}；從正面或背面拋弧線，最高點離板面 35 cm，落點在中心 ±2 cm）', '',
             '| 設定 | ' + ' | '.join(f'出手距離 {d * 100:.0f} cm（水平 {v:.2f} m/s）' for d, v in zip(TOSS_DISTS, vh)) + ' |',
             '|---' * (len(TOSS_DISTS) + 1) + '|']
    for name, _ in variants:
        cells = []
        for d in TOSS_DISTS:
            ms = g[('toss', name, d)]
            cells.append(f"{sum(m['caught'] for m in ms)}/{len(ms)} 接住")
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def _apex_table(g, apexes, plant):
    lines = [f'## 能拍多高（{plant}，DS3218＋壓電片，拍 6 秒）', '',
             '| 目標高度 | 成功 | 實際高度 | 高度標準差 | 落點離中心最多 | 伺服最高 |', '|---|---|---|---|---|---|']
    for ap in apexes:
        j = g[(ap, 'juggle')]
        good = [m for m in j if m['fell'] is None and 'apex_mean' in m]
        lines.append(f"| {ap * 100:.0f} cm | {_ok(j, need_balance=False)} | "
                     f"{_fmt(np.mean([m['apex_mean'] for m in good]) * 100 if good else None, '{:.1f} cm')} | "
                     f"{_fmt(np.mean([m['apex_std'] for m in good]) * 1000 if good else None, '{:.1f} mm')} | "
                     f"{_fmt(max(m['land_r_max'] for m in good) * 1000 if good else None, '{:.0f} mm')} | "
                     f"{max(m['servo_peak'] for m in j) * 100:.0f}% |")
    return '\n'.join(lines)


def _write(name, lines):
    text = '\n'.join(lines)
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(text + '\n', encoding='utf-8')
    print(text)


def cmd_mjcheck(n_seeds):
    from ppsim import mjcheck, viz
    from ppsim.mjplant import calibrate_contact
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    cfg = Config()
    dp, dn = mjcheck.geometry_check(cfg)
    damping, e_table = calibrate_contact(cfg)
    roll5, roll10 = mjcheck.rolling_check(cfg, 5.0), mjcheck.rolling_check(cfg, 10.0)

    trim = dict(**{'errors.trim_deg': (1.0, -1.0, 0.5)})
    no_learn = {'control.tilt_learn': 0.0, 'control.bal_ki': 0.0, 'control.apex_learn': 0.0}
    sens = [('基準（DS3218）', {}),
            ('伺服加速慢一倍（馬達時間常數 30 ms）', {'servo.t_mech': 0.030}),
            ('齒隙 1°', {'mech.backlash_deg': 1.0}),
            ('伺服 trim 偏 (+1°, −1°, +0.5°)，不做自動校正', {**trim, **no_learn}),
            ('伺服 trim 偏 (+1°, −1°, +0.5°)，自動水平校正', trim),
            ('平台重一倍', {'mech.plate_density': 2380.0}),
            ('MG996R', {'servo': 'MG996R'})]
    apexes = [0.15, 0.20, 0.25, 0.30]
    jobs = []
    for plant in ('py', 'mj'):
        for kind in ('catch', 'juggle'):
            jobs += [(({}, kind, s, 8.0, plant), ('plant', plant, kind)) for s in range(1, n_seeds + 1)]
    for name, over in sens[1:]:
        kw = {'servo': over['servo']} if 'servo' in over else {}
        kw.update({k: v for k, v in over.items() if k != 'servo'})
        for kind in ('catch', 'juggle'):
            jobs += [((kw, kind, s, 6.0, 'mj'), (name, kind)) for s in range(1, n_seeds + 1)]
    jobs += [((dict(apex=ap), 'juggle', s, 6.0, 'mj'), (ap, 'juggle')) for ap in apexes for s in range(1, 3)]
    jobs += _toss_jobs([('DS3218＋100° 廣角（預設）', {})], 'mj', n_seeds)
    g = _run_jobs(jobs)
    g[(sens[0][0], 'catch')] = g[('plant', 'mj', 'catch')]
    g[(sens[0][0], 'juggle')] = g[('plant', 'mj', 'juggle')]

    base = g[('plant', 'mj', 'catch')] + g[('plant', 'mj', 'juggle')]
    lines = ['# MuJoCo 機構動力學驗證', '',
             f'每格 {n_seeds} 組隨機丟球（高度 20–35 cm、位置 ±4 cm、水平初速 ±8 cm/s）；耗時 {time.time() - t0:.0f} s。',
             '控制器、估測器、感測器完全相同，只把「真實世界」從純 Python 換成 MuJoCo 多體動力學'
             '（閉鏈 3-RRS、直流馬達伺服＋齒輪剛性、軟接觸、空氣阻力）。', '',
             '## 1. 模型一致性', '',
             f'- 幾何：伺服鎖在同一組角度，MuJoCo 閉鏈算出的平台和運動學正解比，位置最多差 {dp:.2f} mm、'
             f'法向量最多差 {dn:.3f}°（20 組隨機姿態，差異來自約束的軟度與重力下垂）',
             f'- 接觸：阻尼由 e=0.87 反推（{damping:.0f}）；撞擊速度 0.3／1／2／3 m/s 時的 e＝'
             + '／'.join(f'{v:.3f}' for v in e_table.values()),
             f'- 滾動：斜面加速度是薄殼球理論值 0.6·g·sinθ 的 {roll5 * 100:.1f}%（5°）、{roll10 * 100:.1f}%（10°），'
             '少掉的部分是設定的滾動阻力', '',
             '## 2. 同一套控制器，兩種真實世界', '',
             '| 真實世界 | 接球成功 | 接到平衡耗時 | 拍球成功 | 拍球高度（目標 15 cm） | 落點離中心最多 | 撞擊時間誤差最大 |',
             '|---|---|---|---|---|---|---|',
             _row('純 Python', g[('plant', 'py', 'catch')], g[('plant', 'py', 'juggle')]),
             _row('MuJoCo', g[('plant', 'mj', 'catch')], g[('plant', 'mj', 'juggle')]), '',
             '## 3. 機構負載（MuJoCo 基準，所有跑次的最大值）', '',
             '| 項目 | 數值 | 意義 |', '|---|---|---|',
             f"| 馬達扭力峰值 | {max(m['torque_peak'] for m in base) * 100:.0f}% 堵轉 | DS3218 6 V 堵轉約 2.0 N·m |",
             f"| 馬達扭力 RMS | {max(m['torque_rms'] for m in base) * 100:.0f}% 堵轉 | 長時間發熱看這個 |",
             f"| duty 飽和時間 | {max(m['duty_sat'] for m in base) * 100:.1f}% | 驅動全開的時間比例 |",
             f"| 轉速峰值 | {max(m['servo_peak'] for m in base) * 100:.0f}% 空載轉速 | 瓶頸在這裡 |",
             f"| 球頭最大受力 | {max(m['rod_force_max'] for m in base):.1f} N | 平台重量＋加速＋球撞擊 |", '',
             f'## 4. 敏感度（MuJoCo，每種狀況接球＋拍球各 {n_seeds} 組；拍球 6 秒）', '',
             '| 狀況 | 接球成功 | 接到平衡耗時 | 拍球成功 | 拍球高度 | 落點離中心最多 | 撞擊時間誤差最大 | 落點系統性偏移 | 平衡最後離中心 | 扭力峰值 | 轉速峰值 |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for name, _ in sens:
        c, j = g[(name, 'catch')], g[(name, 'juggle')]
        off, fr = _vals(j, 'land_offset', 1000), _vals(c, 'final_r', 1000)
        lines.append(_row(name, c, j, [
            _fmt(np.mean(off) if off else None, '{:.1f} mm'), _fmt(np.mean(fr) if fr else None, '{:.1f} mm'),
            f"{max(m['torque_peak'] for m in c + j) * 100:.0f}%", f"{max(m['servo_peak'] for m in c + j) * 100:.0f}%"]))
    lines += ['', _apex_table(g, apexes, 'MuJoCo'), '', _toss_table(g, [('DS3218＋100° 廣角（預設）', {})], 'MuJoCo')]

    # 一次詳細的拍球：時間曲線＋3D 動畫；互動模型
    res = simulate(cfg, Scenario('juggle', duration=7.0, juggle_until=4.0, seed=1), plant='mj', frames_every=1 / 20)
    viz.plot_run(res, OUT / 'mj_juggle_timeline.png')
    mjcheck.render_gif(res, OUT / 'mj_juggle.gif')
    mjcheck.export_viewer_model(OUT / 'pingpong.xml', cfg)
    if res.e_log:
        lines += ['', '## 5. 壓電片自動校正 e（上面那次詳細拍球）', '',
                  f'控制器起始以為 e={cfg.control.e_model}，每跳用飛行時間反推，'
                  f'{len(res.e_log)} 跳後估到 {res.e_log[-1][2]:.3f}（MuJoCo 接觸實際約 {e_table[2.0]:.3f}）。']
    lines += ['', '## 輸出', '', '- `mj_juggle_timeline.png`：MuJoCo 拍球 4 秒→緩衝→平衡（含馬達扭力、球頭受力）',
              '- `mj_juggle.gif`：同一次的 3D 動畫',
              '- `pingpong.xml`：MuJoCo 互動模型（`.venv/Scripts/python -m mujoco.viewer --mjcf=out/pingpong.xml`）']
    _write('mujoco_report.md', lines)


def cmd_tosscmp(n_seeds):
    """收合改版前後的側拋：同一組拋球，第一版（臂朝內、Ø220）對改版（臂朝外、Ø198），純 Python 和 MuJoCo 各跑一次。"""
    t0 = time.time()
    geoms = [('第一版（臂朝內、平台 Ø220）', {'geom.arm_outward': False, 'geom.plate_diameter': 0.220}),
             ('改版（臂朝外、平台 Ø198）', {})]
    jobs = [((kw, f'toss:{d}', s, 0.0, plant), ('toss', name, d, plant))
            for plant in ('py', 'mj') for name, kw in geoms for d in TOSS_DISTS for s in range(1, n_seeds + 1)]
    g = _run_jobs(jobs)
    lines = [f'# 收合改版前後的側拋（每格 {n_seeds} 組；耗時 {time.time() - t0:.0f} s）', '']
    for plant, label in (('py', '純 Python'), ('mj', 'MuJoCo')):
        lines += [_toss_table({('toss', n, d): g[('toss', n, d, plant)] for n, _ in geoms for d in TOSS_DISTS},
                              geoms, label), '']
    _write('toss_compare.md', lines)


def cmd_run(args):
    from ppsim import viz
    cfg = make_cfg(args.servo, not args.no_piezo, args.apex, args.e)
    if args.trim:
        cfg.errors = PlantErrors(trim_deg=tuple(args.trim))
    if args.cmd == 'catch':
        sc = Scenario('catch', duration=4.0, seed=args.seed)
    else:
        sc = Scenario('juggle', duration=args.juggle_time + 4.0, juggle_until=args.juggle_time, seed=args.seed)
    t0 = time.time()
    res = simulate(cfg, sc, plant=args.plant, frames_every=1 / 20 if args.gif and args.plant == 'mj' else 0.0)
    print(f'模擬 {sc.duration:.0f} 秒（{res.plant}），耗時 {time.time() - t0:.1f} 秒')
    for k, v in res.metrics.items():
        print(f'  {k}: {v}')
    OUT.mkdir(exist_ok=True)
    stem = (f'{args.cmd}{"_mj" if args.plant == "mj" else ""}{"_" + args.servo if args.servo != "DS3218" else ""}'
            f'{"_nopiezo" if args.no_piezo else ""}')
    viz.plot_run(res, OUT / f'{stem}_timeline.png')
    viz.plot_top(res, OUT / f'{stem}_top.png')
    if args.gif:
        if args.plant == 'mj':
            from ppsim import mjcheck
            mjcheck.render_gif(res, OUT / f'{stem}.gif')
        else:
            viz.animate(res, OUT / f'{stem}.gif')
    print(f'圖已存到 {OUT}')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description='三伺服乒乓球平台模擬')
    ap.add_argument('cmd', choices=['catch', 'juggle', 'compare', 'draw', 'mjcheck', 'tosscmp', 'mjview', 'live'])
    ap.add_argument('--juggle', action='store_true', help='live：一開始就拍球')
    ap.add_argument('--speed', type=float, default=1.0, choices=[1.0, 0.5, 0.25], help='live：播放速度')
    ap.add_argument('--plant', default='py', choices=['py', 'mj'], help='真實世界用純 Python 或 MuJoCo')
    ap.add_argument('--servo', default='DS3218', choices=list(SERVOS))
    ap.add_argument('--no-piezo', action='store_true')
    ap.add_argument('--apex', type=float, help='拍球目標高度（m）')
    ap.add_argument('--e', type=float, help='真實恢復係數（預設 0.87）')
    ap.add_argument('--trim', type=float, nargs=3, help='三顆伺服的中位誤差（度），模擬沒校正好')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--juggle-time', type=float, default=8.0)
    ap.add_argument('--gif', action='store_true', help='另外輸出動畫')
    ap.add_argument('--seeds', type=int, default=4, help='compare／mjcheck 每格跑幾組')
    args = ap.parse_args()
    if args.cmd == 'compare':
        cmd_compare(args.seeds)
    elif args.cmd == 'mjcheck':
        cmd_mjcheck(args.seeds)
    elif args.cmd == 'tosscmp':
        cmd_tosscmp(args.seeds)
    elif args.cmd == 'live':
        from ppsim.live import run_live
        try:
            run_live(make_cfg(args.servo, not args.no_piezo, args.apex, args.e), juggle=args.juggle, speed=args.speed)
        except KeyboardInterrupt:                   # 終端機按 Ctrl+C：視窗已經關好了，安靜結束
            pass
    elif args.cmd == 'mjview':
        from ppsim import mjcheck
        OUT.mkdir(exist_ok=True)
        mjcheck.export_viewer_model(OUT / 'pingpong.xml')
        print(f'已匯出 {OUT / "pingpong.xml"}，開啟：.venv/Scripts/python -m mujoco.viewer --mjcf=out/pingpong.xml')
    elif args.cmd == 'draw':
        from ppsim import draw
        OUT.mkdir(exist_ok=True)
        draw.mechanism(OUT / 'mechanism.png')
        draw.system(OUT / 'system.png')
        draw.system(OUT / 'system_teensy.png', scheme='teensy')
        draw.wiring_teensy(OUT / 'wiring_teensy.png')
        draw.wiring_rpi(OUT / 'wiring_rpi.png')
        print(f'圖已存到 {OUT}')
    else:
        cmd_run(args)


if __name__ == '__main__':
    main()
