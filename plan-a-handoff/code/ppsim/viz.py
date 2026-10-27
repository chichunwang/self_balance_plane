"""模擬結果的圖：時間曲線、俯視落點、動畫。"""
import math

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'Noto Sans CJK TC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

MODE_NAME = {'JUGGLE': '拍球', 'CUSHION': '緩衝', 'SETTLE': '下沉', 'BALANCE': '平衡'}
MODE_COLOR = {'JUGGLE': '#f6dfb8', 'CUSHION': '#d4e6f4', 'SETTLE': '#e6dcf1', 'BALANCE': '#d8eed5'}
BALL = '#e8751a'
PLATE = '#3b4a5a'


def _spans(res):
    ev = res.events + [(res.log['t'][-1], None)]
    return [(a[0], b[0], a[1]) for a, b in zip(ev[:-1], ev[1:])]


def _shade(ax, res, label=False):
    seen = set()
    for t0, t1, m in _spans(res):
        lab = MODE_NAME[m] if label and m not in seen else None
        seen.add(m)
        ax.axvspan(t0, t1, color=MODE_COLOR[m], alpha=0.6, lw=0, label=lab)


def _summary(res):
    m = res.metrics
    parts = [f"撞擊 {m['bounces']} 次"]
    if 'apex_mean' in m:
        parts.append(f"拍球高度 {m['apex_mean'] * 100:.1f} cm（±{m['apex_std'] * 1000:.1f} mm）")
        parts.append(f"落點離中心最多 {m['land_r_max'] * 1000:.0f} mm")
    if 't_balance' in m:
        parts.append(f"{m['t_balance']:.2f} s 進入平衡，最後離中心 {m['final_r'] * 1000:.1f} mm")
    parts.append(f"伺服最高 {m['servo_peak'] * 100:.0f}% 極速")
    if 'torque_peak' in m:
        parts.append(f"扭力最高 {m['torque_peak'] * 100:.0f}% 堵轉")
    if m['fell'] is not None:
        parts.append(f"⚠ {m['fell']:.2f} s 掉出平台")
    half = (len(parts) + 1) // 2
    return '　'.join(parts[:half]) + '\n' + '　'.join(parts[half:])


def plot_run(res, path):
    L, cfg = res.log, res.cfg
    t = L['t']
    mj = 'torque' in L
    ratios = [2.2, 1.2, 1, 1] + ([1, 1] if mj else [])
    fig, axs = plt.subplots(len(ratios), 1, figsize=(11, 10 + (3.6 if mj else 0)), sharex=True,
                            gridspec_kw=dict(height_ratios=ratios))
    ax = axs[0]
    _shade(ax, res, label=True)
    ax.plot(t, L['gap'] * 1e3, color=BALL, lw=1.3, label='球底離板面')
    ax.plot(t, (L['c'][:, 2] - cfg.geom.z_joint0) * 1e3, color=PLATE, lw=1.1, label='平台升降')
    for h in res.hits:
        ax.axvline(h['t_hit'], color='#999', lw=0.4, ls=':')
    ax.set_ylabel('高度（mm）')
    ax.legend(loc='upper right', ncol=6, fontsize=9)
    ax.set_title(f"{res.sc.name}｜{res.plant}｜{cfg.servo.name}｜壓電片 {'有' if cfg.piezo.enabled else '無'}\n"
                 f"{_summary(res)}", fontsize=11, loc='left')

    ax = axs[1]
    _shade(ax, res)
    ax.plot(t, L['p'][:, 0] * 1e3, lw=1.0, color='#1f77b4', label='x（真實）')
    ax.plot(t, L['p'][:, 1] * 1e3, lw=1.0, color='#ff7f0e', label='y（真實）')
    if res.est_track:
        te = np.array([a for a, _, _ in res.est_track])
        xe = np.array([x[:2] for _, x, _ in res.est_track]) * 1e3
        ax.plot(te, xe[:, 0], lw=0.8, ls='--', color='#1f77b4', alpha=0.8, label='x（估測）')
        ax.plot(te, xe[:, 1], lw=0.8, ls='--', color='#ff7f0e', alpha=0.8, label='y（估測）')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel('球水平位置（mm）')
    ax.legend(loc='upper right', fontsize=8.5, ncol=2)

    ax = axs[2]
    _shade(ax, res)
    n = L['n']
    ax.plot(t, np.degrees(np.arctan2(np.hypot(n[:, 0], n[:, 1]), n[:, 2])), color=PLATE, lw=1.0)
    ax.set_ylabel('平台傾角（°）')

    ax = axs[3]
    _shade(ax, res)
    w_lim = cfg.servo.w_noload if mj else cfg.servo.w_max
    ax.plot(t, np.abs(L['om']) / w_lim * 100, lw=0.8)
    ax.axhline(100, color='#c0392b', ls='--', lw=0.8)
    ax.set_ylabel('伺服轉速（%極速）')
    ax.set_ylim(0, 110)
    if mj:
        ax = axs[4]
        _shade(ax, res)
        ax.plot(t, L['torque'] / cfg.servo.stall_torque * 100, lw=0.7)
        ax.axhline(100, color='#c0392b', ls='--', lw=0.8)
        ax.axhline(-100, color='#c0392b', ls='--', lw=0.8)
        ax.set_ylabel('馬達扭力（%堵轉）')
        ax.set_ylim(-110, 110)
        ax = axs[5]
        _shade(ax, res)
        ax.plot(t, L['rod_force'], lw=0.7)
        ax.set_ylabel('球頭受力（N）')
    axs[-1].set_xlabel('時間（s）')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_top(res, path):
    cfg = res.cfg
    R = cfg.geom.plate_diameter / 2 * 1e3
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.add_patch(Circle((0, 0), R, fc='#eef3f7', ec=PLATE, lw=1.5))
    for d in cfg.geom.legs_deg:
        a = math.radians(d)
        ax.plot(cfg.geom.r_p * 1e3 * math.cos(a), cfg.geom.r_p * 1e3 * math.sin(a), 'o', color=PLATE, ms=6)
    strong = [i for i in res.impacts if -i['vn'] > 0.1]
    if strong:
        p = np.array([i['p'][:2] for i in strong]) * 1e3
        sc = ax.scatter(p[:, 0], p[:, 1], c=np.arange(len(p)), cmap='viridis', s=22, zorder=3)
        fig.colorbar(sc, ax=ax, shrink=0.7, label='第幾次撞擊')
    L = res.log
    bal = L['mode'] == 3
    if bal.any():
        ax.plot(L['p'][bal, 0] * 1e3, L['p'][bal, 1] * 1e3, color=BALL, lw=1.0, label='平衡時的滾動軌跡')
        ax.legend(loc='lower right', fontsize=9)
    ax.plot(res.sc.drop_xy[0] * 1e3, res.sc.drop_xy[1] * 1e3, 'x', color='#c0392b', ms=10, mew=2)
    ax.annotate('丟球位置', (res.sc.drop_xy[0] * 1e3, res.sc.drop_xy[1] * 1e3), xytext=(8, 8),
                textcoords='offset points', fontsize=9, color='#c0392b')
    ax.set_aspect('equal')
    ax.set_xlim(-R - 10, R + 10)
    ax.set_ylim(-R - 10, R + 10)
    ax.set_xlabel('x（mm）')
    ax.set_ylabel('y（mm）')
    ax.set_title('俯視：每次撞擊的落點', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def animate(res, path, fps=25):
    cfg, L = res.cfg, res.log
    t = L['t']
    Rb = cfg.ball.radius * 1e3
    Rp = cfg.geom.plate_diameter / 2 * 1e3
    zs0 = cfg.geom.z_surface0 * 1e3
    frames = np.searchsorted(t, np.arange(t[0], t[-1], 1 / fps))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
    a1.set_xlim(-140, 140)
    a1.set_ylim(-40, 380)
    a1.set_aspect('equal')
    a1.set_xlabel('x（mm）')
    a1.set_ylabel('高度（離中立板面，mm）')
    a1.set_title('側視', fontsize=10)
    plate_line, = a1.plot([], [], color=PLATE, lw=4, solid_capstyle='round')
    ball_s = Circle((0, 0), Rb, fc=BALL, ec='#9a4a0c')
    a1.add_patch(ball_s)
    a2.add_patch(Circle((0, 0), Rp, fc='#eef3f7', ec=PLATE, lw=1.5))
    ball_t = Circle((0, 0), Rb, fc=BALL, ec='#9a4a0c')
    a2.add_patch(ball_t)
    a2.set_xlim(-Rp - 8, Rp + 8)
    a2.set_ylim(-Rp - 8, Rp + 8)
    a2.set_aspect('equal')
    a2.set_title('俯視', fontsize=10)
    label = a1.text(-132, 355, '', fontsize=11, va='top')

    def draw(i):
        k = frames[i]
        n = L['n'][k]
        s0 = (L['c'][k] + cfg.geom.joint_to_surface * n) * 1e3
        tx = np.array([n[2], -n[0]]) / math.hypot(n[2], n[0])
        plate_line.set_data([s0[0] - Rp * tx[0], s0[0] + Rp * tx[0]],
                            [s0[2] - zs0 - Rp * tx[1], s0[2] - zs0 + Rp * tx[1]])
        p = L['p'][k] * 1e3
        ball_s.center = (p[0], p[2] - zs0)
        ball_t.center = (p[0], p[1])
        mode = ('JUGGLE', 'CUSHION', 'SETTLE', 'BALANCE')[int(L['mode'][k])]
        label.set_text(f"{t[k]:5.2f} s　{MODE_NAME[mode]}")
        return plate_line, ball_s, ball_t, label

    ani = animation.FuncAnimation(fig, draw, frames=len(frames), blit=True)
    fig.tight_layout()
    ani.save(path, writer=animation.PillowWriter(fps=fps), dpi=72)
    plt.close(fig)
