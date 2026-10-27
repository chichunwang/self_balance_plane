"""架構圖：機構（3D＋剖面＋俯視，尺寸直接取自 params）與系統方塊圖。"""
import math

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from .kinematics import Platform  # noqa: E402
from .params import Config  # noqa: E402

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'Noto Sans CJK TC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

WOOD, SERVO, ARM, ROD = '#d9c3a0', '#2f3437', '#e08a2c', '#4a5a6a'
ACRYLIC, BALL, PIEZO, DIM = '#9fd0ea', '#f07f1f', '#d4a82c', '#555555'
SERVO_L, SERVO_W, SERVO_H = 40.0, 20.0, 40.5     # DS3218 本體（mm）


# ---------------------------------------------------------------- 3D
def _box(ax, center, axes, half, color, z=1):
    c = np.asarray(center)
    u, v, w = (np.asarray(a) * h for a, h in zip(axes, half))
    corners = [c + sx * u + sy * v + sz * w for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    idx = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    ax.add_collection3d(Poly3DCollection([[corners[i] for i in f] for f in idx], facecolor=color,
                                         edgecolor='#111', lw=0.3, alpha=0.95, zorder=z))


def _disc(ax, center, n, radius, color, alpha, edge='#333', k=72, z=5):
    n = np.asarray(n) / np.linalg.norm(n)
    a = np.cross(n, [1.0, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    th = np.linspace(0, 2 * np.pi, k, endpoint=False)
    pts = [np.asarray(center) + radius * (math.cos(x) * a + math.sin(x) * b) for x in th]
    ax.add_collection3d(Poly3DCollection([pts], facecolor=color, edgecolor=edge, lw=0.6, alpha=alpha,
                                         zorder=z))


def _sphere(ax, center, r, color):
    u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
    ax.plot_surface(center[0] + r * np.cos(u) * np.sin(v), center[1] + r * np.sin(u) * np.sin(v),
                    center[2] + r * np.cos(v), color=color, linewidth=0, shade=True, zorder=8)


def _draw3d(ax, cfg, dz, tx, ty, title):
    g = cfg.geom
    plat = Platform(g)
    c, R, _ = plat.pose(dz, tx, ty)
    th, _ = plat.ik(c, R)
    mm = 1000.0
    # 關掉自動深度排序、改用固定圖層（底板→伺服→臂→連桿→平台→球），不然大片底板會蓋住前面的伺服
    ax.computed_zorder = False
    h = _base_half(g)
    ax.add_collection3d(Poly3DCollection([[(-h, -h, 0), (h, -h, 0), (h, h, 0), (-h, h, 0)]],
                                         facecolor=WOOD, edgecolor='#8a7550', alpha=0.55, zorder=0))
    tips = plat.arm_tips(th) * mm
    anchors = (c + plat.p_loc @ R.T) * mm
    n = R[:, 2]
    for i in range(3):
        er, et = plat.er[i], plat.et[i]
        shaft = plat.shaft[i] * mm
        _box(ax, shaft - g.arm_sign * et * (SERVO_H / 2 + 3) - np.array([0, 0, 10]), (er, et, [0, 0, 1]),
             (SERVO_W / 2, SERVO_H / 2, SERVO_L / 2), SERVO, z=1)
        ax.plot(*zip(shaft, tips[i]), color=ARM, lw=5, solid_capstyle='round', zorder=2)
        ax.plot(*zip(tips[i], anchors[i]), color=ROD, lw=3, zorder=3)
        ax.scatter(*anchors[i], color='#222', s=25, depthshade=False, zorder=3)
        top = anchors[i] + n * (g.joint_to_surface - g.plate_thickness) * mm
        ax.plot(*zip(anchors[i], top), color='#222', lw=2, zorder=3)
    s0 = (c + g.joint_to_surface * n) * mm
    _disc(ax, s0 - n * (g.plate_thickness * mm + 1), n, cfg.piezo.radius * mm, PIEZO, 0.95, z=4)
    _disc(ax, s0 - n * g.plate_thickness * mm, n, g.plate_diameter / 2 * mm, ACRYLIC, 0.25, z=5)
    _disc(ax, s0, n, g.plate_diameter / 2 * mm, ACRYLIC, 0.4, z=6)
    _sphere(ax, s0 + n * cfg.ball.radius * mm, cfg.ball.radius * mm, BALL)
    ax.set_xlim(-160, 160)
    ax.set_ylim(-160, 160)
    ax.set_zlim(0, 210)
    ax.set_box_aspect((320, 320, 210), zoom=1.2)
    ax.view_init(elev=20, azim=-62)
    ax.set_xticks([-150, 0, 150])
    ax.set_yticks([-150, 0, 150])
    ax.set_zticks([0, 100, 200])
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=11)


# ---------------------------------------------------------------- 2D 標尺
def _dim(ax, p1, p2, text, off=(0, 0), fs=9, rot=0):
    ax.annotate('', xy=p2, xytext=p1, arrowprops=dict(arrowstyle='<->', color=DIM, lw=0.9))
    mx, my = (p1[0] + p2[0]) / 2 + off[0], (p1[1] + p2[1]) / 2 + off[1]
    ax.text(mx, my, text, ha='center', va='center', fontsize=fs, color='#222', rotation=rot,
            bbox=dict(fc='white', ec='none', pad=1.2))


def _base_half(g):
    """圖上底板畫多大（mm，半邊）：蓋得住平台和伺服就好。"""
    return max(g.plate_diameter / 2, g.r_b + 0.040) * 1000.0


def _section(ax, cfg):
    g = cfg.geom
    mm = 1000.0
    s = g.arm_sign
    rp, rb, a, d = g.r_p * mm, g.r_b * mm, g.arm * mm, g.rod * mm
    zs, zj = g.z_shaft * mm, g.z_joint0 * mm
    zsurf, tpl = g.z_surface0 * mm, g.plate_thickness * mm
    R, rball = g.plate_diameter / 2 * mm, cfg.ball.radius * mm
    tipx = rb - s * a                                                                      # 肘關節
    h = _base_half(g)
    ax.add_patch(Rectangle((-h, -8), 2 * h, 8, fc=WOOD, ec='#8a7550'))
    ax.add_patch(Rectangle((rb - 12, 0), 24, zs - 30, fc='#c8ccd0', ec='#888'))           # 支架
    ax.add_patch(Rectangle((rb - SERVO_W / 2, zs - 30), SERVO_W, SERVO_L, fc=SERVO, ec='k'))
    ax.text(rb, zs + 18, 'DS3218 伺服', fontsize=8, ha='center')
    ax.plot([rb, tipx], [zs, zs], color=ARM, lw=6, solid_capstyle='round')
    ax.add_patch(Circle((rb, zs), 3.2, fc='white', ec='k', zorder=5))
    ax.add_patch(Circle((tipx, zs), 2.8, fc='white', ec='k', zorder=5))
    ax.plot([rp, rp], [zs, zj], color=ROD, lw=4)
    ax.add_patch(Circle((rp, zj), 4.5, fc='#222', zorder=5))
    ax.add_patch(Rectangle((rp - 5, zj), 10, zsurf - tpl - zj, fc='#666', ec='k'))       # 球頭座
    ax.add_patch(Rectangle((-R, zsurf - tpl), 2 * R, tpl, fc=ACRYLIC, ec='#3d7aa0', alpha=0.8))
    pr = cfg.piezo.radius * mm
    ax.add_patch(Rectangle((-pr, zsurf - tpl - 2.5), 2 * pr, 2.5, fc=PIEZO, ec='k'))
    ax.add_patch(Circle((0, zsurf + rball), rball, fc=BALL, ec='#9a4a0c'))
    ax.plot([0, 0], [-8, 290], ls='-.', color='#999', lw=0.8)
    # 標尺
    _dim(ax, (min(rb, tipx), zs - 16), (max(rb, tipx), zs - 16), f'臂 {a:.0f}')
    _dim(ax, (rp + 17, zs), (rp + 17, zj), f'連桿 {d:.0f}', rot=90)
    _dim(ax, (0, zj - 22), (rp, zj - 22), f'球頭半徑 {rp:.0f}')
    _dim(ax, (0, -22), (rb, -22), f'伺服軸半徑 {rb:.0f}')
    xz = rb + 32 if s > 0 else rb - 22                                                     # 標在伺服沒有臂的那一側
    _dim(ax, (xz, 0), (xz, zs), f'{zs:.0f}', off=(9 * s, 0))
    _dim(ax, (-R - 16, 0), (-R - 16, zsurf), f'板面高 {zsurf:.0f}', rot=90)
    _dim(ax, (-R, zsurf + 55), (R, zsurf + 55), f'平台 Ø{2 * R:.0f}（{tpl:.0f} mm 壓克力）')
    ax.text(0, 248, f'▲ 相機在板面上方 {cfg.camera.height * mm:.0f} mm，往下拍', ha='center', fontsize=9,
            color='#333')
    ax.text(18, zsurf - tpl - 16, '壓電片', fontsize=8, color='#7a5c0a')
    ax.text(rp - 8, zj - 6, '球頭（S）', fontsize=8, ha='right')
    if s > 0:
        ax.text(tipx - 4, zs + 8, '肘：轉軸（R）', fontsize=8, ha='right')
    else:
        ax.text(tipx + 6, zs - 12, '肘：轉軸（R）', fontsize=8, ha='left')
    ax.text(-165, 292, f'中立姿態：臂水平朝{"內" if s > 0 else "外"}、連桿垂直\n另外兩隻腳在 ±120°，構造相同',
            fontsize=8.5, va='top', color='#333')
    ax.set_xlim(-170, 175)
    ax.set_ylim(-35, 290)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('單腳剖面（徑向平面，mm）', fontsize=11)


def _top(ax, cfg):
    g = cfg.geom
    mm = 1000.0
    plat = Platform(g)
    R = g.plate_diameter / 2 * mm
    s = g.arm_sign
    h = _base_half(g)
    ax.add_patch(Rectangle((-h, -h), 2 * h, 2 * h, fc=WOOD, ec='#8a7550', alpha=0.5))
    for i, deg in enumerate(g.legs_deg):
        er, et = plat.er[i][:2], plat.et[i][:2]
        shaft = plat.shaft[i][:2] * mm
        body_c = shaft - s * et * (SERVO_H / 2 + 3)
        corners = [body_c + sx * er * SERVO_W / 2 + sy * et * SERVO_H / 2 for sx, sy in
                   ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        ax.add_patch(Polygon(corners, fc=SERVO, ec='k'))
        tip = shaft - s * er * g.arm * mm
        ax.plot([shaft[0], tip[0]], [shaft[1], tip[1]], color=ARM, lw=5, solid_capstyle='round')
        ax.add_patch(Circle(tip, 4.5, fc='#222', zorder=5))
        lab = shaft + er * 26 if s > 0 else tip + er * 16
        ax.text(lab[0], lab[1], f'腳{i + 1}', ha='center', va='center', fontsize=9)
    ax.add_patch(Circle((0, 0), R, fc=ACRYLIC, ec='#3d7aa0', alpha=0.35, lw=1.5))
    ax.add_patch(Circle((0, 0), g.r_p * mm, fill=False, ls='--', ec='#333', lw=0.8))
    ax.add_patch(Circle((0, 0), g.r_b * mm, fill=False, ls=':', ec='#333', lw=0.8))
    ax.add_patch(Circle((0, 0), cfg.piezo.radius * mm, fc=PIEZO, ec='k', alpha=0.9))
    ax.add_patch(Circle((0, 0), cfg.ball.radius * mm, fc='none', ec=BALL, lw=2))
    for r, ang, lab in ((g.r_p * mm, 150, '球頭 R'), (g.r_b * mm, 30, '伺服軸 R')):
        ax.text(r * math.cos(math.radians(ang)), r * math.sin(math.radians(ang)) + 4, f'{lab}{r:.0f}',
                fontsize=8, ha='center', bbox=dict(fc='white', ec='none', pad=0.8, alpha=0.8))
    ax.text(0, -R - 14, f'平台 Ø{2 * R:.0f}', ha='center', fontsize=9)
    ax.text(-162, -158, '三隻腳相隔 120°', fontsize=9)
    ax.set_xlim(-170, 170)
    ax.set_ylim(-170, 170)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('俯視（mm）', fontsize=11)


def mechanism(path, cfg=None):
    cfg = cfg or Config()
    fig = plt.figure(figsize=(14, 12))
    a1 = fig.add_subplot(2, 2, 1, projection='3d')
    _draw3d(a1, cfg, 0.0, 0.0, 0.0, '中立姿態')
    a2 = fig.add_subplot(2, 2, 2, projection='3d')
    tilt = math.tan(math.radians(6))
    _draw3d(a2, cfg, 0.010, tilt * math.cos(math.radians(30)), tilt * math.sin(math.radians(30)),
            '升 10 mm＋傾斜 6°（拍球／導向時）')
    _section(fig.add_subplot(2, 2, 3), cfg)
    _top(fig.add_subplot(2, 2, 4), cfg)
    fig.suptitle('三伺服乒乓球平台（3-RRS）機構架構', fontsize=15, y=0.99)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- 系統方塊圖
def _block(ax, x, y, w, h, title, sub='', fc='#ffffff', ec='#333', tfs=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
                                fc=fc, ec=ec, lw=1.3))
    if not sub:
        ax.text(x + w / 2, y + h / 2, title, ha='center', va='center', fontsize=tfs, weight='bold')
        return
    ax.text(x + w / 2, y + h - 0.16, title, ha='center', va='top', fontsize=tfs, weight='bold')
    ax.text(x + w / 2, y + h - 0.5, sub, ha='center', va='top', fontsize=8.5, color='#444', linespacing=1.3)


def _path(ax, pts, text='', color='#333', ls='-', toff=(0.1, 0)):
    """折線箭頭（最後一段有箭頭）"""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=1.3, ls=ls)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle='-|>', mutation_scale=14, color=color, lw=1.3,
                                 ls=ls))
    if text:
        mid = pts[len(pts) // 2]
        ax.text(mid[0] + toff[0], mid[1] + toff[1], text, fontsize=8.5, color=color, va='center',
                bbox=dict(fc='white', ec='none', pad=0.5))


def _arrow(ax, p1, p2, text='', ls='-', color='#333', off=(0, 0.14), rad=0.0, fs=8.5):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=14, color=color, lw=1.3,
                                 ls=ls, connectionstyle=f'arc3,rad={rad}'))
    if text:
        ax.text((p1[0] + p2[0]) / 2 + off[0], (p1[1] + p2[1]) / 2 + off[1], text, ha='center',
                va='center', fontsize=fs, color=color, bbox=dict(fc='white', ec='none', pad=0.5))


def system(path, scheme='rpi'):
    """兩套並列：scheme='rpi' 是方案 B（樹莓派 4＋PCA9685），'teensy' 是方案 A（筆電算、Teensy 出伺服訊號）。
    目前優先哪一套寫在 README「硬體接法」開頭，圖上不寫。"""
    teensy = scheme == 'teensy'
    g = Config().geom
    fig, ax = plt.subplots(figsize=(16, 9.6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.6)
    ax.axis('off')
    name = '方案 A：筆電＋Teensy' if teensy else '方案 B：樹莓派 4＋PCA9685'
    ax.text(0.2, 9.35, f'系統架構（{name}）：感測 → 估測 → 規劃 → 伺服（括號＝模擬裡對應的模組）',
            fontsize=14, weight='bold')

    # 感測與運算
    _block(ax, 0.3, 6.6, 2.2, 1.3, '相機', 'OV9281 全域快門\n1280×800 @120 fps\n(sensors.Camera)', fc='#eaf4fb')
    ax.add_patch(FancyBboxPatch((3.1, 5.9), 7.2, 2.75, boxstyle='round,pad=0.02,rounding_size=0.15',
                                fc='#f7f7f2', ec='#666', lw=1.2, ls='--'))
    host = '筆電（影像、估測、控制都在這裡跑，跟模擬是同一份 Python）' if teensy else '樹莓派 4（4 GB；影像與控制分開跑在不同核心）'
    ax.text(3.25, 8.45, host, fontsize=11, weight='bold', va='center')
    subs = [('① 影像處理', 'HSV 找橘色球\n球心→x,y\n半徑→高度'),
            ('② 狀態估測', '卡爾曼＋物理模型\n壓電片時間戳\n自動校正 e\n(estimator)'),
            ('③ 控制器', '狀態機\n鏡射律規劃每一跳\n平衡 PD\n(controller)'),
            ('④ 逆運動學', '升降＋傾斜\n→ 三個伺服角\n(kinematics)')]
    for i, (t, s) in enumerate(subs):
        _block(ax, 3.3 + i * 1.75, 6.1, 1.55, 2.0, t, s, fc='white', tfs=10)
        if i:
            _arrow(ax, (3.3 + i * 1.75 - 0.2, 7.1), (3.3 + i * 1.75, 7.1))
    if teensy:
        _block(ax, 11.0, 6.6, 2.1, 1.3, 'Teensy 4.1', 'USB 收三個脈寬\n3 路 PWM 300 Hz\n記撞擊時間、讀 IMU', fc='#f3f0fa')
    else:
        _block(ax, 11.0, 6.6, 2.1, 1.3, 'PCA9685', 'I²C → 3 路 PWM\n300 Hz（12 bit）\n波形由硬體產生', fc='#f3f0fa')
    _block(ax, 13.7, 6.6, 2.1, 1.3, '3× DS3218 伺服', f'臂 {g.arm * 1000:.0f} mm\n(servo)', fc='#fdf1e4')
    _block(ax, 13.7, 8.35, 2.1, 0.75, '6 V 10 A 電源', '只供伺服', fc='#eeeeee', tfs=10)
    _arrow(ax, (2.5, 7.25), (3.1, 7.25), 'USB\n約20 ms' if teensy else 'CSI／USB\n約20 ms', off=(0, 0.42))
    _arrow(ax, (10.3, 7.25), (11.0, 7.25), 'USB' if teensy else 'I²C', off=(0, 0.25))
    _arrow(ax, (13.1, 7.25), (13.7, 7.25), 'PWM', off=(0, 0.25))
    _arrow(ax, (14.75, 8.35), (14.75, 7.9))

    # 實體
    _block(ax, 12.6, 4.1, 3.2, 1.4, '平台＋乒乓球',
           f'Ø{g.plate_diameter * 1000:.0f} 壓克力、連桿 {g.rod * 1000:.0f} mm\n球 40 mm / 2.7 g (ball)', fc='#fdf1e4')
    _block(ax, 12.6, 2.35, 3.2, 1.05, '壓電片＋比較器', '撞擊→電壓尖峰→LM393→數位脈衝', fc='#fbf5dc')
    _arrow(ax, (14.75, 6.6), (14.75, 5.5), '連桿', off=(0.35, 0))
    _arrow(ax, (14.2, 4.1), (14.2, 3.4), '震動', off=(0.35, 0))
    _arrow(ax, (12.8, 5.5), (1.4, 6.6), '相機拍到球（位置延遲約 20 ms）', ls='--', color='#2f6f9f',
           off=(0.6, -0.45), rad=-0.12)
    if teensy:
        _path(ax, [(12.6, 2.75), (12.2, 2.75), (12.2, 6.6)],
              '中斷記時間（µs 級）\n經 USB 回報給筆電', color='#8a6d00', ls='--', toff=(-1.9, 0.0))
        _block(ax, 9.9, 4.25, 1.7, 0.95, 'GY-521', '黏在平台底面', fc='#e8f3e6', tfs=10)
        _arrow(ax, (11.1, 5.2), (11.5, 6.6), 'I²C', off=(-0.35, 0.05))
    else:
        _path(ax, [(12.6, 2.75), (9.6, 2.75), (9.6, 5.9)],
              'GPIO 中斷\nkernel 時間戳（µs 級）', color='#8a6d00', ls='--', toff=(0.15, 0.42))
        _block(ax, 9.9, 4.25, 1.7, 0.95, 'GY-521', '黏在平台底面\n跟 PCA9685 同一組 I²C', fc='#e8f3e6', tfs=10)
        _arrow(ax, (10.4, 5.2), (10.1, 5.9), 'I²C', off=(0.35, 0.0))
    other = ('另一套（方案 B）：\n樹莓派 4＋PCA9685\n（system.png）' if teensy else
             '另一套（方案 A）：\n筆電算、Teensy 出伺服訊號\n（system_teensy.png）')
    ax.add_patch(FancyBboxPatch((9.75, 1.45), 2.45, 0.95, boxstyle='round,pad=0.02,rounding_size=0.12',
                                fc='#fafafa', ec='#aaa', lw=1.0, ls=':'))
    ax.text(10.975, 1.925, other, ha='center', va='center', fontsize=8.3, color='#555', linespacing=1.3)

    # 狀態機
    ax.add_patch(FancyBboxPatch((0.3, 0.25), 9.1, 3.9, boxstyle='round,pad=0.02,rounding_size=0.15',
                                fc='#fbfbfb', ec='#999', lw=1.0))
    ax.text(0.5, 3.9, '控制器狀態機（controller.py）', fontsize=11.5, weight='bold')
    _block(ax, 0.6, 2.25, 1.9, 1.15, '拍球', '每跳彈回目標高度\n同時往中心修正', fc='#f6dfb8', tfs=10.5)
    _block(ax, 3.1, 2.25, 1.9, 1.15, '緩衝', '每跳高度減半\n最後一跳全吸收', fc='#d4e6f4', tfs=10.5)
    _block(ax, 5.6, 2.25, 1.9, 1.15, '下沉', '小跳來不及揮拍\n平台慢慢往下', fc='#e6dcf1', tfs=10.5)
    _block(ax, 7.3, 0.5, 1.9, 1.15, '平衡', '球貼板滾動\nPD 傾斜推回中心', fc='#d8eed5', tfs=10.5)
    ax.text(4.05, 3.52, '丟球 ↓', fontsize=9, color='#333', ha='center')
    _arrow(ax, (2.5, 2.83), (3.1, 2.83), '時間到', off=(0, 0.2), fs=8)
    _arrow(ax, (5.0, 2.83), (5.6, 2.83), '來球<3 mm', off=(0, 0.2), fs=8)
    _arrow(ax, (6.55, 2.25), (7.6, 1.65), '判定貼板', off=(0.55, 0.1), fs=8)
    _arrow(ax, (4.05, 2.25), (7.3, 1.1), '球已貼板', off=(-0.5, -0.25), fs=8, rad=0.15)
    ax.text(0.6, 0.95, '每一跳的規劃：預測落點時間與速度 → 鏡射律算出\n平台要斜幾度、撞擊瞬間往上/往下多快 → 排出\n一段升降軌跡（撞擊前後 ±15 ms 保持等速），\n再檢查伺服跟不跟得上，跟不上就逐步讓步。',
            fontsize=8.8, va='center', color='#333')
    fig.savefig(path, dpi=110, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


# ---------------------------------------------------------------- Teensy 接線圖
def _wire(ax, pts, color, lw=2.0, z=2):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle='round', zorder=z)


def _resistor(ax, x, y, text):
    ax.add_patch(Rectangle((x - 0.35, y - 0.11), 0.7, 0.22, fc='white', ec='#333', lw=1.2, zorder=4))
    ax.text(x, y + 0.18, text, ha='center', va='bottom', fontsize=8.5, color='#333', zorder=4)


def _stub(ax, x, y, color, text):
    """短線加文字：接到別處（不畫整條線，免得跟別的線交叉）。"""
    _wire(ax, [(x, y), (x - 0.6, y)], color)
    ax.text(x - 0.68, y, text, ha='right', va='center', fontsize=9, color=color)


def wiring_teensy(path):
    """方案 A（筆電算、Teensy 出伺服訊號）的接線圖。腳的位置是示意，照板子上印的號碼接。"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.text(0.2, 9.75, '方案 A 接線：筆電算、Teensy 4.1 出伺服訊號（4.0 用到的腳也一樣）', fontsize=14, weight='bold')
    red, blk, sig, i2c = '#c0392b', '#222222', '#d08a00', '#2c6fbb'
    ax.text(9.6, 8.8, '線色：紅＝電源正　黑＝地　橘＝訊號　藍＝I²C', fontsize=10, color='#444', va='center')

    # 相機、筆電
    _block(ax, 0.3, 8.3, 2.6, 1.0, 'OV9281 相機', 'USB 線 2 m', fc='#eaf4fb')
    _block(ax, 5.3, 8.3, 3.6, 1.0, '筆電', '影像、估測、控制（Python）', fc='#f7f7f2')
    _arrow(ax, (2.9, 8.8), (5.3, 8.8), 'USB', off=(0, 0.18))

    # Teensy
    tx0, tx1, ty0, ty1 = 5.9, 8.3, 2.9, 7.6
    ax.add_patch(FancyBboxPatch((tx0, ty0), tx1 - tx0, ty1 - ty0, boxstyle='round,pad=0.02,rounding_size=0.1',
                                fc='#2f6b3a', ec='#1d3f23', lw=1.5))
    ax.add_patch(Rectangle((6.75, ty1 - 0.05), 0.7, 0.3, fc='#bbbbbb', ec='#555', lw=1.0))   # micro-USB 座
    ax.text((tx0 + tx1) / 2, 3.25, 'Teensy 4.1', ha='center', va='center', fontsize=13, weight='bold', color='white')
    _arrow(ax, (7.1, 8.3), (7.1, ty1 + 0.25), 'micro-USB 傳輸線\n（Teensy 吃這條的電）', off=(1.6, -0.08))
    left = [('2', 6.6), ('3', 5.8), ('4', 5.0), ('GND', 4.0)]
    right = [('VIN（不接）', 7.2), ('GND', 6.6), ('3.3V', 6.0), ('19 SCL', 5.2), ('18 SDA', 4.6), ('14', 3.7)]
    for x, pins, ha, dx in ((tx0, left, 'left', 0.15), (tx1, right, 'right', -0.15)):
        for name, y in pins:
            ax.add_patch(Circle((x, y), 0.07, fc='#e8c547', ec='#555', zorder=5))
            ax.text(x + dx, y, name, ha=ha, va='center', fontsize=10, weight='bold',
                    color='#bbbbbb' if '不接' in name else 'white')

    # 伺服：訊號串 330 Ω 進 Teensy；電從端子台來
    for k, y in enumerate((6.6, 5.8, 5.0)):
        _block(ax, 0.3, y - 0.35, 2.2, 0.7, f'伺服 {k + 1}（第 {k + 1} 隻腳）', fc='#fdf1e4', tfs=9.5)
        _wire(ax, [(2.5, y), (tx0, y)], sig)
        _resistor(ax, 4.0, y, '330 Ω')
    _wire(ax, [(0.8, 6.6), (0.8, 3.4)], red, z=0.5)          # 畫在方塊後面，看起來像三條並在一起
    _wire(ax, [(1.9, 6.6), (1.9, 3.4)], blk, z=0.5)
    ax.text(0.88, 4.1, '紅（+6 V）', fontsize=9, color=red, va='center')
    ax.text(1.98, 4.1, '棕（地）', fontsize=9, color=blk, va='center')
    _block(ax, 0.3, 2.2, 2.2, 1.2, '端子台', '+6 V／GND\n伺服的電從這裡分', fc='#eeeeee', tfs=10)
    _block(ax, 0.3, 0.55, 2.2, 1.15, '6 V 10 A 變壓器', 'DC 母座轉端子\n正負跨 1000 µF', fc='#eeeeee', tfs=10)
    _wire(ax, [(0.8, 1.7), (0.8, 2.2)], red)
    _wire(ax, [(1.9, 1.7), (1.9, 2.2)], blk)
    _wire(ax, [(2.5, 2.5), (4.9, 2.5), (4.9, 4.0), (tx0, 4.0)], blk)
    ax.text(2.75, 2.2, '共地：端子台的地拉一條到 Teensy 的 GND', fontsize=9, color=blk, va='center')

    # GY-521（HW-123）
    gx0, gx1 = 10.3, 13.4
    ax.add_patch(FancyBboxPatch((gx0, 4.25), gx1 - gx0, 3.25, boxstyle='round,pad=0.02,rounding_size=0.1',
                                fc='#1f4e8c', ec='#14335c', lw=1.5))
    ax.text((gx0 + gx1) / 2, 7.2, 'GY-521（HW-123）', ha='center', va='center', fontsize=11.5, weight='bold',
            color='white')
    ax.text(gx1 - 0.15, 4.5, 'XDA、XCL、AD0、INT\n不接', ha='right', va='center', fontsize=8.5, color='#c9d6ea')
    for name, y, color in (('GND', 6.6, blk), ('VCC', 6.0, red), ('SCL', 5.2, i2c), ('SDA', 4.6, i2c)):
        _wire(ax, [(tx1, y), (gx0, y)], color)
        ax.add_patch(Circle((gx0, y), 0.07, fc='#e8c547', ec='#555', zorder=5))
        ax.text(gx0 + 0.15, y, name, ha='left', va='center', fontsize=10, weight='bold', color='white')

    # 比較器、壓電片
    cx0, cx1 = 10.3, 12.9
    _block(ax, cx0, 2.0, cx1 - cx0, 1.95, '', fc='#fbf5dc')
    ax.text(11.95, 3.15, 'LM393\n比較器模組', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(11.95, 2.45, '門檻用旋鈕調', ha='center', va='center', fontsize=8.5, color='#444')
    for name, y in (('DO', 3.7), ('VCC', 3.1), ('GND', 2.5)):
        ax.add_patch(Circle((cx0, y), 0.07, fc='#e8c547', ec='#555', zorder=5))
        ax.text(cx0 + 0.15, y, name, ha='left', va='center', fontsize=9.5, weight='bold', color='#333')
    _wire(ax, [(tx1, 3.7), (cx0, 3.7)], sig)
    _stub(ax, cx0, 3.1, red, '接 Teensy 3.3V')
    _stub(ax, cx0, 2.5, blk, '接 Teensy GND')
    _block(ax, 13.5, 1.9, 2.35, 2.1, '壓電片＋保護', '平台底面中心\n並聯 100 kΩ、3.3 V 稽納、\n蕭特基（陽極接地）\n再串 10 kΩ 進比較器',
           fc='#fbf5dc', tfs=10)
    _arrow(ax, (13.5, 2.95), (cx1, 2.95))

    notes = ['① Teensy 的腳只吃 3.3 V，不能碰到 5 V。伺服的 6 V 不進 Teensy，VIN 不接（Teensy 吃 USB 的電）。',
             '② 330 Ω 是保護用：伺服接頭插反、6 V 碰到訊號線時，把流進 Teensy 的電流限住。',
             '③ DS3218 吃不吃 3.3 V 的訊號，到貨先試一顆；會抖或不動，就在 2、3、4 後面加一顆 74AHCT125'
             '（它的 VCC 接 Teensy 的 VIN，也就是 USB 的 5 V）。',
             '④ 圖上腳的位置是示意，照板子背面印的號碼（或附的腳位卡）接；GND 有好幾個，都是通的。']
    ax.text(2.9, 1.6, '\n'.join(notes), fontsize=9.5, va='top', color='#333', linespacing=1.6)
    fig.savefig(path, dpi=110, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


# ---------------------------------------------------------------- 樹莓派接線圖
def wiring_rpi(path):
    """方案 B（樹莓派 4＋PCA9685）的接線圖。樹莓派的 40 支腳只有一排，都畫在板子右邊；位置是示意，照實體編號接。"""
    fig, ax = plt.subplots(figsize=(16, 11.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(-1.5, 10)
    ax.axis('off')
    ax.text(0.2, 9.75, '方案 B 接線：樹莓派 4（4 GB）算、PCA9685 出伺服訊號（樹莓派 5 的腳位也一樣）',
            fontsize=14, weight='bold')
    red, blk, sig, i2c = '#c0392b', '#222222', '#d08a00', '#2c6fbb'
    ax.text(10.4, 9.15, '線色：紅＝電源正　黑＝地　橘＝訊號　藍＝I²C', fontsize=10, color='#444', va='center')

    def pin(x, y, name, ha, dx, color='white'):
        ax.add_patch(Circle((x, y), 0.07, fc='#e8c547', ec='#555', zorder=5))
        ax.text(x + dx, y, name, ha=ha, va='center', fontsize=10, weight='bold', color=color)

    # 樹莓派（腳位跟 docs/硬體上機檢查清單.md 同一組）
    px0, px1 = 2.8, 6.4
    ax.add_patch(FancyBboxPatch((px0, 2.3), px1 - px0, 6.45, boxstyle='round,pad=0.02,rounding_size=0.1',
                                fc='#2f6b3a', ec='#1d3f23', lw=1.5))
    ax.text(3.65, 5.0, 'Raspberry Pi 4\n（4 GB）', ha='center', va='center', fontsize=12, weight='bold',
            color='white', linespacing=1.4)
    P = dict(io27=8.45, gnd_led=8.05, v5=7.65, v33=7.2, sda=6.65, scl=6.1, gnd=5.55, gnd_c=4.1, do=3.0)
    for key, name in (('io27', '13  GPIO27（選配）'), ('gnd_led', '14  GND（選配）'), ('v5', '2、4  5V（平常不接）'),
                      ('v33', '1  3.3V'), ('sda', '3  SDA（GPIO2）'), ('scl', '5  SCL（GPIO3）'), ('gnd', '6  GND'),
                      ('gnd_c', '9  GND'), ('do', '11  GPIO17')):
        pin(px1, P[key], name, 'right', -0.15, color='#bbbbbb' if '不接' in name else 'white')

    # 選配的對時 LED：GPIO27 → 330 Ω → LED → 隔壁的第 14 腳
    _wire(ax, [(px1, P['io27']), (8.18, P['io27'])], sig)
    _resistor(ax, 7.2, P['io27'], '330 Ω')
    ax.add_patch(Polygon([(8.18, 8.58), (8.18, 8.32), (8.40, 8.45)], closed=True, fc='#e74c3c', ec='#333', lw=1.0,
                         zorder=4))
    _wire(ax, [(8.40, 8.32), (8.40, 8.58)], blk, lw=1.5, z=4)
    for x0 in (8.26, 8.38):
        ax.add_patch(FancyArrowPatch((x0, 8.62), (x0 + 0.1, 8.78), arrowstyle='-|>', mutation_scale=7, color='#e74c3c',
                                     lw=1.0))
    _wire(ax, [(8.40, P['io27']), (9.1, P['io27']), (9.1, P['gnd_led']), (px1, P['gnd_led'])], blk)
    ax.text(9.25, 8.25, '對時 LED（選配，紅色）：量相機延遲、\n跟手機錄影對時；放在相機拍得到的地方', fontsize=9,
            color='#444', va='center', linespacing=1.4)

    # 相機、電源插在樹莓派左邊
    for y in (7.6, 6.2):
        ax.add_patch(Rectangle((px0 - 0.28, y - 0.17), 0.3, 0.34, fc='#bbbbbb', ec='#555', lw=1.0, zorder=3))
    _block(ax, 0.2, 7.1, 1.75, 1.0, 'OV9281 相機', 'USB 線 2 m\n插藍色的 USB 3.0', fc='#eaf4fb', tfs=10)
    _arrow(ax, (1.95, 7.6), (px0 - 0.28, 7.6))
    _block(ax, 0.2, 5.65, 1.75, 1.1, '樹莓派電源', '官方 USB-C 15 W\n（5.1 V 3 A）', fc='#eeeeee', tfs=10)
    _arrow(ax, (1.95, 6.2), (px0 - 0.28, 6.2))

    # PCA9685：左邊那排接樹莓派，右邊那排（兩頭是通的）接 IMU
    cx0, cx1, cy0 = 7.4, 11.4, 5.1
    ax.add_patch(FancyBboxPatch((cx0, cy0), cx1 - cx0, 2.5, boxstyle='round,pad=0.02,rounding_size=0.1',
                                fc='#4a3a7a', ec='#2e2450', lw=1.5))
    ax.text(9.4, 7.2, 'PCA9685', ha='center', va='center', fontsize=12, weight='bold', color='white')
    ax.text(9.4, 6.45, '16 路 PWM，用第 0、1、2 路\n兩頭這 4 支腳是通的\nOE、V+、綠色端子都不接', ha='center',
            va='center', fontsize=8.5, color='#dcd6ee', linespacing=1.4)
    bus = (('VCC', P['v33'], red), ('SDA', P['sda'], i2c), ('SCL', P['scl'], i2c), ('GND', P['gnd'], blk))
    for name, y, color in bus:
        _wire(ax, [(px1, y), (cx0, y)], color)
        pin(cx0, y, name, 'left', 0.15)
        pin(cx1, y, name, 'right', -0.15)

    # 伺服：訊號從 PCA9685 第 0、1、2 路出來；電從端子台來
    for k, x in enumerate((8.0, 9.4, 10.8)):
        ax.add_patch(Circle((x, cy0), 0.07, fc='#e8c547', ec='#555', zorder=5))
        ax.text(x, cy0 + 0.2, f'{k}', ha='center', va='center', fontsize=9.5, weight='bold', color='white')
        _wire(ax, [(x, cy0), (x, 4.0)], sig)
        _block(ax, x - 0.62, 3.3, 1.24, 0.7, f'伺服 {k + 1}', f'第 {k + 1} 隻腳', fc='#fdf1e4', tfs=9.5)
        _wire(ax, [(x - 0.3, 3.3), (x - 0.3, 2.6)], red)
        _wire(ax, [(x + 0.3, 3.3), (x + 0.3, 2.6)], blk)
    ax.text(11.2, 2.95, '紅＝+6 V、棕＝地', fontsize=9, color='#444', va='center')
    _block(ax, 7.38, 1.6, 4.04, 1.0, '端子台', '+6 V／GND，三顆伺服的電從這裡分', fc='#eeeeee', tfs=10)
    _block(ax, 12.0, 1.6, 2.7, 1.0, '6 V 10 A 變壓器', 'DC 母座轉端子，正負跨 1000 µF', fc='#eeeeee', tfs=10)
    _wire(ax, [(12.0, 2.25), (11.42, 2.25)], red)
    _wire(ax, [(12.0, 1.85), (11.42, 1.85)], blk)
    _wire(ax, [(7.1, P['gnd']), (7.1, 1.85), (7.38, 1.85)], blk)             # 共地：從第 6 腳那條地線分出來
    ax.add_patch(Circle((7.1, P['gnd']), 0.06, fc=blk, ec=blk, zorder=4))
    ax.text(7.16, 4.6, '共地', ha='left', va='center', fontsize=9, color=blk)

    # GY-521（HW-123）
    gx0, gx1 = 12.4, 15.4
    ax.add_patch(FancyBboxPatch((gx0, cy0), gx1 - gx0, 2.5, boxstyle='round,pad=0.02,rounding_size=0.1',
                                fc='#1f4e8c', ec='#14335c', lw=1.5))
    ax.text(14.15, 7.25, 'GY-521（HW-123）', ha='center', va='center', fontsize=11.5, weight='bold', color='white')
    ax.text(14.15, 6.5, '黏在平台底面\nI²C 位址 0x68\n（PCA9685 是 0x40）', ha='center', va='center', fontsize=8.5,
            color='#c9d6ea', linespacing=1.4)
    ax.text(gx1 - 0.15, 5.38, 'XDA、XCL、AD0、INT 不接', ha='right', va='center', fontsize=8.5, color='#c9d6ea')
    for name, y, color in bus:
        _wire(ax, [(cx1, y), (gx0, y)], color)
        pin(gx0, y, name, 'left', 0.15)

    # 比較器、壓電片（期末拍球才用）：DO、GND 從樹莓派右邊繞到板子下面再往左接，不跟別的線交叉；VCC 接第 1 腳
    for key, xv, yh, color in (('do', 6.6, 1.8, sig), ('gnd_c', 6.9, 0.8, blk)):
        _wire(ax, [(px1, P[key]), (xv, P[key]), (xv, yh), (2.2, yh)], color)
    _wire(ax, [(2.2, 1.3), (2.75, 1.3)], red)
    ax.text(2.85, 1.3, '接第 1 腳（3.3V）', ha='left', va='center', fontsize=9, color=red)
    _block(ax, 0.2, 0.45, 2.0, 1.7, '', fc='#fbf5dc')
    ax.text(0.95, 1.55, 'LM393\n比較器模組', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(0.95, 0.85, '門檻用旋鈕調', ha='center', va='center', fontsize=8.5, color='#444')
    ax.text(1.2, 0.25, '期中只平衡，可以先不接', ha='center', va='center', fontsize=8.5, color='#8a6d00')
    for name, y in (('DO', 1.8), ('VCC', 1.3), ('GND', 0.8)):
        ax.add_patch(Circle((2.2, y), 0.07, fc='#e8c547', ec='#555', zorder=5))
        ax.text(2.05, y, name, ha='right', va='center', fontsize=9.5, weight='bold', color='#333')
    _block(ax, 0.2, 2.75, 2.0, 2.1, '壓電片＋保護', '平台底面中心\n並聯 100 kΩ、\n3.3 V 稽納、\n蕭特基（陽極接地）\n再串 10 kΩ\n進比較器',
           fc='#fbf5dc', tfs=10)
    _arrow(ax, (1.2, 2.75), (1.2, 2.15))

    notes = ['① 樹莓派的腳只吃 3.3 V，不能碰到 5 V 或伺服的 6 V。PCA9685、IMU、比較器的 VCC 都接第 1 腳的 3.3V'
             '（一支腳要分給好幾樣時，用 Y 型杜邦線或在洞洞板上分）；第 2、4 腳的 5V 平常不接（見 ③）。',
             '② 伺服的電不走 PCA9685 的綠色端子（副廠板的防反接電晶體很小顆），從端子台直接給，PCA9685 只接訊號線；'
             '端子台的地拉一條到樹莓派第 6 腳（共地，跟 PCA9685 的 GND 同一支）。',
             '③ PCA9685 的 VCC 接 3.3 V，伺服訊號就是 3.3 V。DS3218 吃不吃 3.3 V，到貨先試一顆；會抖或不動，'
             '就在第 0、1、2 路後面加一顆 74AHCT125（它的 VCC 接樹莓派第 2 腳的 5V）。',
             '④ 腳的位置是示意，照實體編號接：第 1 腳在排針離 USB 孔最遠的那一端；GND 有好幾支，都是通的。'
             'IMU 接 PCA9685 另一頭那排同名的腳；板子只有一頭有排針，就並到樹莓派的同一支腳。',
             '⑤ 第 13、14 腳的對時 LED 是選配：串 330 Ω，用來量相機延遲、跟手機錄影對時，要放在相機拍得到的地方。']
    ax.text(0.3, -0.05, '\n'.join(notes), fontsize=9.5, va='top', color='#333', linespacing=1.6)
    fig.savefig(path, dpi=110, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
