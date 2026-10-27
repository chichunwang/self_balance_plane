import math

import numpy as np
import pytest

from ppsim.ball import Ball, impact_model
from ppsim.controller import hermite, launch_speed, mirror_solve
from ppsim.kinematics import Platform, tilt_normal
from ppsim.params import BallSpec, Config, PlantErrors
from ppsim.sim import Scenario, simulate, toss


def test_mirror_solve_gives_requested_rebound_with_friction_and_spin():
    rng = np.random.default_rng(1)
    b = BallSpec()
    for spin_scale in (0.0, 20.0):
        for _ in range(20):
            v_in = np.array([*rng.uniform(-0.6, 0.6, 2), rng.uniform(-3.0, -0.5)])
            v_out = np.array([*rng.uniform(-0.2, 0.2, 2), rng.uniform(0.3, 2.5)])
            spin = rng.uniform(-1, 1, 3) * spin_scale

            def bounce(n, v_pl):
                return impact_model(v_in, spin, n, v_pl, 0.86, 0.25, b.radius, b.k_inertia)[0]

            tau, w = mirror_solve(v_in, v_out, 0.86, bounce)
            assert np.allclose(bounce(tilt_normal(*tau), np.array([0, 0, w])), v_out, atol=1e-8)


def test_vertical_bounce_needs_textbook_plate_speed():
    # 維持同樣高度：w = v(1−e)/(1+e)
    v, e = 2.43, 0.88
    tau, w = mirror_solve(np.array([0, 0, -v]), np.array([0, 0, v]), e)
    assert np.allclose(tau, 0)
    assert abs(w - v * (1 - e) / (1 + e)) < 1e-12


def test_launch_speed_reaches_apex_with_drag():
    spec = BallSpec()
    h = 0.25
    ball = Ball(spec, [0, 0, 0], [0, 0, launch_speed(h, spec.beta, spec.g_eff)])
    top = 0.0
    for _ in range(20000):
        ball.fly(2e-5)
        top = max(top, ball.p[2])
        if ball.v[2] < 0:
            break
    assert abs(top - h) < 5e-4


def test_hermite_hits_endpoints():
    p, v = hermite(0.0, 0.1, 0.02, -0.2, 0.3, 0.0)
    assert (p, v) == (0.0, 0.1)
    p, v = hermite(0.0, 0.1, 0.02, -0.2, 0.3, 1.0)
    assert abs(p - 0.02) < 1e-12 and abs(v + 0.2) < 1e-12


def test_auto_level_learns_plate_tilt_caused_by_servo_trim_errors():
    trim = (1.0, -1.0, 0.5)
    cfg = Config()
    cfg.errors = PlantErrors(trim_deg=trim)
    res = simulate(cfg, Scenario('juggle', duration=7.0, juggle_until=6.0, seed=1))
    plat = Platform(cfg.geom)
    c0, R0 = plat.neutral()
    _, R = plat.fk(np.radians(trim), c0, R0, iters=30)
    true_tilt = R[:2, 2] / R[2, 2]           # 命令水平時，平台其實歪成這樣
    learned = res.trim_log[-1][1]           # 學到的修正量應該剛好抵銷它
    assert np.linalg.norm(learned + true_tilt) < 0.35 * np.linalg.norm(true_tilt)
    assert res.metrics['land_offset'] < 0.008


def test_gentle_side_toss_is_caught_and_balanced():
    cfg = Config()
    sc = toss(cfg, (0.01, -0.01), 200.0, 0.3, apex=0.35, launch_h=0.25, duration=4.0, seed=1)
    assert np.hypot(*sc.start_v[:2]) > 0.6          # 真的是帶水平速度從旁邊來的
    m = simulate(cfg, sc).metrics
    assert m['fell'] is None and m['t_balance'] < 3.5 and m['final_r'] < 0.003


def test_estimator_gives_up_on_stale_track():
    from ppsim.estimator import Estimator
    from ppsim.sensors import Camera
    cfg = Config()
    cam = Camera(cfg.camera, cfg.geom.z_surface0 + cfg.camera.height, cfg.ball.radius, np.random.default_rng(0))
    est = Estimator(cfg, cam, lambda t: (0.0, 0.0, (0.0, 0.0)))
    est.on_camera(0.000, [0.0, 0.0, 0.40], 0.02)
    est.on_camera(0.008, [0.0, 0.0, 0.40], 0.028)
    assert est.t is not None
    x, mode = est.predict(1.0)                      # 一秒沒看到球：不要硬推，放棄等重新鎖定
    assert x is None and est.t is None


def on_plate(cfg, duration, xy=(0.0, 0.0), push=None, seed=1):
    """球靜止放在板上；push＝(時間, vx, vy)：那時候用手推一下，球變成這個速度貼著板子滾。"""
    z = cfg.geom.z_surface0 + cfg.ball.radius
    return Scenario('push' if push else 'rest', duration=duration, seed=seed, start_p=(xy[0], xy[1], z),
                    start_v=(0.0, 0.0, 0.0), push=push)


def test_balance_holds_the_ball_for_20_seconds():
    """期中目標 1：球靜止放在離中心 3.6 cm 的地方，拉回中心以後 20 秒都在中心 1 cm 以內。"""
    cfg = Config()
    res = simulate(cfg, on_plate(cfg, 24.0, xy=(0.03, -0.02)))
    t, p = res.log['t'], res.log['p']
    assert res.fell is None
    assert np.hypot(p[t > 4.0, 0], p[t > 4.0, 1]).max() < 0.01


@pytest.mark.parametrize('az', range(0, 360, 45))
@pytest.mark.parametrize('v', [0.25, 0.30])
def test_hand_push_is_recovered(v, az):
    """期中目標 2：球停在中心時被推成 v 滾出去（8 個方向），不能掉下去，3 秒內回到中心 1 cm 以內。
    0.30 m/s 要靠平衡傾角上限 8°（5° 時 8 個方向全掉，2026-10-26 試算）。"""
    cfg = Config()
    a = math.radians(az)
    res = simulate(cfg, on_plate(cfg, 6.0, push=(2.0, v * math.cos(a), v * math.sin(a))))
    t, p = res.log['t'], res.log['p']
    assert res.fell is None
    assert np.hypot(p[t >= 5.0, 0], p[t >= 5.0, 1]).max() < 0.01


def test_catch_scenario_ends_balanced_near_center():
    res = simulate(Config(), Scenario('catch', duration=3.5, seed=1))
    m = res.metrics
    assert m['fell'] is None
    assert m['t_balance'] < 3.0
    assert m['final_r'] < 0.003
    assert m['servo_peak'] < 1.0
