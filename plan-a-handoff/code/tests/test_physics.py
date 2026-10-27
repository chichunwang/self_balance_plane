import math

import numpy as np

from ppsim.ball import Ball, Surface, collide, impact_model, step
from ppsim.kinematics import tilt_normal
from ppsim.params import BallSpec, MechSpec, SERVOS, SurfaceSpec
from ppsim.servo import MotorServo, Servo


def flat(n=(0.0, 0.0, 1.0), vel=(0.0, 0.0, 0.0)):
    n = np.array(n, float)
    return Surface(np.zeros(3), n / np.linalg.norm(n), np.array(vel, float), np.zeros(3), 1.0)


def test_drop_on_static_plate_rebounds_to_e_squared():
    spec = BallSpec(cd=0.0)
    ss = SurfaceSpec(e=0.87, mu=0.3)
    h = 0.30
    ball = Ball(spec, [0, 0, spec.radius + h], [0, 0, 0])
    surf = flat()
    hits, apex = 0, 0.0
    for _ in range(int(1.2 / 1e-4)):
        if step(ball, surf, ss, 1e-4):
            hits += 1
        if hits == 1:
            apex = max(apex, ball.p[2] - spec.radius)
        if hits == 2:
            break
    assert abs(apex / h - ss.e ** 2) < 0.01


def test_moving_plate_adds_velocity():
    spec = BallSpec()
    e, u, v_in = 0.87, 0.15, 2.0
    ball = Ball(spec, [0, 0, spec.radius - 1e-4], [0, 0, -v_in])
    collide(ball, flat(vel=(0, 0, u)), e, 0.3, 0.02)
    assert abs(ball.v[2] - (e * v_in + (1 + e) * u)) < 1e-12


def test_ball_rolls_down_slope_at_0_6_g_sin_theta():
    spec = BallSpec(cd=0.0)
    ss = SurfaceSpec(mu=0.5)
    th = math.radians(5)
    n = np.array([math.sin(th), 0.0, math.cos(th)])   # 往 +x 下坡
    surf = flat(n=n)
    ball = Ball(spec, n * spec.radius, [0, 0, 0])
    dt, T = 5e-4, 0.4
    for _ in range(int(T / dt)):
        step(ball, surf, ss, dt)
    v_along = ball.v @ np.array([math.cos(th), 0.0, -math.sin(th)])
    expected = 0.6 * spec.g_eff * math.sin(th) * T
    assert abs(v_along / expected - 1) < 0.02


def test_controller_impact_model_matches_physics_with_friction_and_spin():
    spec, e, mu = BallSpec(), 0.87, 0.25
    rng = np.random.default_rng(3)
    for _ in range(30):
        n = tilt_normal(*rng.uniform(-0.15, 0.15, 2))
        v_in = np.array([*rng.uniform(-1.5, 1.5, 2), rng.uniform(-3.0, -0.4)])
        w_in = rng.uniform(-40, 40, 3)
        v_pl = np.array([0.0, 0.0, rng.uniform(-0.2, 0.2)])
        ball = Ball(spec, [0, 0, 0], v_in, w_in)
        collide(ball, Surface(np.zeros(3), n, v_pl, np.zeros(3), 1.0), e, mu, 0.0)
        v_out, w_out = impact_model(v_in, w_in, n, v_pl, e, mu, spec.radius, spec.k_inertia)
        assert np.allclose(ball.v, v_out) and np.allclose(ball.w, w_out)


def test_side_ball_loses_horizontal_speed_on_first_bounce():
    # 沒旋轉的薄殼球平平撞上靜止平板：摩擦夠大時變成滾動，水平速度剩 60%
    spec = BallSpec()
    v_out, _ = impact_model(np.array([1.0, 0, -2.5]), np.zeros(3), np.array([0, 0, 1.0]), np.zeros(3),
                            0.87, 0.25, spec.radius, spec.k_inertia)
    assert abs(v_out[0] - 0.6) < 1e-9


def test_servo_respects_speed_limit_and_reaches_target():
    spec = SERVOS["DS3218"]
    s = Servo(spec)
    s.command(math.radians(40))
    peak, dt = 0.0, 1e-4
    for k in range(int(0.5 / dt)):
        s.step(k * dt, dt)
        peak = max(peak, abs(s.omega))
    assert peak <= spec.w_max + 1e-9
    assert abs(s.theta - math.radians(40)) < spec.deadband + spec.quantum


def test_motor_servo_tops_out_at_no_load_speed_and_settles():
    spec, j_out, h = SERVOS["DS3218"], 2e-4, 2.5e-4
    s = MotorServo(spec, MechSpec(), j_out)
    s.command(math.radians(40))
    th = w = peak = 0.0
    for k in range(int(0.6 / h)):
        tau = s.update(k * h, h, th, w)     # 輸出軸當成單純的慣量
        w += tau / j_out * h
        th += w * h
        peak = max(peak, abs(w))
    assert 0.8 * spec.w_noload < peak <= 1.02 * spec.w_noload
    assert abs(th - math.radians(40)) < math.radians(0.5)
