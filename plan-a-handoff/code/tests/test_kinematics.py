import math

import numpy as np
import pytest

from ppsim.kinematics import Platform
from ppsim.params import Geometry

# 改版後臂朝外（預設），第一版的臂朝內也要能算
BOTH = pytest.mark.parametrize("outward", [True, False], ids=["臂朝外", "臂朝內"])


def make(outward=True):
    return Platform(Geometry(arm_outward=outward))


def test_default_geometry_is_the_compact_one():
    g = Geometry()
    assert g.arm_outward
    assert g.r_b == pytest.approx(0.035)            # 80 − 45：伺服靠中間
    assert Geometry(arm_outward=False).r_b == pytest.approx(0.125)


@BOTH
def test_neutral_pose_puts_arms_horizontal(outward):
    plat = make(outward)
    c, R, _ = plat.pose(0.0, 0.0, 0.0)
    th, ok = plat.ik(c, R)
    assert ok
    assert np.allclose(th, 0.0, atol=1e-9)
    tips = plat.arm_tips(th)                        # 臂端在球頭正下方
    assert np.allclose(tips[:, :2], plat.p_loc[:, :2], atol=1e-12)


@BOTH
def test_positive_servo_angle_lifts_the_arm_tip(outward):
    plat = make(outward)
    low, high = plat.arm_tips(np.full(3, -0.3)), plat.arm_tips(np.full(3, 0.3))
    assert np.all(high[:, 2] > low[:, 2])


@BOTH
def test_heave_raises_all_arms_equally(outward):
    plat = make(outward)
    c, R, _ = plat.pose(0.010, 0.0, 0.0)
    th, ok = plat.ik(c, R)
    assert ok
    assert np.allclose(th, th[0]) and th[0] > 0
    # 連桿幾乎垂直，臂端抬高量≈平台抬高量
    assert abs(plat.g.arm * math.sin(th[0]) - 0.010) < 2e-4


@BOTH
def test_ik_then_fk_recovers_pose(outward):
    plat = make(outward)
    rng = np.random.default_rng(0)
    c0, R0 = plat.neutral()
    for _ in range(50):
        dz = rng.uniform(-0.02, 0.02)
        tilt = math.tan(math.radians(rng.uniform(0, 8)))
        ang = rng.uniform(0, 2 * math.pi)
        c, R, _ = plat.pose(dz, tilt * math.cos(ang), tilt * math.sin(ang))
        th, ok = plat.ik(c, R)
        assert ok
        c2, R2 = plat.fk(th, c0, R0, iters=30)
        assert np.allclose(c, c2, atol=1e-9)
        assert np.allclose(R, R2, atol=1e-9)


@BOTH
def test_parasitic_motion_is_small(outward):
    plat = make(outward)
    tilt = math.tan(math.radians(8))
    for ang in np.linspace(0, 2 * math.pi, 7):
        c, R, (x, y, psi) = plat.pose(0.0, tilt * math.cos(ang), tilt * math.sin(ang))
        assert math.hypot(x, y) < 0.001          # < 1 mm
        assert abs(math.degrees(psi)) < 0.5
