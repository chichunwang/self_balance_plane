"""「真實世界」的純 Python 版：簡化伺服＋運動學正解＋球的衝量物理。

MuJoCo 版（mjplant.MjPlant）提供同樣的介面，模擬主迴圈不用管底下是哪一個。
"""
import math

import numpy as np

from .ball import Ball, Surface, step as ball_step
from .kinematics import Platform
from .servo import Servo


def ball_start(cfg, sc):
    if sc.start_p is not None:          # 側拋：直接給初始位置與速度
        return list(sc.start_p), list(sc.start_v)
    p = [sc.drop_xy[0], sc.drop_xy[1], cfg.geom.z_surface0 + cfg.ball.radius + sc.drop_height]
    v = [sc.drop_vxy[0], sc.drop_vxy[1], 0.0]
    return p, v


class PyPlant:
    name = '純 Python'

    def __init__(self, cfg, sc, rng):
        self.cfg = cfg
        gm = cfg.geom
        self.plat = Platform(gm)
        lim = math.radians(gm.servo_limit_deg)
        er = cfg.errors
        self.servos = [Servo(cfg.servo, 0.0, rng.uniform(0, 1 / cfg.servo.pwm_hz), lim,
                             math.radians(er.trim_deg[i]), er.gain_err[i]) for i in range(3)]
        self.c, self.R = self.plat.neutral()
        self.point, self.n = self.plat.surface(self.c, self.R)
        self.ball = Ball(cfg.ball, *ball_start(cfg, sc))
        self.surf = Surface(self.point, self.n, np.zeros(3), np.zeros(3), gm.plate_diameter / 2)

    def command(self, th):
        for s, x in zip(self.servos, th):
            s.command(x)

    def reset_ball(self, p, v, w=(0.0, 0.0, 0.0)):
        self.ball = Ball(self.cfg.ball, p, v, w)

    def step(self, t, dt):
        for s in self.servos:
            s.step(t, dt)
        R_prev, point_prev = self.R, self.point
        self.c, self.R = self.plat.fk(self.th, self.c, self.R)
        self.point, self.n = self.plat.surface(self.c, self.R)
        Rr = self.R @ R_prev.T
        omega = np.array([Rr[2, 1] - Rr[1, 2], Rr[0, 2] - Rr[2, 0], Rr[1, 0] - Rr[0, 1]]) / (2 * dt)
        self.surf = Surface(self.point, self.n, (self.point - point_prev) / dt, omega,
                            self.cfg.geom.plate_diameter / 2)
        imp = ball_step(self.ball, self.surf, self.cfg.surface, dt)
        if imp is None:
            return []
        imp['t'] = t + dt - imp['t_rem']
        return [imp]

    @property
    def ball_p(self):
        return self.ball.p

    @property
    def ball_v(self):
        return self.ball.v

    @property
    def th(self):
        return np.array([s.theta for s in self.servos])

    @property
    def om(self):
        return np.array([s.omega for s in self.servos])

    def extras(self):
        """MuJoCo 版才有的量（馬達扭力、球頭受力），這裡沒有"""
        return None
