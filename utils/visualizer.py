"""
Real-time pygame visualizer for the BDC motor + vertical rod system.

Usage (in main)
---------------
from utils.visualizer import MotorVisualizer

vis = MotorVisualizer(title="BDC Motor", history_len=500)
vis.open()

# inside simulation loop:
running = vis.update(state, voltage, omega_noisy=None)
if not running:
    break

vis.close()
"""

from __future__ import annotations
import math
from collections import deque

import numpy as np

try:
    import pygame
except ImportError as e:
    raise ImportError("pygame is required: pip install pygame") from e


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
_BG      = (18,  18,  24)
_PANEL   = (30,  30,  40)
_WHITE   = (230, 230, 230)
_GRAY    = (110, 110, 130)
_ACCENT  = (80,  160, 255)   # blue  — true speed / rod
_NOISY   = (255, 120,  60)   # orange — noisy measurement
_GREEN   = ( 80, 220, 120)
_RED     = (220,  70,  70)
_YELLOW  = (240, 210,  50)


class MotorVisualizer:
    """
    Pygame window with three panels:
      Left   — rotating rod animation
      Centre — scrolling speed history (true + noisy)
      Right  — numeric readouts
    """

    def __init__(
        self,
        title:       str = "BDC Motor Simulator",
        history_len: int = 600,
        width:       int = 960,
        height:      int = 480,
        fps:         int = 60,
    ):
        self._title       = title
        self._history_len = history_len
        self._W           = width
        self._H           = height
        self._fps         = fps
        self._screen      = None
        self._clock       = None
        self._font_sm     = None
        self._font_md     = None
        self._font_lg     = None

        # scrolling history buffers
        self._omega_true  : deque[float] = deque(maxlen=history_len)
        self._omega_noisy : deque[float] = deque(maxlen=history_len)
        self._voltage_buf : deque[float] = deque(maxlen=history_len)
        self._t_buf       : deque[float] = deque(maxlen=history_len)

    # ------------------------------------------------------------------
    def open(self):
        """Initialise pygame and open the window."""
        pygame.init()
        self._screen = pygame.display.set_mode((self._W, self._H))
        pygame.display.set_caption(self._title)
        self._clock   = pygame.time.Clock()
        self._font_sm = pygame.font.SysFont("consolas", 13)
        self._font_md = pygame.font.SysFont("consolas", 16)
        self._font_lg = pygame.font.SysFont("consolas", 22, bold=True)

    def close(self):
        """Tear down pygame."""
        pygame.quit()

    # ------------------------------------------------------------------
    def update(
        self,
        state,                          # BDCMotorState
        voltage:      float,
        omega_noisy:  float | None = None,
        omega_target: float | None = None,
    ) -> bool:
        """
        Redraw the window with the current simulation state.

        Returns False if the user closed the window, True otherwise.
        """
        # ---- event pump ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        # ---- buffer history ----
        self._omega_true.append(state.omega)
        self._omega_noisy.append(
            omega_noisy if omega_noisy is not None
            else state.omega
        )
        self._voltage_buf.append(voltage)
        self._t_buf.append(state.t)

        # ---- draw ----
        self._screen.fill(_BG)
        self._draw_rod_panel(state.theta, state.omega)
        self._draw_plot_panel(omega_target)
        self._draw_readout_panel(state, voltage, omega_target)

        pygame.display.flip()
        self._clock.tick(self._fps)
        return True

    # ------------------------------------------------------------------
    # Panel helpers
    # ------------------------------------------------------------------

    def _draw_rod_panel(self, theta: float, omega: float):
        """Left third: rotating rod centred in a circle."""
        panel_w = self._W // 3
        rect = pygame.Rect(0, 0, panel_w, self._H)
        pygame.draw.rect(self._screen, _PANEL, rect)

        cx, cy = panel_w // 2, self._H // 2
        rod_len = min(panel_w, self._H) // 2 - 20

        # pivot circle
        pygame.draw.circle(self._screen, _GRAY, (cx, cy), rod_len + 10, 1)
        pygame.draw.circle(self._screen, _WHITE, (cx, cy), 6)

        # rod tip
        tip_x = int(cx + rod_len * math.sin(theta))
        tip_y = int(cy + rod_len * math.cos(theta))   # +cos: 0 rad = hanging down

        # colour by speed: blue → green → red
        spd = abs(omega)
        r = min(255, int(spd * 2))
        g = max(0, 200 - int(spd * 2))
        rod_colour = (r, g, 80)

        pygame.draw.line(self._screen, rod_colour, (cx, cy), (tip_x, tip_y), 5)
        pygame.draw.circle(self._screen, rod_colour, (tip_x, tip_y), 8)

        # label
        lbl = self._font_sm.render("Rod position", True, _GRAY)
        self._screen.blit(lbl, (8, 8))

        theta_deg = math.degrees(theta) % 360
        ang_lbl = self._font_md.render(f"{theta_deg:6.1f} deg", True, _WHITE)
        self._screen.blit(ang_lbl, (8, self._H - 28))

    def _draw_plot_panel(self, omega_target: float | None):
        """Centre third: speed history (top 60%) + voltage input (bottom 40%)."""
        panel_w = self._W // 3
        ox      = panel_w
        margin  = 30
        plot_w  = panel_w - 2 * margin

        speed_h   = int(self._H * 0.58) - margin
        voltage_h = self._H - speed_h - 3 * margin
        split_y   = margin + speed_h + margin   # top of voltage sub-plot

        rect = pygame.Rect(ox, 0, panel_w, self._H)
        pygame.draw.rect(self._screen, _PANEL, rect)

        # divider line
        pygame.draw.line(self._screen, _GRAY,
                         (ox, split_y - margin // 2),
                         (ox + panel_w, split_y - margin // 2), 1)

        if len(self._omega_true) < 2:
            return

        def series_points(vals, top_y, height, lo, hi):
            n = len(vals)
            pts = []
            for j, v in enumerate(vals):
                px = ox + margin + int(j / max(n - 1, 1) * plot_w)
                py = int(top_y + height - (v - lo) / (hi - lo) * height)
                py = max(top_y, min(top_y + height, py))
                pts.append((px, py))
            return pts

        # ---- speed sub-plot ----
        arr_true  = list(self._omega_true)
        arr_noisy = list(self._omega_noisy)
        all_vals  = arr_true + arr_noisy
        if omega_target is not None:
            all_vals.append(omega_target)
        lo_s, hi_s = min(all_vals), max(all_vals)
        span = max(hi_s - lo_s, 1.0)
        lo_s -= span * 0.05; hi_s += span * 0.05

        # axes
        pygame.draw.line(self._screen, _GRAY,
                         (ox + margin, margin), (ox + margin, margin + speed_h))
        pygame.draw.line(self._screen, _GRAY,
                         (ox + margin, margin + speed_h),
                         (ox + margin + plot_w, margin + speed_h))

        if omega_target is not None:
            ty = int(margin + speed_h - (omega_target - lo_s) / (hi_s - lo_s) * speed_h)
            pygame.draw.line(self._screen, _YELLOW,
                             (ox + margin, ty), (ox + margin + plot_w, ty), 1)

        pts_n = series_points(arr_noisy, margin, speed_h, lo_s, hi_s)
        if len(pts_n) >= 2:
            pygame.draw.lines(self._screen, _NOISY, False, pts_n, 1)

        pts_t = series_points(arr_true, margin, speed_h, lo_s, hi_s)
        if len(pts_t) >= 2:
            pygame.draw.lines(self._screen, _ACCENT, False, pts_t, 2)

        for frac in (0.0, 0.5, 1.0):
            v  = lo_s + frac * (hi_s - lo_s)
            py = int(margin + speed_h - frac * speed_h)
            lbl = self._font_sm.render(f"{v:6.0f}", True, _GRAY)
            self._screen.blit(lbl, (ox + 2, py - 7))

        # legend
        pygame.draw.line(self._screen, _ACCENT, (ox + margin + 4, 12), (ox + margin + 24, 12), 2)
        self._screen.blit(self._font_sm.render("true",  True, _ACCENT), (ox + margin + 28, 6))
        pygame.draw.line(self._screen, _NOISY,  (ox + margin + 74, 12), (ox + margin + 94, 12), 1)
        self._screen.blit(self._font_sm.render("noisy", True, _NOISY),  (ox + margin + 98, 6))

        lbl = self._font_sm.render("Speed (rad/s)", True, _GRAY)
        self._screen.blit(lbl, (ox + margin, margin + speed_h + 4))

        # ---- voltage sub-plot ----
        arr_v  = list(self._voltage_buf)
        lo_v   = min(arr_v + [0.0])
        hi_v   = max(arr_v + [0.0])
        span_v = max(hi_v - lo_v, 1.0)
        lo_v  -= span_v * 0.1; hi_v += span_v * 0.1

        # axes
        pygame.draw.line(self._screen, _GRAY,
                         (ox + margin, split_y),
                         (ox + margin, split_y + voltage_h))
        pygame.draw.line(self._screen, _GRAY,
                         (ox + margin, split_y + voltage_h),
                         (ox + margin + plot_w, split_y + voltage_h))

        # zero line
        if lo_v < 0 < hi_v:
            zy = int(split_y + voltage_h - (0 - lo_v) / (hi_v - lo_v) * voltage_h)
            pygame.draw.line(self._screen, _GRAY,
                             (ox + margin, zy), (ox + margin + plot_w, zy), 1)

        pts_v = series_points(arr_v, split_y, voltage_h, lo_v, hi_v)
        if len(pts_v) >= 2:
            pygame.draw.lines(self._screen, _GREEN, False, pts_v, 2)

        for frac in (0.0, 0.5, 1.0):
            v  = lo_v + frac * (hi_v - lo_v)
            py = int(split_y + voltage_h - frac * voltage_h)
            lbl = self._font_sm.render(f"{v:+5.1f}", True, _GRAY)
            self._screen.blit(lbl, (ox + 2, py - 7))

        lbl = self._font_sm.render("Voltage (V)", True, _GRAY)
        self._screen.blit(lbl, (ox + margin, split_y + voltage_h + 4))

    def _draw_readout_panel(self, state, voltage: float, omega_target: float | None):
        """Right third: numeric readouts."""
        panel_w = self._W // 3
        ox = 2 * panel_w
        rect = pygame.Rect(ox, 0, panel_w, self._H)
        pygame.draw.rect(self._screen, _PANEL, rect)

        def row(label: str, value: str, colour, y: int):
            lbl = self._font_sm.render(label, True, _GRAY)
            val = self._font_lg.render(value, True, colour)
            self._screen.blit(lbl, (ox + 16, y))
            self._screen.blit(val, (ox + 16, y + 16))

        y = 24
        row("Time (s)",           f"{state.t:8.3f}",                          _WHITE,  y); y += 60
        row("Voltage (V)",        f"{voltage:+8.2f}",                          _YELLOW, y); y += 60
        row("Current (A)",        f"{state.i:+8.4f}",                          _GREEN,  y); y += 60
        row("Speed (rad/s)",      f"{state.omega:+8.3f}",                      _ACCENT, y); y += 60
        row("Position (deg)",     f"{math.degrees(state.theta) % 360:8.1f}",   _WHITE,  y); y += 60

        if omega_target is not None:
            err = state.omega - omega_target
            colour = _GREEN if abs(err) < 0.1 else _RED
            row("Target (rad/s)", f"{omega_target:+8.3f}",  _YELLOW, y); y += 60
            row("Error  (rad/s)", f"{err:+8.3f}",           colour,  y)
