"""Render our own Plotly chart as a PNG snapshot for annotation overlay.

The most reliable screenshot backend — uses our actual data + kaleido image
export. No TV Desktop required, no external API, works fully offline inside
Docker. Coordinates produced here match exactly what tv_annotate uses for
PIL overlay (because both work off the same OHLC series).

Usage:
    path = render_tf_snapshot(
        ticker="BTC-USD",
        tf="4h",
        prices=close_series,
        major_swings=swings_list,
        out_path="predictions/screenshots/abc123_4h.png",
    )

Returns the path on success, None if rendering failed (kaleido missing,
disk full, etc.). Best-effort — never raises.
"""
from __future__ import annotations

import os
from typing import Any

import pandas as pd

try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:                                # pragma: no cover
    _PLOTLY_AVAILABLE = False
    go = None  # type: ignore


def _has_kaleido() -> bool:
    """Check if kaleido is installed (required for fig.write_image)."""
    try:
        import kaleido  # noqa: F401
        return True
    except ImportError:
        return False


def _has_matplotlib() -> bool:
    """Check if matplotlib is available — Docker fallback when kaleido
    fails (e.g. when Chromium is absent in the image)."""
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


def _render_via_matplotlib(ticker: str, tf: str,
                            prices, major_swings,
                            minor_swings, out_path: str,
                            width: int = 1600, height: int = 900) -> str | None:
    """Pure-matplotlib renderer — no browser, no kaleido, no JS engine.

    Works in any Docker container that has matplotlib (already a transitive
    dependency of plotly + streamlit). Matches the dark style we annotate
    with PIL: same chart-area paddings, same color palette.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")           # no display
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        bg_color = "#0F172A"
        line_color = "#FFFFFF"
        major_color = "#00D4FF"
        minor_color = "#94A3B8"

        dpi = 100
        fig, ax = plt.subplots(
            figsize=(width / dpi, height / dpi), dpi=dpi,
            facecolor=bg_color,
        )
        ax.set_facecolor(bg_color)

        # Close-price line
        try:
            ax.plot(prices.index, prices.values,
                    color=line_color, linewidth=1.2)
        except Exception:
            return None

        # Minor zigzag
        if minor_swings:
            try:
                ax.plot([s.index for s in minor_swings],
                        [s.price for s in minor_swings],
                        color=minor_color, linewidth=0.7, alpha=0.45)
            except Exception:
                pass

        # Major zigzag
        if major_swings:
            try:
                xs = [s.index for s in major_swings]
                ys = [s.price for s in major_swings]
                ax.plot(xs, ys, color=major_color, linewidth=1.8)
                ax.scatter(xs, ys, color=major_color, s=24, zorder=3)
            except Exception:
                pass

        # Title + styling
        ax.set_title(
            f"{ticker.upper()}  ·  {tf.upper()}  ·  {len(prices)} bars",
            loc="left", color="#E2E8F0", fontsize=12, fontweight="bold",
            pad=12,
        )
        ax.tick_params(colors="#94A3B8", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#475569")
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel("Price", color="#94A3B8", fontsize=10)
        ax.grid(True, color="#1E293B", linewidth=0.5)

        # Date format
        try:
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        except Exception:
            pass

        # Margins match TVLayout("dark_default")
        fig.subplots_adjust(left=60/width, right=1 - 90/width,
                             top=1 - 80/height, bottom=110/height)

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        tmp = out_path + ".tmp.png"
        fig.savefig(tmp, dpi=dpi, facecolor=bg_color, edgecolor="none")
        plt.close(fig)

        if os.path.isfile(tmp) and os.path.getsize(tmp) > 100:
            try:
                if os.path.isfile(out_path):
                    os.remove(out_path)
                os.rename(tmp, out_path)
                return out_path
            except OSError as e:
                import sys
                print(f"[plotly_snapshot] mpl rename failed: {e}",
                      file=sys.stderr)
                return None
        return None
    except Exception as e:
        import sys
        print(f"[plotly_snapshot] matplotlib render error: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        return None


def render_tf_snapshot(
    ticker: str,
    tf: str,
    prices: pd.Series,
    major_swings: list | None = None,
    minor_swings: list | None = None,
    out_path: str = "",
    *,
    width: int = 1600,
    height: int = 900,
) -> str | None:
    """Render one TF as a PNG using our own Plotly figure.

    Includes:
      * close-price line
      * minor zigzag (thin gray, if provided)
      * major zigzag (bold cyan, if provided)
      * title with ticker + TF + bar count

    The image is intentionally minimal — no wave labels, no Claude overlay
    here. Those get painted on top by tv_annotate later. We want a clean
    canvas with structure so the PIL annotations are clearly visible.

    Returns the absolute path on success; None on any failure.
    """
    if len(prices) == 0:
        import sys
        print("[plotly_snapshot] empty prices", file=sys.stderr)
        return None
    if not out_path:
        return None

    # If plotly+kaleido aren't usable (e.g. Chromium missing in Docker),
    # fall back to matplotlib straight away. matplotlib is a transitive
    # dependency and works in any container without a browser.
    if not _PLOTLY_AVAILABLE or not _has_kaleido():
        import sys
        if not _has_kaleido():
            print("[plotly_snapshot] kaleido missing → matplotlib fallback",
                  file=sys.stderr)
        if _has_matplotlib():
            return _render_via_matplotlib(
                ticker, tf, prices, major_swings, minor_swings,
                out_path, width, height,
            )
        print("[plotly_snapshot] neither kaleido nor matplotlib available",
              file=sys.stderr)
        return None

    try:
        fig = go.Figure()

        # Background — match the screenshot style we know how to annotate
        bg_color = "#0F172A"
        line_color = "#FFFFFF"

        fig.add_trace(go.Scatter(
            x=prices.index, y=prices.values,
            mode="lines", line=dict(color=line_color, width=1.4),
            showlegend=False, hoverinfo="skip",
        ))

        # Minor zigzag — light, supplementary
        if minor_swings:
            try:
                mx = [s.index for s in minor_swings]
                my = [s.price for s in minor_swings]
                fig.add_trace(go.Scatter(
                    x=mx, y=my,
                    mode="lines", line=dict(color="#94A3B8", width=0.8),
                    opacity=0.45, showlegend=False, hoverinfo="skip",
                ))
            except Exception:
                pass

        # Major zigzag — bold cyan, the "structural" backbone
        if major_swings:
            try:
                mx = [s.index for s in major_swings]
                my = [s.price for s in major_swings]
                fig.add_trace(go.Scatter(
                    x=mx, y=my,
                    mode="lines+markers",
                    line=dict(color="#00D4FF", width=2.0),
                    marker=dict(size=7, color="#00D4FF"),
                    showlegend=False, hoverinfo="skip",
                ))
            except Exception:
                pass

        fig.update_layout(
            width=width, height=height,
            plot_bgcolor=bg_color, paper_bgcolor=bg_color,
            font=dict(color="#E2E8F0", size=12, family="Arial"),
            margin=dict(l=60, t=80, r=90, b=110),
            title=dict(
                text=f"<b>{ticker.upper()}</b>  ·  {tf.upper()}  ·  "
                     f"{len(prices)} bars",
                x=0.01, xanchor="left",
                font=dict(size=14, color="#E2E8F0"),
            ),
            xaxis=dict(
                gridcolor="#1E293B", linecolor="#475569",
                tickfont=dict(color="#94A3B8"),
                showgrid=True, zeroline=False,
            ),
            yaxis=dict(
                gridcolor="#1E293B", linecolor="#475569",
                tickfont=dict(color="#94A3B8"),
                side="right",
                showgrid=True, zeroline=False,
                title=dict(text="Price",
                            font=dict(color="#94A3B8", size=11)),
            ),
            hovermode=False,
        )

        # Render to PNG via kaleido. Write to a temp + rename to avoid
        # half-written files being read by annotate.
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        tmp = out_path + ".tmp.png"
        try:
            fig.write_image(tmp, format="png", width=width, height=height,
                              scale=1)
        except Exception as e:
            # kaleido often fails silently or with cryptic errors when
            # Chromium isn't bundled. Fall back to matplotlib straight away.
            import sys
            print(f"[plotly_snapshot] kaleido write_image raised "
                  f"{type(e).__name__}: {e} → matplotlib fallback",
                  file=sys.stderr)
            if _has_matplotlib():
                return _render_via_matplotlib(
                    ticker, tf, prices, major_swings, minor_swings,
                    out_path, width, height,
                )
            return None

        # Some kaleido configs return without raising but produce a 0-byte
        # or tiny file. Treat anything under 1 KB as failure → matplotlib.
        if os.path.isfile(tmp) and os.path.getsize(tmp) >= 1024:
            try:
                if os.path.isfile(out_path):
                    os.remove(out_path)
                os.rename(tmp, out_path)
                return out_path
            except OSError as e:
                import sys
                print(f"[plotly_snapshot] rename failed: {e}", file=sys.stderr)
                return None

        # Kaleido silently produced nothing — try matplotlib.
        import sys
        size = os.path.getsize(tmp) if os.path.isfile(tmp) else 0
        print(f"[plotly_snapshot] kaleido produced {size}-byte file "
              f"→ matplotlib fallback", file=sys.stderr)
        try:
            os.remove(tmp)
        except OSError:
            pass
        if _has_matplotlib():
            return _render_via_matplotlib(
                ticker, tf, prices, major_swings, minor_swings,
                out_path, width, height,
            )
        return None
    except Exception as e:
        import sys
        print(f"[plotly_snapshot] render error: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        if _has_matplotlib():
            return _render_via_matplotlib(
                ticker, tf, prices, major_swings, minor_swings,
                out_path, width, height,
            )
        return None
