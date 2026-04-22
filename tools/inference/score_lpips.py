"""LPIPS scoring pass over a predictions.jsonl — step 2 of checkpoint eval.

Reads the JSONL produced by `generate_mathcanvas.py`, extracts <svg>…</svg>
from both the prediction and the gold assistant text, renders each via
resvg-py at 512x512 (matches the base image resolution — downsampling to
the canonical LPIPS 256 loses stroke-width 1-2 px red dashed lines), and
computes LPIPS between the two.

Render failures (no <svg> tag or resvg error) score LPIPS = 1.0 (max), so
mean_lpips_all is monotonic in render success rate. `mean_lpips_render_ok_only`
is reported separately for quality comparison among the successfully-rendered
subset.

Example:
    uv run python tools/inference/score_lpips.py \\
        --in-dir /ephemeral/vuvlm/nemo-RL-render-reward/evals/step_275

Writes `summary.json` (and `scored.jsonl`) next to `predictions.jsonl`.
"""

from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path

import numpy as np
import resvg_py
import torch
from PIL import Image


_SVG_RE = re.compile(r"<svg\b[^>]*>.*?</svg>", re.DOTALL | re.IGNORECASE)
# Match a ```...``` code fence (optional language tag like svg/xml/html). The
# body may contain partial SVG if the model ran out of budget before </svg>.
_FENCE_RE = re.compile(
    r"```(?:svg|xml|html)?\s*(.*?)```", re.DOTALL | re.IGNORECASE
)
# Unterminated fence — opening ``` with no closing fence. Keeps everything
# after the opener so a truncated fenced output is still recoverable.
_OPEN_FENCE_RE = re.compile(
    r"```(?:svg|xml|html)?\s*(<svg\b.*)", re.DOTALL | re.IGNORECASE
)
# Open <svg> tag with no </svg> close — same recovery idea for non-fenced
# truncations.
_OPEN_SVG_RE = re.compile(r"(<svg\b[^>]*>.*)", re.DOTALL | re.IGNORECASE)


def _close_svg(svg_open: str) -> str:
    """Append a `</svg>` if missing — resvg-py is unforgiving about truncation."""
    return svg_open if "</svg>" in svg_open.lower() else svg_open + "</svg>"


def extract_svg(text: str) -> str | None:
    """Best-effort pull of an SVG string from arbitrary LLM output.

    Priority order:
      1. Last full `<svg>...</svg>` block anywhere in the text (including
         inside fences — the regex crosses fence markers fine with DOTALL).
      2. Last fenced block whose body starts with `<svg` even when the
         `</svg>` close is missing (truncated within the fence); synthetic
         close appended so resvg-py can render.
      3. Trailing `<svg>` with no close (pure truncation, no fence) —
         same synthetic-close recovery.
    Returns None if none of the above hit.
    """
    if not text:
        return None

    # (1) prefer complete <svg>...</svg>, searching inside fences first for
    # tighter capture (avoids trailing non-SVG prose appearing in #svg.
    candidates: list[str] = []
    for fence_body in _FENCE_RE.findall(text):
        candidates.extend(_SVG_RE.findall(fence_body))
    candidates.extend(_SVG_RE.findall(text))
    if candidates:
        return candidates[-1]

    # (2) unterminated fenced <svg> — take from opener to EOF, synthetic close.
    m = _OPEN_FENCE_RE.search(text)
    if m:
        return _close_svg(m.group(1).rstrip("`").rstrip())

    # (3) unterminated bare <svg> — same recovery.
    m = _OPEN_SVG_RE.search(text)
    if m:
        return _close_svg(m.group(1).rstrip("`").rstrip())

    return None


def render_svg_to_pil(svg_text: str, target_size: int) -> Image.Image | None:
    """Render SVG → PIL RGB with WHITE background.

    resvg-py returns RGBA where the canvas is transparent. Naive `.convert("RGB")`
    fills transparent pixels with black, which hides the diagram's black strokes —
    the GT mathcanvas SVGs draw black lines on a transparent canvas, so we'd end
    up comparing "black on black" against the model's output. Composite onto a
    white background first so black and red strokes are both visible under LPIPS.
    """
    try:
        png = resvg_py.svg_to_bytes(svg_string=svg_text)
    except Exception:
        return None
    rgba = Image.open(io.BytesIO(bytes(png))).convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    img = Image.alpha_composite(bg, rgba).convert("RGB")
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    return img


def pil_to_lpips_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    t = torch.from_numpy(arr).unsqueeze(0).to(device)
    return t * 2.0 - 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True,
                    help="Directory containing predictions.jsonl from "
                         "generate_mathcanvas.py.")
    ap.add_argument("--predictions", type=Path, default=None,
                    help="Override the predictions path (default: "
                         "<in-dir>/predictions.jsonl)")
    ap.add_argument("--save-renders", action="store_true",
                    help="Save per-sample pred/gt PNGs under <in-dir>/renders/")
    ap.add_argument("--lpips-net", choices=("alex", "vgg"), default="alex")
    ap.add_argument("--image-size", type=int, default=512,
                    help="LPIPS comparison resolution (matches base-diagram "
                         "resolution of 512).")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    in_dir: Path = args.in_dir
    preds_path = args.predictions or (in_dir / "predictions.jsonl")
    if not preds_path.exists():
        raise SystemExit(f"predictions file not found: {preds_path}")

    # Read predictions
    records: list[dict] = []
    with preds_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[score] loaded {len(records)} predictions from {preds_path}", flush=True)

    if args.save_renders:
        (in_dir / "renders").mkdir(exist_ok=True)

    # LPIPS net
    import lpips

    device = torch.device(args.device)
    print(f"[score] loading LPIPS net={args.lpips_net} on {device}", flush=True)
    lpips_model = lpips.LPIPS(net=args.lpips_net, verbose=False).to(device).eval()

    rows: list[dict] = []
    lpips_vals: list[float] = []
    n_render_ok = 0

    for rec in records:
        idx = rec["idx"]
        pred_svg = extract_svg(rec.get("pred_text", ""))
        gt_raw = rec.get("gt_svg", "")
        gt_svg = extract_svg(gt_raw) or gt_raw  # GT is already an SVG string

        pred_pil = render_svg_to_pil(pred_svg, args.image_size) if pred_svg else None
        gt_pil = render_svg_to_pil(gt_svg, args.image_size) if gt_svg else None

        if pred_pil is None or gt_pil is None:
            lpips_score = 1.0
            render_ok = False
            render_reason = (
                "no_svg_tag_in_pred" if pred_svg is None
                else "pred_render_error" if pred_pil is None
                else "gt_render_error"
            )
        else:
            n_render_ok += 1
            render_ok = True
            render_reason = "ok"
            with torch.no_grad():
                pt = pil_to_lpips_tensor(pred_pil, device)
                gt = pil_to_lpips_tensor(gt_pil, device)
                lpips_score = float(lpips_model(pt, gt).item())

        lpips_vals.append(lpips_score)
        rows.append({
            "idx": idx,
            "image_path": rec.get("image_path"),
            "instruction": rec.get("instruction"),
            "pred_text_head": (rec.get("pred_text") or "")[:400].replace("\n", " "),
            "pred_output_tokens": rec.get("output_tokens"),
            "finish_reason": rec.get("finish_reason"),
            "render_ok": render_ok,
            "render_reason": render_reason,
            "lpips": lpips_score,
        })

        if args.save_renders and render_ok:
            pred_pil.save(in_dir / "renders" / f"{idx:03d}_pred.png")
            gt_pil.save(in_dir / "renders" / f"{idx:03d}_gt.png")

        print(
            f"[score] {idx:3d}/{len(records)}  lpips={lpips_score:.4f}  "
            f"render_ok={render_ok}  ({render_reason})",
            flush=True,
        )

    mean_lpips = float(np.mean(lpips_vals)) if lpips_vals else 1.0
    mean_lpips_render_ok = (
        float(np.mean([r["lpips"] for r in rows if r["render_ok"]]))
        if n_render_ok
        else None
    )

    # Carry generation metadata into the summary if present.
    gen_info_path = in_dir / "generation_info.json"
    gen_info = json.loads(gen_info_path.read_text()) if gen_info_path.exists() else {}

    summary = {
        "n": len(rows),
        "n_render_ok": n_render_ok,
        "render_ok_rate": n_render_ok / max(1, len(rows)),
        "mean_lpips_all": mean_lpips,
        "mean_lpips_render_ok_only": mean_lpips_render_ok,
        "lpips_net": args.lpips_net,
        "image_size": args.image_size,
        "generation_info": gen_info,
    }

    with (in_dir / "scored.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (in_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n[score] SUMMARY", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
