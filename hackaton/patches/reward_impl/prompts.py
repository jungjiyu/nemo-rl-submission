"""Prompt templates for Geometry3K + SVG-augmented reasoning.

Three prompt families:
1. baseline_direct  — solve from image alone (UniG2U baseline)
2. svg_generate     — generate auxiliary-line SVG given image + question
3. gta_two_image    — solve given original + auxiliary diagram (GtA stage 2)
"""

from __future__ import annotations

from hackathon.data.geometry3k import GeoProblem


def baseline_direct(p: GeoProblem) -> str:
    """Direct MCQ solve from image — UniG2U baseline prompt."""
    opts = "\n".join(f"({chr(ord('A') + i)}) {c}" for i, c in enumerate(p.choices))
    return (
        "Look at the geometry diagram and solve the problem.\n\n"
        f"Question: {p.problem_text}\n"
        f"Options:\n{opts}\n\n"
        "Answer the question with the option's letter from the given choices directly."
    )


SVG_GENERATE_TEMPLATE = """\
You are given a geometry problem diagram. Generate an SVG that REPRODUCES the original \
diagram and ADDS auxiliary construction lines that would help solve the problem.

Requirements:
1. Reproduce the original diagram elements as SVG (black solid lines, fill=\"none\").
2. Add auxiliary construction lines in red dashed style: \
stroke=\"red\" stroke-dasharray=\"5,5\" stroke-width=\"2\".
3. Label any new points with <text> elements (font-size=\"14\").
4. Use viewBox=\"0 0 {width} {height}\" matching the original image.
5. Wrap your output in a single <svg>...</svg> block. The SVG must be valid and renderable.

Problem: {problem}
"""


def svg_generate(p: GeoProblem) -> str:
    w, h = p.image.size
    return SVG_GENERATE_TEMPLATE.format(width=w, height=h, problem=p.problem_text)


def gta_two_image(p: GeoProblem) -> str:
    """GtA stage 2: solve using original + auxiliary diagram."""
    opts = "\n".join(f"({chr(ord('A') + i)}) {c}" for i, c in enumerate(p.choices))
    return (
        "You are given TWO images:\n"
        "1) ORIGINAL DIAGRAM: the geometry problem diagram\n"
        "2) AUXILIARY DIAGRAM: the same diagram with helpful auxiliary constructions added\n\n"
        "Use both diagrams to solve the problem.\n\n"
        f"Question: {p.problem_text}\n"
        f"Options:\n{opts}\n\n"
        "Answer the question with the option's letter from the given choices directly."
    )
