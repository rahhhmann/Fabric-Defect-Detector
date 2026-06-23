"""
postprocess.py
-------------------------------------------------------------------
Purpose : Merge fragmented same-class detections that represent a
          single continuous defect — line-shaped defects such as a
          stitched "seam" or a "Warp_Weft" thread run — which
          standard NMS does not catch. The set of classes this
          applies to is configurable via PostprocessConfig.merge_classes
          (see api/main.py for the active config).

Background
----------
Standard YOLO NMS (governed by the `iou` parameter at inference
time) only suppresses boxes that overlap each other ABOVE the
iou threshold. For a long, curved, or diagonal defect like a
stitched seam or a warp/weft thread line, the model sometimes
proposes two (or more) spatially adjacent but barely-overlapping
boxes covering different segments of the same continuous line.
Their IoU can be well below 0.5, so lowering `iou` at inference
time does not merge them — and lowering it too far risks merging
genuinely separate nearby defects (false suppression), which is a
worse failure mode for a QC system than an occasional duplicate
label.

This module implements a class-aware, proximity-AND-orientation
based merge step that runs AFTER NMS, as a separate, explicit,
auditable stage. It only merges boxes of the SAME class that are
(a) touching or within `gap_threshold` pixels of an already-grouped
ORIGINAL box (not a grown union — see merge_fragmented_detections
docstring for why), and (b) consistently oriented/elongated in the
same general direction, when both boxes are elongated enough to
have a confident direction. Boxes from different classes (e.g. a
"hole" box overlapping a "seam" box, as seen in the diagnostic
images) are never merged, since they represent genuinely different
defect types occupying nearby regions.

Why orientation matters (not just gap distance):
  - Two same-class boxes can be geometrically close (small gap)
    while being fragments of two DIFFERENT, unrelated defects —
    e.g. two distinct stitch lines in different parts of a garment
    that happen to be within gap_threshold_px of each other via a
    chain of intermediate boxes. Gap distance alone can't tell
    these apart from genuine fragments of one continuous line.
  - A true fragmented line-defect (seam or warp/weft thread) tends
    to keep a consistent elongation direction across its pieces
    (they're slices of the same line). Unrelated nearby defects
    usually don't share that direction.
  - When a box isn't elongated enough to have a confident
    direction (close to square/blob), the orientation check is
    skipped for that pair and gap distance alone decides — this
    keeps the check from blocking merges it has no real signal
    about.

Why a separate module instead of tuning `iou` further:
  - iou is a single global knob shared by ALL 5 classes. Lowering

    it to fix seam fragmentation risks under-suppressing
    legitimately separate Stain/hole/Warp_Weft detections.
  - This merge step is class-scoped and distance-bounded, so it
    only acts on the specific failure pattern observed (seam
    fragmentation), leaving other classes' NMS behavior untouched.
-------------------------------------------------------------------
"""

from dataclasses import dataclass, field


@dataclass
class PostprocessConfig:
    """Tunable parameters for the merge step. Keep all magic
    numbers here instead of inline in business logic."""

    # Classes eligible for proximity merging. Only classes prone to
    # being split into multiple boxes along a continuous line
    # should be listed here (currently just "seam").
    merge_classes: set[str] = field(default_factory=lambda: {"seam"})

    # Max gap (in pixels, in the ORIGINAL image's coordinate space)
    # between two same-class boxes for them to be considered part
    # of the same continuous defect. Boxes that already overlap
    # have a gap of 0 and are always merged.
    gap_threshold_px: float = 20.0

    # Minimum elongation ratio (long side / short side) for a box
    # to be treated as "line-like" for orientation purposes. Boxes
    # below this ratio are closer to square/blob-shaped and skip
    # the orientation check (handled as a safe default — see
    # _orientation_compatible docstring).
    elongation_ratio_min: float = 1.5

    # Max allowed angular difference (in degrees) between two
    # line-like boxes' long-axis directions for them to be
    # considered "the same line" and eligible for merging.
    max_angle_diff_deg: float = 35.0


def _box_gap(box_a: list[float], box_b: list[float]) -> float:
    """
    Returns the Chebyshev-style gap between two axis-aligned boxes:
    0 if they overlap or touch, otherwise the max of the
    x-gap and y-gap between their nearest edges.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    gap_x = max(0.0, max(bx1, ax1) - min(bx2, ax2))
    gap_y = max(0.0, max(by1, ay1) - min(by2, ay2))

    return max(gap_x, gap_y)


def _box_orientation(box: list[float]) -> tuple[float, float] | None:
    """
    Estimate the long-axis "direction" of a box as (elongation_ratio, angle_deg).

    angle_deg is measured from the horizontal axis, in [0, 180):
        0   -> long axis is horizontal
        90  -> long axis is vertical

    Returns None if the box is not elongated enough (close to
    square) to have a meaningful direction — see
    `elongation_ratio_min` in PostprocessConfig. A box without a
    clear direction is treated as direction-agnostic (handled by
    the caller, not here).
    """
    x1, y1, x2, y2 = box
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)

    long_side = max(w, h)
    short_side = min(w, h)
    ratio = long_side / short_side

    # Axis-aligned bbox: angle is simply 0 (wide) or 90 (tall).
    # This is a coarse proxy for true stitch-line orientation
    # (we don't have the actual line/contour, only its bbox), but
    # it is enough to distinguish "this fragment runs roughly
    # left-right" from "this one runs roughly top-bottom or is in
    # a totally different part of the image at a different angle".
    angle_deg = 0.0 if w >= h else 90.0

    return ratio, angle_deg


def _orientation_compatible(
    box_a: list[float],
    box_b: list[float],
    config: "PostprocessConfig",
) -> bool:
    """
    Decide whether two boxes are plausibly fragments of the SAME
    continuous line-shaped defect, based on shape + alignment —
    not just proximity.

    Rationale: two same-class boxes can be geometrically close
    (small gap) yet belong to clearly different defects — e.g. two
    separate, distant stitch lines on a garment that happen to be
    within `gap_threshold_px` of each other via a chain of
    intermediate boxes. Proximity alone can't tell them apart.
    Requiring similar elongation + similar long-axis direction
    catches the common real-world pattern: a single seam
    fragmented by the detector keeps a consistent, strongly
    elongated (line-like) shape across its pieces, while spurious
    or imprecise fragments of unrelated nearby defects tend to be
    closer to square.

    Default-to-BLOCK for non-elongated boxes (validated against
    real failure cases, not assumed): a box that is NOT clearly
    elongated (ratio below `elongation_ratio_min`) is treated as
    untrustworthy shape evidence for "this is a slice of a long
    line" and merging is blocked for that pair, even if the gap is
    small. This is intentionally the conservative direction for a
    QC system — failing to merge two genuine fragments produces an
    extra box (a minor annoyance, caught by a human reviewer over-
    splitting a defect into two regions); wrongly merging produces
    one giant box that can swallow most of the frame and obscure
    or misrepresent the actual defect location/extent, which is
    the worse failure mode we're specifically trying to eliminate.
    """
    ratio_a, angle_a = _box_orientation(box_a)
    ratio_b, angle_b = _box_orientation(box_b)

    if ratio_a < config.elongation_ratio_min or ratio_b < config.elongation_ratio_min:
        return False  # not clearly line-like — don't merge on shape grounds

    # Compare long-axis direction. Since our coarse angle is only
    # ever 0 or 90 (axis-aligned proxy), "compatible" simply means
    # both fragments run the same general way (both wide, or both
    # tall). A genuinely diagonal seam tends to produce fragments
    # that are each elongated in a consistent way relative to one
    # another even under this coarse proxy; a wide fragment paired
    # with a tall, unrelated fragment elsewhere in the frame is the
    # pattern we want to block.
    angle_diff = abs(angle_a - angle_b)
    angle_diff = min(angle_diff, 180.0 - angle_diff)

    return angle_diff <= config.max_angle_diff_deg


def merge_fragmented_detections(
    defects: list[dict],
    config: PostprocessConfig | None = None,
) -> list[dict]:
    """
    Merge same-class detections that are likely fragments of a
    single continuous defect (e.g. a seam split across 2+ boxes).

    Args:
        defects: list of dicts as produced by run_inference(), each
                 with keys "type", "confidence", "bbox" ([x1,y1,x2,y2]).
        config:  PostprocessConfig. Uses defaults if not provided.

    Returns:
        A new list of defect dicts, with eligible same-class
        fragments merged into a single bounding box (the union of
        the merged boxes) and the confidence set to the max of the
        merged group. Defects of non-merge-eligible classes, or
        with no nearby same-class neighbor, are returned unchanged.
    """
    if config is None:
        config = PostprocessConfig()

    n = len(defects)
    used = [False] * n
    merged_defects: list[dict] = []

    for i in range(n):
        if used[i]:
            continue

        current = defects[i]
        used[i] = True

        if current["type"] not in config.merge_classes:
            merged_defects.append(current)
            continue

        # group = indices of all defects merged into this cluster so far.
        # Gap and orientation are always measured against the ORIGINAL
        # boxes in this group (never against the growing union box),
        # so one large/loose fragment can't "tow in" a distant box just
        # because the union happens to have grown close to it. This
        # prevents the transitive chain-growth failure mode where
        # A merges with B, the union then reaches C, and a single
        # detection ends up swallowing the whole image.
        group = [i]
        merged_box = list(current["bbox"])
        merged_conf = current["confidence"]
        changed = True

        while changed:
            changed = False
            for j in range(n):
                if used[j] or defects[j]["type"] != current["type"]:
                    continue

                candidate_box = defects[j]["bbox"]

                # Must be close to (and oriented consistently with)
                # at least one ORIGINAL box already in the group —
                # not just the accumulated union.
                is_candidate = False
                for k in group:
                    member_box = defects[k]["bbox"]
                    if _box_gap(member_box, candidate_box) > config.gap_threshold_px:
                        continue
                    if not _orientation_compatible(member_box, candidate_box, config):
                        continue
                    is_candidate = True
                    break

                if not is_candidate:
                    continue

                bx1, by1, bx2, by2 = candidate_box
                merged_box = [
                    min(merged_box[0], bx1),
                    min(merged_box[1], by1),
                    max(merged_box[2], bx2),
                    max(merged_box[3], by2),
                ]
                merged_conf = max(merged_conf, defects[j]["confidence"])
                used[j] = True
                group.append(j)
                changed = True

        merged_defects.append({
            "type": current["type"],
            "confidence": merged_conf,
            "bbox": [round(v, 1) for v in merged_box],
        })

    return merged_defects