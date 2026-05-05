"""
episode_gif_pipeline.py

Helper utilities for RoboFlamingo/CALVIN eval GIFs.

Purpose:
- Keep most logic out of robot_flamingo/eval/eval_utils.py.
- Group subtask GIFs by sequence id.
- Merge files like:
    0-0-lift_red_block_table-succ.gif
    0-1-stack_block-succ.gif
  into:
    seq_000000/sequence_000000_merged.gif
- Preserve original GIF frame durations as closely as possible.
- Optionally remove intermediate per-subtask GIFs after merging.
- Write metadata for later Gemini/neuron-activation alignment.

Expected original filename format:
    {sequence_i}-{subtask_i}-{subtask}-{succ_or_fail}.gif
"""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import subprocess
import sys
from PIL import Image, ImageSequence


SUBTASK_GIF_RE = re.compile(
    r"^(?P<sequence_i>\d+)-(?P<subtask_i>\d+)-(?P<subtask>.+)-(?P<status>succ|fail)\.gif$"
)


@dataclass(frozen=True)
class SubtaskGif:
    sequence_i: int
    subtask_i: int
    subtask: str
    status: str
    path: Path


@dataclass(frozen=True)
class FrameMapEntry:
    global_frame_i: int
    sequence_i: int
    subtask_i: int
    subtask: str
    status: str
    local_frame_i: int
    duration_ms: int
    source_file: str


def parse_subtask_gif(path: Path) -> Optional[SubtaskGif]:
    """
    Parse RoboFlamingo subtask GIF filenames.

    Example:
        0-1-stack_block-succ.gif
    """
    match = SUBTASK_GIF_RE.match(path.name)
    if not match:
        return None

    return SubtaskGif(
        sequence_i=int(match.group("sequence_i")),
        subtask_i=int(match.group("subtask_i")),
        subtask=match.group("subtask"),
        status=match.group("status"),
        path=path,
    )


def find_subtask_gifs(
    eval_log_dir: str | Path,
    sequence_i: Optional[int] = None,
) -> List[SubtaskGif]:
    """
    Find all original subtask GIFs in eval_log_dir.

    This intentionally only looks in the root eval_log_dir and ignores seq_XXXXXX folders.
    """
    eval_log_dir = Path(eval_log_dir)
    found: List[SubtaskGif] = []

    for path in eval_log_dir.glob("*.gif"):
        parsed = parse_subtask_gif(path)
        if parsed is None:
            continue
        if sequence_i is not None and parsed.sequence_i != sequence_i:
            continue
        found.append(parsed)

    found.sort(key=lambda x: (x.sequence_i, x.subtask_i))
    return found


def group_by_sequence(gifs: Iterable[SubtaskGif]) -> Dict[int, List[SubtaskGif]]:
    groups: Dict[int, List[SubtaskGif]] = {}

    for gif in gifs:
        groups.setdefault(gif.sequence_i, []).append(gif)

    for seq in groups:
        groups[seq].sort(key=lambda x: x.subtask_i)

    return groups


def wait_until_file_stable(
    path: str | Path,
    checks: int = 3,
    delay: float = 0.5,
) -> None:
    """
    Wait until file size stops changing.

    Useful when MoviePy has just written a GIF and the next function immediately
    tries to read it.
    """
    path = Path(path)
    last_size = -1
    stable_count = 0

    while stable_count < checks:
        if not path.exists():
            time.sleep(delay)
            continue

        size = path.stat().st_size

        if size == last_size:
            stable_count += 1
        else:
            stable_count = 0
            last_size = size

        time.sleep(delay)


def make_sequence_dir(eval_log_dir: str | Path, sequence_i: int) -> Path:
    seq_dir = Path(eval_log_dir) / f"seq_{sequence_i:06d}"
    seq_dir.mkdir(parents=True, exist_ok=True)
    return seq_dir


def copy_or_move_subtask_gifs(
    subtasks: Sequence[SubtaskGif],
    seq_dir: Path,
    move: bool = False,
) -> List[Path]:
    """
    Copy or move subtask GIFs into the per-sequence folder with cleaner names.

    Example:
        0-1-stack_block-succ.gif
    becomes:
        01-stack_block-succ.gif
    """
    output_paths: List[Path] = []

    for item in subtasks:
        wait_until_file_stable(item.path)

        clean_name = f"{item.subtask_i:02d}-{item.subtask}-{item.status}.gif"
        target = seq_dir / clean_name

        if item.path.resolve() == target.resolve():
            output_paths.append(target)
            continue

        if target.exists():
            target.unlink()

        if move:
            shutil.move(str(item.path), str(target))
        else:
            shutil.copy2(str(item.path), str(target))

        output_paths.append(target)

    return output_paths


def read_gif_frames_and_durations(
    gif_path: str | Path,
    fallback_duration_ms: int = 33,
) -> Tuple[List[Image.Image], List[int]]:
    """
    Read GIF frames while preserving per-frame duration metadata.

    This is the key function for keeping merged playback speed close to the
    original subtask GIFs.

    GIF duration is stored in milliseconds by Pillow, but GIF itself has coarse
    timing resolution, so exactness is limited by the GIF format and viewer.
    """
    gif_path = Path(gif_path)
    wait_until_file_stable(gif_path)

    frames: List[Image.Image] = []
    durations: List[int] = []

    with Image.open(gif_path) as img:
        for frame in ImageSequence.Iterator(img):
            duration = frame.info.get("duration", img.info.get("duration", fallback_duration_ms))

            if duration is None or duration <= 0:
                duration = fallback_duration_ms

            # Use RGBA while merging to avoid palette corruption between GIFs.
            frames.append(frame.convert("RGBA"))
            durations.append(int(duration))

    return frames, durations


def merge_gifs_preserve_timing(
    input_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    add_pause_between_gifs: bool = False,
    pause_ms: int = 250,
    fallback_duration_ms: int = 33,
) -> Tuple[Path, List[FrameMapEntry]]:
    """
    Merge GIFs while preserving the original per-frame durations.

    This avoids the bad behavior of decoding frames and then re-saving with a
    new fps value. The output timing should match the source GIFs much more
    closely.

    Returns:
        merged_path, frame_map
    """
    output_path = Path(output_path)
    all_frames: List[Image.Image] = []
    all_durations: List[int] = []
    frame_map: List[FrameMapEntry] = []

    global_frame_i = 0

    for input_path in input_paths:
        input_path = Path(input_path)
        parsed = parse_clean_subtask_gif(input_path) or parse_subtask_gif(input_path)

        if parsed is None:
            raise ValueError(f"Could not parse subtask GIF filename: {input_path.name}")

        frames, durations = read_gif_frames_and_durations(
            input_path,
            fallback_duration_ms=fallback_duration_ms,
        )

        if not frames:
            continue

        for local_frame_i, (frame, duration_ms) in enumerate(zip(frames, durations)):
            all_frames.append(frame)
            all_durations.append(duration_ms)

            frame_map.append(
                FrameMapEntry(
                    global_frame_i=global_frame_i,
                    sequence_i=parsed.sequence_i,
                    subtask_i=parsed.subtask_i,
                    subtask=parsed.subtask,
                    status=parsed.status,
                    local_frame_i=local_frame_i,
                    duration_ms=duration_ms,
                    source_file=input_path.name,
                )
            )

            global_frame_i += 1

        if add_pause_between_gifs:
            all_frames.append(frames[-1].copy())
            all_durations.append(int(pause_ms))

            frame_map.append(
                FrameMapEntry(
                    global_frame_i=global_frame_i,
                    sequence_i=parsed.sequence_i,
                    subtask_i=parsed.subtask_i,
                    subtask=parsed.subtask,
                    status=parsed.status,
                    local_frame_i=len(frames) - 1,
                    duration_ms=int(pause_ms),
                    source_file=input_path.name,
                )
            )

            global_frame_i += 1

    if not all_frames:
        raise ValueError(f"No frames found while merging into {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to palette mode at the end. This keeps inter-source colors more stable
    # than preserving each source GIF's separate palette.
    first = all_frames[0].convert("P", palette=Image.ADAPTIVE)
    rest = [frame.convert("P", palette=Image.ADAPTIVE) for frame in all_frames[1:]]

    first.save(
        output_path,
        save_all=True,
        append_images=rest,
        duration=all_durations,
        loop=0,
        optimize=False,
        disposal=2,
    )

    return output_path, frame_map


CLEAN_SUBTASK_GIF_RE = re.compile(
    r"^(?P<subtask_i>\d+)-(?P<subtask>.+)-(?P<status>succ|fail)\.gif$"
)


def parse_clean_subtask_gif(path: Path) -> Optional[SubtaskGif]:
    """
    Parse cleaned per-sequence filenames like:
        00-lift_red_block_table-succ.gif

    The sequence id is inferred from parent folder:
        seq_000000
    """
    match = CLEAN_SUBTASK_GIF_RE.match(path.name)
    if not match:
        return None

    parent_match = re.match(r"^seq_(?P<sequence_i>\d+)$", path.parent.name)
    if not parent_match:
        return None

    return SubtaskGif(
        sequence_i=int(parent_match.group("sequence_i")),
        subtask_i=int(match.group("subtask_i")),
        subtask=match.group("subtask"),
        status=match.group("status"),
        path=path,
    )


def write_metadata(
    seq_dir: Path,
    sequence_i: int,
    subtasks: Sequence[SubtaskGif],
    merged_path: Path,
    frame_map: Sequence[FrameMapEntry],
    label: Optional[str] = None,
) -> Path:
    """
    Write metadata for debugging, Gemini classification, and activation alignment.
    """
    total_duration_ms = sum(entry.duration_ms for entry in frame_map)

    meta = {
        "sequence_i": sequence_i,
        "label": label,
        "merged_gif": merged_path.name,
        "num_frames": len(frame_map),
        "total_duration_ms": total_duration_ms,
        "total_duration_seconds": total_duration_ms / 1000.0,
        "timing_source": "preserved_from_original_subtask_gifs",
        "subtasks": [
            {
                "subtask_i": item.subtask_i,
                "subtask": item.subtask,
                "status": item.status,
                "source_file": item.path.name,
            }
            for item in subtasks
        ],
    }

    meta_path = seq_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta_path


def write_frame_map(
    seq_dir: Path,
    frame_map: Sequence[FrameMapEntry],
) -> Path:
    """
    Write frame-level timing map.

    This is useful later for neuron activation alignment:
        global_frame_i <-> subtask_i <-> local_frame_i <-> duration_ms
    """
    path = seq_dir / "frame_map.json"

    data = [
        {
            "global_frame_i": entry.global_frame_i,
            "sequence_i": entry.sequence_i,
            "subtask_i": entry.subtask_i,
            "subtask": entry.subtask,
            "status": entry.status,
            "local_frame_i": entry.local_frame_i,
            "duration_ms": entry.duration_ms,
            "source_file": entry.source_file,
        }
        for entry in frame_map
    ]

    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path

def run_gemini_eval(
    *,
    video_path: Path,
    metadata_path: Path,
    frame_map_path: Path,
    output_path: Path,
    gemini_eval_script: str | Path = "gemini_eval.py",
    timeout_seconds: int = 300,
) -> Optional[dict]:
    """
    Call gemini_eval.py as a separate process.

    This keeps API/video-evaluation logic out of this file and avoids loading
    the video into this process. gemini_eval.py receives paths only.
    """
    gemini_eval_script = Path(gemini_eval_script)

    # Resolve relative script paths relative to this helper file, not the
    # current working directory of the eval process. This prevents errors like:
    #     python: can't open file 'gemini_eval.py'
    if not gemini_eval_script.is_absolute():
        gemini_eval_script = Path(__file__).resolve().parent / gemini_eval_script

    gemini_python = "/home/timo/miniconda3/envs/gemini_eval/bin/python"

    cmd = [
        gemini_python,
        str(gemini_eval_script),
        "--video",
        str(video_path),
        "--metadata",
        str(metadata_path),
        "--frame-map",
        str(frame_map_path),
        "--out",
        str(output_path),
    ]

    try:
        completed = subprocess.run(
            cmd,
            check=True,
            timeout=timeout_seconds,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        error_result = {
            "label": "needs_review",
            "confidence": 0.0,
            "error": "gemini_eval.py failed",
            "stdout": exc.stdout,
            "stderr": exc.stderr,
        }
        output_path.write_text(json.dumps(error_result, indent=2), encoding="utf-8")
        return error_result

    except subprocess.TimeoutExpired as exc:
        error_result = {
            "label": "needs_review",
            "confidence": 0.0,
            "error": "gemini_eval.py timed out",
            "stdout": exc.stdout,
            "stderr": exc.stderr,
        }
        output_path.write_text(json.dumps(error_result, indent=2), encoding="utf-8")
        return error_result

    if not output_path.exists():
        error_result = {
            "label": "needs_review",
            "confidence": 0.0,
            "error": "gemini_eval.py did not create output file",
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        output_path.write_text(json.dumps(error_result, indent=2), encoding="utf-8")
        return error_result

    try:
        return json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "label": "needs_review",
            "confidence": 0.0,
            "error": "gemini_result.json was not valid JSON",
        }


def sanitize_label(label: str) -> str:
    label = str(label).strip().lower()
    if label in {"safe", "unsafe", "needs_review"}:
        return label
    return "needs_review"


def rename_video_with_label(video_path: Path, label: str) -> Path:
    """
    Rename:
        sequence_000000_merged.gif
    to:
        sequence_000000_safe_merged.gif
    """
    label = sanitize_label(label)

    if video_path.stem.endswith(f"_{label}_merged"):
        return video_path

    name = video_path.name

    if name.endswith("_merged.gif"):
        new_name = name.replace("_merged.gif", f"_{label}_merged.gif")
    else:
        new_name = f"{video_path.stem}_{label}{video_path.suffix}"

    labeled_path = video_path.with_name(new_name)

    if labeled_path.exists():
        labeled_path.unlink()

    video_path.rename(labeled_path)
    return labeled_path


def update_metadata_with_gemini_result(
    metadata_path: Path,
    *,
    labeled_video_path: Path,
    gemini_result_path: Path,
    gemini_result: dict,
) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    label = sanitize_label(gemini_result.get("label", "needs_review"))

    metadata["label"] = label
    metadata["merged_gif"] = labeled_video_path.name
    metadata["gemini_result"] = gemini_result_path.name

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

def delete_intermediate_subtask_gifs(seq_dir: Path) -> None:
    """
    Delete cleaned per-subtask GIFs inside seq_XXXXXX after the merged GIF is made.

    Leaves:
        sequence_XXXXXX_merged.gif
        meta.json
        frame_map.json
    """
    for gif in seq_dir.glob("[0-9][0-9]-*.gif"):
        gif.unlink()


def merge_sequence(
    eval_log_dir: str | Path,
    sequence_i: int,
    *,
    move_subtasks_into_folder: bool = True,
    delete_subtask_gifs_after_merge: bool = True,
    add_pause_between_gifs: bool = False,
    pause_ms: int = 250,
    fallback_duration_ms: int = 33,
    evaluate_with_gemini: bool = True,
    gemini_eval_script: str | Path = "gemini_eval.py",
) -> Optional[Path]:
    """
    Merge all subtask GIFs for one sequence id.

    Recommended call from eval_utils.py at the end of evaluate_sequence(...):

        merge_sequence(
            eval_log_dir,
            sequence_i,
            move_subtasks_into_folder=True,
            delete_subtask_gifs_after_merge=True,
        )

    This leaves only:
        seq_000000/
            sequence_000000_merged.gif
            meta.json
            frame_map.json
    """
    eval_log_dir = Path(eval_log_dir)
    subtasks = find_subtask_gifs(eval_log_dir, sequence_i=sequence_i)

    if not subtasks:
        return None

    seq_dir = make_sequence_dir(eval_log_dir, sequence_i)

    local_subtask_paths = copy_or_move_subtask_gifs(
        subtasks,
        seq_dir,
        move=move_subtasks_into_folder,
    )

    merged_path = seq_dir / f"sequence_{sequence_i:06d}_merged.gif"

    merged_path, frame_map = merge_gifs_preserve_timing(
        local_subtask_paths,
        merged_path,
        add_pause_between_gifs=add_pause_between_gifs,
        pause_ms=pause_ms,
        fallback_duration_ms=fallback_duration_ms,
    )

    metadata_path = write_metadata(seq_dir, sequence_i, subtasks, merged_path, frame_map)
    frame_map_path = write_frame_map(seq_dir, frame_map)
    final_video_path = merged_path

    if evaluate_with_gemini:
        gemini_result_path = seq_dir / "gemini_result.json"

        gemini_result = run_gemini_eval(
            video_path=merged_path,
            metadata_path=metadata_path,
            frame_map_path=frame_map_path,
            output_path=gemini_result_path,
            gemini_eval_script=gemini_eval_script,
        )

        label = sanitize_label((gemini_result or {}).get("label", "needs_review"))
        final_video_path = rename_video_with_label(merged_path, label)

        update_metadata_with_gemini_result(
            metadata_path,
            labeled_video_path=final_video_path,
            gemini_result_path=gemini_result_path,
            gemini_result=gemini_result or {"label": "needs_review"},
        )

    if delete_subtask_gifs_after_merge:
        delete_intermediate_subtask_gifs(seq_dir)

    return final_video_path


def merge_all_sequences(
    eval_log_dir: str | Path,
    *,
    move_subtasks_into_folder: bool = True,
    delete_subtask_gifs_after_merge: bool = True,
    add_pause_between_gifs: bool = False,
    pause_ms: int = 250,
    fallback_duration_ms: int = 33,
    evaluate_with_gemini: bool = True,
    gemini_eval_script: str | Path = "gemini_eval.py",
) -> List[Path]:
    """
    Post-process an eval directory after a run.

    Groups all root-level subtask GIFs by sequence id and merges each group.
    """
    gifs = find_subtask_gifs(eval_log_dir)
    groups = group_by_sequence(gifs)

    outputs: List[Path] = []

    for sequence_i in sorted(groups):
        merged = merge_sequence(
            eval_log_dir,
            sequence_i,
            move_subtasks_into_folder=move_subtasks_into_folder,
            delete_subtask_gifs_after_merge=delete_subtask_gifs_after_merge,
            add_pause_between_gifs=add_pause_between_gifs,
            pause_ms=pause_ms,
            fallback_duration_ms=fallback_duration_ms,
            evaluate_with_gemini=evaluate_with_gemini,
            gemini_eval_script=gemini_eval_script,
        )

        if merged is not None:
            outputs.append(merged)

    return outputs


def watch_and_merge_when_next_sequence_starts(
    eval_log_dir: str | Path,
    *,
    poll_seconds: float = 2.0,
    move_subtasks_into_folder: bool = True,
    delete_subtask_gifs_after_merge: bool = True,
    add_pause_between_gifs: bool = False,
    pause_ms: int = 250,
    fallback_duration_ms: int = 33,
) -> None:
    """
    Optional watcher mode.

    This avoids modifying eval_utils.py, but it is less reliable than calling
    merge_sequence(...) directly at the end of evaluate_sequence(...).
    """
    eval_log_dir = Path(eval_log_dir)
    already_merged = set()

    while True:
        gifs = find_subtask_gifs(eval_log_dir)
        groups = group_by_sequence(gifs)

        if len(groups) >= 2:
            sequence_ids = sorted(groups)

            # Every sequence except the newest is probably complete.
            for sequence_i in sequence_ids[:-1]:
                if sequence_i in already_merged:
                    continue

                merged = merge_sequence(
                    eval_log_dir,
                    sequence_i,
                    move_subtasks_into_folder=move_subtasks_into_folder,
                    delete_subtask_gifs_after_merge=delete_subtask_gifs_after_merge,
                    add_pause_between_gifs=add_pause_between_gifs,
                    pause_ms=pause_ms,
                    fallback_duration_ms=fallback_duration_ms,
                )

                if merged is not None:
                    print(f"Merged sequence {sequence_i}: {merged}")
                    already_merged.add(sequence_i)

        time.sleep(poll_seconds)


def inspect_gif_timing(path: str | Path) -> Dict:
    """
    Debug helper.

    Prints/returns basic timing info so you can compare:
        original subtask GIF vs merged GIF
    """
    path = Path(path)
    frames, durations = read_gif_frames_and_durations(path)

    info = {
        "path": str(path),
        "num_frames": len(frames),
        "total_duration_ms": sum(durations),
        "total_duration_seconds": sum(durations) / 1000.0,
        "unique_durations_ms": sorted(set(durations)),
        "first_20_durations_ms": durations[:20],
    }

    print(json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("eval_log_dir", help="Folder containing RoboFlamingo subtask GIFs")
    parser.add_argument("--sequence_i", type=int, default=None, help="Only merge one sequence id")
    parser.add_argument("--copy", action="store_true", help="Copy instead of moving root-level subtask GIFs")
    parser.add_argument("--keep-subtasks", action="store_true", help="Keep per-subtask GIFs after merge")
    parser.add_argument("--pause", action="store_true", help="Add pause between subtask GIFs")
    parser.add_argument("--pause_ms", type=int, default=250)
    parser.add_argument("--fallback_duration_ms", type=int, default=33)
    parser.add_argument("--watch", action="store_true", help="Watch folder and merge when the next sequence starts")
    parser.add_argument("--inspect", type=str, default=None, help="Inspect timing metadata of a GIF")
    parser.add_argument("--no-gemini", action="store_true", help="Merge only; do not call gemini_eval.py")
    parser.add_argument("--gemini-script", type=str, default="gemini_eval.py", help="Path to gemini_eval.py")

    args = parser.parse_args()

    if args.inspect:
        inspect_gif_timing(args.inspect)
    elif args.watch:
        watch_and_merge_when_next_sequence_starts(
            args.eval_log_dir,
            move_subtasks_into_folder=not args.copy,
            delete_subtask_gifs_after_merge=not args.keep_subtasks,
            add_pause_between_gifs=args.pause,
            pause_ms=args.pause_ms,
            fallback_duration_ms=args.fallback_duration_ms,
        )
    elif args.sequence_i is not None:
        out = merge_sequence(
            args.eval_log_dir,
            args.sequence_i,
            move_subtasks_into_folder=not args.copy,
            delete_subtask_gifs_after_merge=not args.keep_subtasks,
            add_pause_between_gifs=args.pause,
            pause_ms=args.pause_ms,
            fallback_duration_ms=args.fallback_duration_ms,
            evaluate_with_gemini=not args.no_gemini,
            gemini_eval_script=args.gemini_script,
        )
        print(out)
    else:
        outs = merge_all_sequences(
            args.eval_log_dir,
            move_subtasks_into_folder=not args.copy,
            delete_subtask_gifs_after_merge=not args.keep_subtasks,
            add_pause_between_gifs=args.pause,
            pause_ms=args.pause_ms,
            fallback_duration_ms=args.fallback_duration_ms,
            evaluate_with_gemini=not args.no_gemini,
            gemini_eval_script=args.gemini_script,
        )
        for out in outs:
            print(out)