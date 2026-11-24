"""Select sharp, temporally distributed video frames for COLMAP."""
import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

VIDEO_SUFFIXES = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.webm'}


def choose_video(path=None):
    if path is not None:
        video = Path(path).resolve()
        if not video.is_file():
            raise ValueError(f'Video not found: {video}')
        return video
    video_dir = Path('inputs/videos') if Path('inputs/videos').is_dir() else Path('videos')
    videos = sorted(p for p in video_dir.glob('*')
                    if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
    if len(videos) != 1:
        raise ValueError(f'Place exactly one video in {video_dir}/, or specify --video PATH.')
    return videos[0].resolve()


def thumbnail(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale = 320 / max(gray.shape)
    return cv2.resize(gray, (max(1, round(gray.shape[1] * scale)),
                             max(1, round(gray.shape[0] * scale))))


def extract(video, output, max_frames=60, sample_fps=2.0, min_sharpness=25.0,
            min_change=2.0, max_image_size=1600):
    if max_frames < 3 or sample_fps <= 0 or max_image_size < 256:
        raise ValueError('Require max-frames >= 3, sample-fps > 0, max-image-size >= 256.')
    if not all(math.isfinite(v) for v in (sample_fps, min_sharpness, min_change)):
        raise ValueError('Filtering settings must be finite numbers.')
    if min_sharpness < 0 or not 0 <= min_change <= 255:
        raise ValueError('Require min-sharpness >= 0 and min-change between 0 and 255.')
    video, output = Path(video).resolve(), Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f'Output must be empty or new: {output}. Choose --output with a new name.')
    capture = cv2.VideoCapture(str(video))
    selected, previous = [], None
    sampled = blurry = duplicates = decoded = 0
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot open video: {video}. Try an H.264 MP4 export.')
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(frame_count) or frame_count < 3:
            raise ValueError('Video has invalid FPS/frame-count metadata. Export it as an MP4 first.')
        frame_count = int(frame_count)
        stride = max(1, round(fps / sample_fps))
        output.mkdir(parents=True, exist_ok=True)

        def save(candidate):
            nonlocal previous, duplicates
            if candidate is None:
                return
            score, index, frame, thumb = candidate
            if previous is not None:
                change = float(np.abs(thumb.astype(np.float32) - previous.astype(np.float32)).mean())
                if change < min_change:
                    duplicates += 1
                    return
            scale = min(1.0, max_image_size / max(frame.shape[:2]))
            if scale < 1:
                frame = cv2.resize(frame, (round(frame.shape[1]*scale), round(frame.shape[0]*scale)),
                                   interpolation=cv2.INTER_AREA)
            name = f'frame_{len(selected)+1:05d}.jpg'
            if not cv2.imwrite(str(output/name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise ValueError(f'Could not write {output/name}')
            selected.append({'file': name, 'source_frame': index,
                             'time_seconds': round(index/fps, 3), 'sharpness': round(score, 2)})
            previous = thumb
            print(f'Saved {name} at {index/fps:.1f}s (sharpness {score:.1f})', flush=True)

        best, current_bin = None, 0
        # Decode sequentially: only one winning full-resolution frame stays in memory.
        for index in range(frame_count):
            if not capture.grab():
                break
            decoded += 1
            if index % stride:
                continue
            ok, frame = capture.retrieve()
            if not ok:
                continue
            bucket = min(max_frames-1, index * max_frames // frame_count)
            if bucket != current_bin:
                save(best)
                best, current_bin = None, bucket
            sampled += 1
            thumb = thumbnail(frame)
            score = float(cv2.Laplacian(thumb, cv2.CV_64F).var())
            if score < min_sharpness:
                blurry += 1
                continue
            if best is None or score > best[0]:
                best = (score, index, frame, thumb)
        save(best)
    finally:
        capture.release()
    report = {'video': str(video), 'output': str(output), 'fps': fps,
              'source_frames': frame_count, 'decoded_frames': decoded,
              'sampled_candidates': sampled, 'blurry_candidates': blurry,
              'duplicate_bins': duplicates, 'selected_count': len(selected),
              'settings': {'max_frames': max_frames, 'sample_fps': sample_fps,
                           'min_sharpness': min_sharpness, 'min_change': min_change,
                           'max_image_size': max_image_size}, 'frames': selected}
    report_path = output.parent / (output.name + '.selection.json')
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if decoded < frame_count:
        print(f'Warning: decoded {decoded}/{frame_count} frames; video may be truncated.', file=sys.stderr)
    if len(selected) < 3:
        raise ValueError(f'Only {len(selected)} usable frames. Inspect the video or lower filter thresholds. '
                         f'Report: {report_path}')
    print(f'\nSelected {len(selected)} frames: {output}\nSelection report: {report_path}')
    print('Review coverage before reconstruction; sharpness and duplicate filtering do not guarantee overlap.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, help='Defaults to the only video in inputs/videos/ (or videos/).')
    parser.add_argument('--output', type=Path, default=Path('inputs/images/video_frames'))
    parser.add_argument('--max-frames', type=int, default=60, help='Upper limit, distributed over the video.')
    parser.add_argument('--sample-fps', type=float, default=2.0, help='Candidate samples per second.')
    parser.add_argument('--min-sharpness', type=float, default=25.0, help='Laplacian variance on a 320px thumbnail.')
    parser.add_argument('--min-change', type=float, default=2.0, help='Minimum thumbnail difference (0 disables).')
    parser.add_argument('--max-image-size', type=int, default=1600, help='Maximum saved image dimension.')
    args = parser.parse_args()
    try:
        extract(choose_video(args.video), args.output, args.max_frames, args.sample_fps,
                args.min_sharpness, args.min_change, args.max_image_size)
    except (ValueError, OSError, cv2.error) as exc:
        parser.exit(1, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
