from pathlib import Path
fp = "video13_f0000.jpg"
staged_path = "/tmp/outputs/proj/frames_run/video13_f0000.jpg"
frame_to_media = {staged_path: "media_123"}
staged_to_original = {staged_path: "/tmp/uploads/proj/frames/video13/video13_f0000.jpg"}

mid = frame_to_media.get(fp)
if mid is None:
    for k, v in frame_to_media.items():
        if k.endswith(fp) or Path(k).name == Path(fp).name:
            mid = v
            fp = k
            break

if fp in staged_to_original:
    fp = staged_to_original[fp]
    print(f"Success! fp is now {fp}")
else:
    print(f"Failed! fp is {fp}")
