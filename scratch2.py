import json

out_json = {
 "images": [
  {
   "file": "video12_43bfbe_f0000_t000.00.jpg",
   "detections": []
  }
 ]
}

frame_to_media = {
  "/Users/mobashirsifat/Desktop/dist/outputs/proj/frames_run/video12_43bfbe_f0000_t000.00.jpg": "media_123",
  "uploads/proj/frames/video12/video12_43bfbe_f0000_t000.00.jpg": "media_123"
}

staged_to_original = {
  "/Users/mobashirsifat/Desktop/dist/outputs/proj/frames_run/video12_43bfbe_f0000_t000.00.jpg": "uploads/proj/frames/video12/video12_43bfbe_f0000_t000.00.jpg"
}

fp = "video12_43bfbe_f0000_t000.00.jpg"
mid = frame_to_media.get(fp)
if mid is None:
    from pathlib import Path
    for k, v in frame_to_media.items():
        if k.endswith(fp) or Path(k).name == Path(fp).name:
            mid = v
            fp = k
            break
            
print(f"After mapping, fp = {fp}")

if fp in staged_to_original:
    fp = staged_to_original[fp]
    print(f"After restore, fp = {fp}")
else:
    print("NOT FOUND in staged_to_original!")
