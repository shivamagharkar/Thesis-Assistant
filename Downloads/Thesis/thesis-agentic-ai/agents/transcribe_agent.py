import whisper
from pathlib import Path

def transcribe_all_videos(input_dir: str, output_dir: str):
    model = whisper.load_model("base")
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for video_file in input_path.glob("*.mp4"):
        print(f"Transcribing: {video_file.name}")
        result = model.transcribe(str(video_file))
        out_file = output_path / (video_file.stem + ".txt")
        with open(out_file, "w") as f:
            f.write(result["text"])
        print(f"Saved transcript: {out_file}")

if __name__ == "__main__":
    transcribe_all_videos("data/audio", "data/transcripts")
