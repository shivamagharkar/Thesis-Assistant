import whisper
import openai
import os
from pathlib import Path

openai.api_key = os.getenv("OPENAI_API_KEY")

def transcribe_and_summarize(video_path: str) -> str:
    model = whisper.load_model("base")
    result = model.transcribe(video_path)
    transcript = result["text"]

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Summarize this for thesis writing expectations:\n\n{transcript}"}]
    )
    return response.choices[0].message["content"]

if __name__ == "__main__":
    input_dir = "data/audio"
    output_dir = "data/video_summaries"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for video_file in Path(input_dir).glob("*.mp4"):
        print(f"Processing: {video_file.name}")
        summary = transcribe_and_summarize(str(video_file))
        out_file = Path(output_dir) / (video_file.stem + "_summary.txt")
        with open(out_file, "w") as f:
            f.write(summary)
        print(f"Saved summary: {out_file}")
