import openai
import os
from pathlib import Path

openai.api_key = os.getenv("OPENAI_API_KEY")

def summarize_text(text: str, custom_prompt: str = "Summarize this for thesis writing expectations:"):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"{custom_prompt}\n\n{text}"}]
    )
    return response.choices[0].message["content"]

def summarize_all_transcripts(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for transcript_file in input_path.glob("*.txt"):
        print(f"Summarizing: {transcript_file.name}")
        with open(transcript_file) as f:
            text = f.read()

        summary = summarize_text(text)
        summary_file = output_path / transcript_file.name
        with open(summary_file, "w") as f:
            f.write(summary)
        print(f"Saved summary: {summary_file}")

if __name__ == "__main__":
    summarize_all_transcripts("data/transcripts", "data/summaries")
