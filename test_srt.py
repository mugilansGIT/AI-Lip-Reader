from src.inference.subtitle_writer import write_srt

subtitles = [
    {
        "start": 0,
        "end": 2,
        "text": "hello"
    },
    {
        "start": 2,
        "end": 4,
        "text": "how are you"
    },
    {
        "start": 4,
        "end": 6,
        "text": "welcome"
    }
]

write_srt(
    subtitles,
    "sample.srt"
)