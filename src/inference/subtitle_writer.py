import srt
from datetime import timedelta


def write_srt(subtitles, output_path):

    srt_subtitles = []

    for idx, sub in enumerate(subtitles, start=1):

        srt_subtitles.append(
            srt.Subtitle(
                index=idx,
                start=timedelta(seconds=sub["start"]),
                end=timedelta(seconds=sub["end"]),
                content=sub["text"]
            )
        )

    srt_content = srt.compose(srt_subtitles)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    print(f"SRT saved to {output_path}")