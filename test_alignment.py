from src.utils.text_utils import parse_alignment

path = "data/raw_videos/s1_processed/align/bbaf2n.align"

text = parse_alignment(path)

print(text)