import sys
import os
import tempfile
import torch
import streamlit as st

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from src.inference.predict_video import predict_video
from src.inference.subtitle_writer import write_srt

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AI Lip Reader",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("🎥 AI Lip Reader")

st.sidebar.markdown("### Project Information")

device = "CUDA" if torch.cuda.is_available() else "CPU"

st.sidebar.info(
    f"""
**Version:** v1.0

**Model:** LipNet

**Framework:** PyTorch

**Dataset:** GRID Corpus

**Device:** {device}

**Word Accuracy:** 90.31%

**Status:** 🟢 Ready
"""
)

st.sidebar.markdown("---")

st.sidebar.markdown(
"""
### Features

✅ Video Upload

✅ Subtitle Generation

✅ Download SRT

✅ Live Webcam (Experimental)

✅ Deep Learning

✅ MediaPipe Lip Detection
"""
)

# =====================================
# HEADER
# =====================================

st.title("🎥 AI Lip Reader")

st.markdown(
"""
### Real-Time Visual Speech Recognition

Generate subtitles from silent videos using a deep learning based LipNet model.
"""
)

st.divider()

# =====================================
# UPLOAD SECTION
# =====================================

st.subheader("📤 Upload Video")

uploaded_file = st.file_uploader(
    "Choose a silent video",
    type=["mp4", "avi", "mov", "mpg"]
)

if uploaded_file is not None:

    video_bytes = uploaded_file.getvalue()

    col1, col2 = st.columns([2, 1])

    with col1:

        st.subheader("📹 Video Preview")

        st.video(video_bytes)

    with col2:

        st.subheader("ℹ Video Information")

        st.write(f"Filename: **{uploaded_file.name}**")

        st.write(
            f"Size: **{len(video_bytes)/1024/1024:.2f} MB**"
        )

        st.write(
            "Supported ✔"
        )

    temp_video = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    temp_video.write(video_bytes)
    temp_video.close()

    st.divider()

    if st.button(
        "🚀 Generate Subtitles",
        use_container_width=True
    ):

        progress = st.progress(0)

        with st.spinner(
            "Running LipNet inference..."
        ):

            progress.progress(20)

            try:

                prediction = predict_video(
                    temp_video.name
                )

                progress.progress(80)

                subtitles = [
                    {
                        "start": 0,
                        "end": 3,
                        "text": prediction
                    }
                ]

                srt_path = "prediction.srt"

                write_srt(
                    subtitles,
                    srt_path
                )

                progress.progress(100)

                st.success(
                    "Prediction completed successfully!"
                )

                st.divider()

                st.subheader("📝 Prediction")

                st.code(
                    prediction,
                    language=None
                )

                col1, col2 = st.columns(2)

                with col1:

                    with open(
                        srt_path,
                        "rb"
                    ) as f:

                        st.download_button(
                            "📥 Download Subtitle (.srt)",
                            data=f,
                            file_name="prediction.srt",
                            mime="text/plain",
                            use_container_width=True
                        )

                with col2:

                    st.download_button(
                        "📥 Download Transcript (.txt)",
                        data=prediction,
                        file_name="prediction.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

            except Exception as e:

                st.error(str(e))

# =====================================
# FOOTER
# =====================================

st.divider()

st.caption(
    "© 2026 Mugilan • AI Lip Reader • Built with PyTorch, MediaPipe and Streamlit"
)