import sys
import os
import tempfile

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import streamlit as st

from src.inference.predict_video import predict_video
from src.inference.subtitle_writer import write_srt

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AI Lip Reader",
    page_icon="🎥",
    layout="centered"
)

st.title("🎥 AI Lip Reader")

st.write(
    "Upload a silent video and automatically generate subtitles."
)

# =====================================
# FILE UPLOAD
# =====================================

uploaded_file = st.file_uploader(
    "Upload Silent Video",
    type=["mp4", "avi", "mov", "mpg"]
)

if uploaded_file is not None:

    st.subheader("📹 Video Preview")

    # Read the uploaded file ONCE
    video_bytes = uploaded_file.getvalue()

    # Display video preview
    st.video(video_bytes)

    # Save temporary file for prediction
    temp_video = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    temp_video.write(video_bytes)
    temp_video.close()

    if st.button("Generate Subtitles"):

        with st.spinner("Processing video..."):

            try:

                # =====================================
                # PREDICT
                # =====================================

                prediction = predict_video(
                    temp_video.name
                )

                st.success("Prediction completed!")

                st.subheader("📝 Predicted Text")

                st.write(prediction)

                # =====================================
                # CREATE SUBTITLE
                # =====================================

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

                st.success("Subtitle file generated successfully!")

                # =====================================
                # DOWNLOAD BUTTON
                # =====================================

                with open(srt_path, "rb") as f:

                    st.download_button(
                        label="📥 Download Subtitle (.srt)",
                        data=f,
                        file_name="prediction.srt",
                        mime="text/plain"
                    )

            except Exception as e:

                st.error(
                    f"Error: {str(e)}"
                )

# =====================================
# FOOTER
# =====================================

st.markdown("---")
st.caption("AI Lip Reader • Built with PyTorch + MediaPipe + Streamlit")