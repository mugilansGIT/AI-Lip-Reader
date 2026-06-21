import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import streamlit as st
import tempfile

from src.inference.predict_video import predict_video
from src.inference.subtitle_writer import write_srt

# =====================================
# UI
# =====================================

st.set_page_config(
    page_title="AI Lip Reader",
    page_icon="🎥"
)

st.title("🎥 AI Lip Reader")
st.write(
    "Upload a silent video and generate subtitles automatically."
)

# =====================================
# FILE UPLOAD
# =====================================

uploaded_file = st.file_uploader(
    "Upload Silent Video",
    type=["mp4", "avi", "mov", "mpg"]
)

if uploaded_file is not None:

    st.subheader("Video Preview")

    st.video(uploaded_file)

    temp_video = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    temp_video.write(uploaded_file.read())
    temp_video.close()

    if st.button("Generate Subtitles"):

        with st.spinner("Processing video..."):

            try:

                # =========================
                # PREDICTION
                # =========================

                prediction = predict_video(
                    temp_video.name
                )

                st.success("Prediction completed!")

                st.subheader("Predicted Text")

                st.write(prediction)

                # =========================
                # CREATE SUBTITLE FORMAT
                # =========================

                subtitles = [
                    {
                        "start": 0,
                        "end": 3,
                        "text": prediction
                    }
                ]

                # =========================
                # SAVE SRT
                # =========================

                srt_path = "prediction.srt"

                write_srt(
                    subtitles,
                    srt_path
                )

                st.success(
                    "Subtitle file generated successfully!"
                )

                # =========================
                # DOWNLOAD BUTTON
                # =========================

                with open(srt_path, "rb") as f:

                    st.download_button(
                        label="📥 Download SRT",
                        data=f,
                        file_name="prediction.srt",
                        mime="text/plain"
                    )

            except Exception as e:

                st.error(
                    f"Error occurred: {str(e)}"
                )