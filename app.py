import streamlit as st
from PIL import Image

from imagecaption import generate_caption
from pipeline import ToxicClassificationPipeline
from database import save_to_csv


# -------------------------
# load pipeline once
# -------------------------
@st.cache_resource
def load_pipeline():
    return ToxicClassificationPipeline(
        model_path="toxic_model.keras",
        tokenizer_path="tokenizer.pkl",
        label_encoder_path="label_encoder.pkl",
        max_length=35
    )


pipeline = load_pipeline()


# -------------------------
# app title
# -------------------------
st.title("Toxic Content Detection System")


# -------------------------
# sidebar navigation
# -------------------------
st.write("App is running...")
page = st.sidebar.selectbox(
    "Choose Page",
    ["Text Classification", "Image Captioning"]
)


# ==================================================
# TEXT CLASSIFICATION PAGE
# ==================================================
if page == "Text Classification":

    st.header("Text Classification")

    text_input = st.text_area(
        "Enter Text"
    )

    if st.button("Predict Text"):

        if text_input.strip():

            result = pipeline.predict(text_input)

            st.subheader("Prediction:")
            st.write(result["prediction"])

            st.subheader("Confidence:")
            st.write(f"{result['confidence']:.2%}")

            save_to_csv(
                input_type="text",
                user_query=text_input,
                caption="",
                prediction=result["prediction"]
            )

        else:
            st.warning("Please enter text")


# ==================================================
# IMAGE CAPTIONING PAGE
# ==================================================
elif page == "Image Captioning":

    st.header("Image Captioning + Classification")

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "png", "jpeg"]
    )

    # query is REQUIRED
    user_query = st.text_input(
        "Enter your query about this image"
    )

    # only show button if image uploaded
    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file)

            st.image(
                image,
                caption="Uploaded Image"
            )

            # generate caption
            caption = generate_caption(image)

            st.subheader("Generated Caption:")
            st.write(caption)

            # analyze button
            if st.button("Analyze Image"):

                if user_query.strip():

                    # maintain train-inference consistency
                    combined_text = user_query + " " + caption

                else:
                    # fallback query
                    combined_text = "Analyze this image " + caption

                st.subheader("Combined Input:")
                st.write(combined_text)

                result = pipeline.predict(combined_text)

                st.subheader("Prediction:")
                st.write(result["prediction"])

                st.subheader("Confidence:")
                st.write(f"{result['confidence']:.2%}")

                save_to_csv(
                    input_type="image",
                    user_query=user_query,
                    caption=caption,
                    prediction=result["prediction"]
                )

        except Exception as e:
            st.error(f"Error processing image: {e}")