import pickle
import numpy as np
import re
import string

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


class ToxicClassificationPipeline:

    def __init__(
        self,
        model_path,
        tokenizer_path,
        label_encoder_path,
        max_length=35
    ):
        # load trained model
        self.model = load_model(model_path)

        # load tokenizer
        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)

        # load label encoder
        with open(label_encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)

        self.max_length = max_length

    # -------------------------
    # same preprocessing as training
    # -------------------------
    def preprocess(self, text):

        text = str(text).lower()

        # remove urls
        text = re.sub(
            r"http\S+|www\S+|https\S+",
            "",
            text
        )

        # remove html tags
        text = re.sub(
            r"<.*?>",
            "",
            text
        )

        # remove punctuation
        text = text.translate(
            str.maketrans(
                "",
                "",
                string.punctuation
            )
        )

        # remove numbers
        text = re.sub(
            r"\d+",
            "",
            text
        )

        # remove extra spaces
        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        return text

    # -------------------------
    # tokenize + pad
    # -------------------------
    def tokenize(self, text):

        sequence = self.tokenizer.texts_to_sequences(
            [text]
        )

        padded = pad_sequences(
            sequence,
            maxlen=self.max_length,
            padding="post",
            truncating="post"
        )

        return padded

    # -------------------------
    # predict
    # -------------------------
    def predict(self, text):

        # preprocess text
        clean_text = self.preprocess(text)

        # tokenize
        tokenized_text = self.tokenize(
            clean_text
        )

        # model prediction
        prediction = self.model.predict(
            tokenized_text,
            verbose=0
        )

        # predicted class index
        predicted_class = np.argmax(
            prediction,
            axis=1
        )[0]

        # predicted label
        predicted_label = (
            self.label_encoder
            .inverse_transform(
                [predicted_class]
            )[0]
        )

        # confidence score
        confidence = prediction[0][predicted_class]

        return {
            "prediction": predicted_label,
            "confidence": float(confidence)
        }