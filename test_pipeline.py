from pipeline import ToxicClassificationPipeline

pipeline = ToxicClassificationPipeline(
    model_path="toxic_model.keras",
    tokenizer_path="tokenizer.pkl",
    label_encoder_path="label_encoder.pkl",
    max_length=35
)

result = pipeline.predict(
    "How can I protect my house from criminals?"
)

print(result)