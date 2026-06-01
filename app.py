import gradio as gr

from spanish_text_classifier import DEFAULT_MODEL_ID, classify_text


EXAMPLES = [
    "Me encanta este producto, funciona muy bien.",
    "El servicio fue lento y la experiencia bastante mala.",
    "La pelicula tiene partes interesantes, aunque el final es normal.",
]


def predict(text: str):
    try:
        rows = classify_text(text)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    return {row["label"]: row["score"] for row in rows}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        label="Texto en espanol",
        lines=5,
        placeholder="Escribe una frase o resena en espanol...",
    ),
    outputs=gr.Label(label="Clasificacion"),
    examples=EXAMPLES,
    title="Spanish Text Classification with BETO",
    description=(
        "Demo de clasificacion de sentimiento para textos en espanol usando "
        f"el modelo {DEFAULT_MODEL_ID}."
    ),
)


if __name__ == "__main__":
    demo.launch()
