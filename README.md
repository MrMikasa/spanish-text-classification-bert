# spanish-text-classification-bert

A small demo for Spanish text classification with a BERT-based pretrained
model. The first version uses
[`finiteautomata/beto-sentiment-analysis`](https://huggingface.co/finiteautomata/beto-sentiment-analysis),
a Hugging Face text-classification model based on BETO, the Spanish BERT
variant.

## What It Does

- Classifies Spanish text as positive, neutral, or negative.
- Provides a Gradio web UI with ready-to-run examples.
- Keeps the classifier logic separate from the UI so it can be tested or reused
  in scripts.

## Quick Start

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py
```

Open the local Gradio URL printed in the terminal, enter Spanish text, and view
the predicted sentiment scores.

## Example Inputs

- `Me encanta este producto, funciona muy bien.`
- `El servicio fue lento y la experiencia bastante mala.`
- `La pelicula tiene partes interesantes, aunque el final es normal.`

## Run Tests

```powershell
python -m unittest discover -s tests
```

The unit tests avoid downloading the model by injecting a fake classifier into
the core classification function.

## Project Structure

```text
.
|-- app.py
|-- spanish_text_classifier/
|   |-- __init__.py
|   `-- classifier.py
|-- tests/
|   `-- test_classifier.py
|-- requirements.txt
`-- requirements-dev.txt
```

## Model

Default model: `finiteautomata/beto-sentiment-analysis`

The model is downloaded automatically by `transformers` the first time the demo
runs. A working internet connection is required for the first launch unless the
model is already cached locally.
