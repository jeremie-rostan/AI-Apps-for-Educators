# IEP Local Assistant

Local-first app for:
- Discovering IEPs by grade level
- Selecting one or multiple IEPs
- Chatting with selected IEP context
- Adapting assessments with selected IEP context
- Downloading adapted assessments as Markdown

## Run

1. Install dependencies:

```bash
cd /Users/jrostan/HuggingFace/app
pip install -r requirements.txt
```

2. (Optional) Copy legacy IEP files into grade subfolders:

```bash
cd /Users/jrostan/HuggingFace/app
python migrate_ieps.py
```

3. Start Streamlit app:

```bash
cd /Users/jrostan/HuggingFace/app
./venv/bin/streamlit run streamlit_app.py
```

4. Start Gradio app (multi-turn chat):

```bash
cd /Users/jrostan/HuggingFace/app
IEP_GPU_LAYERS=8 ./venv/bin/python gradio_app.py
```

## Model setup (llama.cpp GGUF backend)

The app uses:
- `llama-cli`: `/tmp/llama.cpp-hf-quant/build/bin/llama-cli`
- Chat model: `/Users/jrostan/HuggingFace/models_gguf/chat/chat-q5_k_m.gguf`
- Adapt model: `/Users/jrostan/HuggingFace/models_gguf/adapt/adapt-q5_k_m.gguf`

This is much faster than the previous PyTorch/PEFT path on Apple Silicon.

## Notes

- `.doc` parsing uses macOS `textutil`. If conversion fails, export as `.docx` or `.pdf`.
- Google Doc import requires a shareable doc URL.
