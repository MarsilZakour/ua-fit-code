# Checkpoints

The released weights are on Hugging Face: **https://huggingface.co/Marsil123/ua-fit**
(`uafit_w40.safetensors`, 189 MB; network only, no MANO data). The code downloads them when given the model id:

```bash
python scripts/evaluate.py --checkpoint Marsil123/ua-fit
```

For offline use, place the file here (`checkpoints/uafit_w40/uafit_w40.safetensors`) and pass the directory:

```bash
python scripts/evaluate.py --checkpoint checkpoints/uafit_w40
```

`--checkpoint` accepts a weights file, a directory holding one, or a Hugging Face model id.
