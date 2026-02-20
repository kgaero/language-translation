## Run This Repo On Kaggle

Use these cells in a Kaggle notebook:

```bash
!git clone https://github.com/<your-user>/<your-repo>.git
%cd <your-repo>
!pip install -q -r requirements.txt
!python train.py
```

### Kaggle settings
- Turn `GPU` on.
- Turn `Internet` on (required for `load_dataset(...)` to download `opus_books`).

### Notes
- Your project can stay as multiple `.py` files.
- Running `python train.py` from the repo root will resolve imports like `from model import ...`.
