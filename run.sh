# set env (Windows cmd)
set WANDB_API_KEY=your_key_here
set PYTHONPATH=%CD%

# run from project root, override paths and scheduler
uv run src/method2/train.py --processed-dir C:\Users\Admin\PycharmProjects\NewsRecSys\processed_parquet --embedding-path C:\Users\Admin\PycharmProjects\NewsRecSys\embedding\body_emb.npy --lr-scheduler onecycle
