from datasets import load_dataset
import time

cnt = 0
while True:
    try:
        dataset = load_dataset("tatsu-lab/alpaca_eval", force_download=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        cnt += 1
        # if cnt > 10:
        #     break
        print(f'Retrying in 5 seconds... (Attempt {cnt})')
        time.sleep(3)
    else:
        break