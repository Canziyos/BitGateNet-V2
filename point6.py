# """
# profile_latency.py
# Profiles forward-pass latency and memory usage of BitGateNetV2.
# Reports batch latency, per-sample latency, throughput, and memory usage.
# """

# import time
# import torch
# from torch.utils.data import DataLoader
# from model import BitGateNetV2
# from train import AudioFolder, collate, classes, dataset_dir, batch_size, n_worker
# import tqdm
# # For CPU memory.
# try:
#     import psutil
# except ImportError:
#     psutil = None

# # ------------------------------------------------------------------ #
# # Main latency + memory profiling.                                   #
# # ------------------------------------------------------------------ #
# def main():
#     # Dataset (validation for representative batch)
#     val_ds = AudioFolder(dataset_dir, "val")
#     val_dl = DataLoader(val_ds, batch_size, shuffle=False,
#     collate_fn=collate, num_workers=n_worker)

#     # Grab one batch
#     x, _ = next(iter(val_dl))

#     # Model
#     model = BitGateNetV2(num_classes=len(classes), q_en=True)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model.to(device)
#     model.eval()

#     # Move data to device
#     x = x.to(device)

#     # Warmup runs (stabilize GPU clocks)
#     for _ in range(3):
#         _ = model(x)

#     # ------- Memory tracking ---------------- #
#     if device == "cuda":
#         torch.cuda.reset_peak_memory_stats()

#     if device == "cpu" and psutil:
#         process = psutil.Process()

#     # ----------Timing multiple runs ------------ #
#     n_runs = 20
#     if device == "cuda":
#         torch.cuda.synchronize()
#     t0 = time.time()
#     for _ in range(n_runs):
#         _ = model(x)
#     if device == "cuda":
#         torch.cuda.synchronize()
#     t1 = time.time()

#     # ---------------- Latency metrics ----------------- #
#     total_time = (t1 - t0) * 1000  # ms
#     avg_batch_ms = total_time / n_runs
#     avg_sample_ms = avg_batch_ms / batch_size
#     throughput = 1000.0 / avg_sample_ms  # samples/sec

#     # ---------------- Memory metrics ------------------ #
#     if device == "cuda":
#         mem_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
#         mem_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
#         print(f"\nMemory (CUDA): Current={mem_alloc:.2f} MB, Peak={mem_peak:.2f} MB")
#     elif psutil:
#         mem_info = process.memory_info()
#         print(f"\nMemory (CPU): RSS={mem_info.rss/1024**2:.2f} MB")
#     else:
#         print("\nMemory info: psutil not installed, skipping CPU memory measurement.")

#     # ---------------- Print results ------------------- #
#     print(f"\nLatency profiling results ({device}):")
#     print(f"  Avg batch latency : {avg_batch_ms:.2f} ms")
#     print(f"  Avg per-sample    : {avg_sample_ms:.2f} ms")
#     print(f"  Throughput        : {throughput:.2f} samples/sec")

# # ------------------------------------------------------------------ #
# if __name__ == "__main__":
#     main()
