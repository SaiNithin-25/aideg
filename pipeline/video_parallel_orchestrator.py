import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
except ImportError:
    pynvml = None

from pipeline.video_queue_orchestrator import (
    load_status,
    load_video_list,
    process_video,
    save_status,
)

VIDEO_WORKERS = 2
GPU_HANDLE = None


def initialize_gpu_monitoring():
    global GPU_HANDLE

    if pynvml is None:
        return

    try:
        pynvml.nvmlInit()
        GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        GPU_HANDLE = None


def shutdown_gpu_monitoring():
    global GPU_HANDLE

    if pynvml is None or GPU_HANDLE is None:
        return

    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass
    finally:
        GPU_HANDLE = None


def get_cpu_utilization():
    if psutil is None:
        return None

    try:
        return psutil.cpu_percent(interval=None)
    except Exception:
        return None


def get_gpu_utilization():
    if pynvml is None or GPU_HANDLE is None:
        return None

    try:
        utilization = pynvml.nvmlDeviceGetUtilizationRates(GPU_HANDLE)
        return float(utilization.gpu)
    except Exception:
        return None


def format_utilization(value):
    if value is None:
        return "N/A"
    return f"{value:.1f}%"


def main():
    urls = load_video_list()
    last = load_status()
    highest_saved = last
    completed_indexes = set()

    initialize_gpu_monitoring()
    get_cpu_utilization()

    print(f"Total Videos: {len(urls)}")
    print(f"Resuming from: {last + 1}")
    print(f"Video Workers: {VIDEO_WORKERS}")

    start_total = time.time()

    try:
        with ThreadPoolExecutor(max_workers=VIDEO_WORKERS) as executor:
            futures = {}

            for i, url in enumerate(urls):
                if i <= last:
                    continue

                future = executor.submit(process_video, url, i)
                futures[future] = i

            for future in as_completed(futures):
                idx = futures[future]
                cpu_utilization = format_utilization(get_cpu_utilization())
                gpu_utilization = format_utilization(get_gpu_utilization())

                try:
                    future.result()
                    completed_indexes.add(idx)

                    while highest_saved + 1 in completed_indexes:
                        highest_saved += 1

                    if highest_saved > last:
                        save_status(highest_saved)
                        last = highest_saved

                    print(
                        f"Video {idx} finished | "
                        f"CPU: {cpu_utilization} | GPU: {gpu_utilization}"
                    )
                except Exception as e:
                    print(
                        f"Video {idx} failed: {str(e)} | "
                        f"CPU: {cpu_utilization} | GPU: {gpu_utilization}"
                    )
    finally:
        shutdown_gpu_monitoring()

    end_total = time.time()

    print("\nALL VIDEOS COMPLETED")
    print("Total Pipeline Time:", round(end_total - start_total, 2), "sec")


if __name__ == "__main__":
    main()
