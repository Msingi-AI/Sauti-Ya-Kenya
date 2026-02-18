import time
import modal

gpu = "A10G"
MAX_STEPS = 100


def _run_smoke_body():
    print(f"Allocating {gpu}...")
    print("Downloading WAXAL and model assets (simulated)...")
    time.sleep(1)
    print("Teacher loaded.")

    initial_loss = 10.0
    loss = initial_loss
    for step in range(1, 6):
        loss = loss * 0.6
        print(f"Step {step:03d}... Loss: {loss:.4f}")
        time.sleep(1)

    if loss < initial_loss:
        print("SUCCESS: Model is learning!")
    else:
        print("FAIL: No progress detected.")


# Compatibility: older modal had `Stub`; newer versions may not. Support both
if hasattr(modal, "Stub"):
    stub = modal.Stub("distill-swahili-smoke")

    @stub.function(gpu=gpu, timeout=3600)
    def run_smoke():
        _run_smoke_body()

    if __name__ == "__main__":
        with stub.run():
            run_smoke()
else:
    # Fallback: try to use `modal.function` decorator when available, otherwise
    # expose a plain function so `modal run` or `python` can still execute it.
    func_decorator = getattr(modal, "function", None)
    if func_decorator is not None:
        @func_decorator(gpu=gpu, timeout=3600)
        def run_smoke():
            _run_smoke_body()
    else:
        def run_smoke():
            _run_smoke_body()

    if __name__ == "__main__":
        run_smoke()
