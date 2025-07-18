from ultralytics import YOLO
import os

model_variants = ["yolov8n", "yolov8m", "yolov8l", "yolov8x"]
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

for variant in model_variants:
    pt_path = os.path.join(models_dir, f"{variant}.pt")
    engine_path = os.path.join(models_dir, f"{variant}.engine")
    onnx_path = os.path.join(models_dir, f"{variant}.onnx") # Added ONNX path

    if not os.path.exists(pt_path):
        print(f"Downloading {pt_path}...")
        YOLO(f"{variant}.pt") # This downloads the .pt model if it doesn't exist
        print(f"Downloaded {pt_path}")

    # Try to export to .engine (TensorRT) first if on a system that supports it
    if os.path.exists(pt_path) and not os.path.exists(engine_path) and hasattr(YOLO(pt_path), 'export') and YOLO(pt_path).device.type == 'cuda':
        print(f"Exporting {variant}.pt to {variant}.engine (TensorRT)... This may take a while.")
        try:
            model = YOLO(pt_path)
            # imgsz=640 is a common input size for YOLOv8
            # half=True for FP16 quantization (recommended for speed with minimal accuracy loss)
            model.export(format='engine', imgsz=640, half=True)
            print(f"Successfully exported {engine_path}")
        except Exception as e:
            print(f"Failed to export {variant}.pt to .engine: {e}")
            print("TensorRT export requires a compatible NVIDIA GPU, CUDA, and cuDNN.")
            print("Falling back to ONNX or PyTorch for this model.")
    elif os.path.exists(engine_path):
        print(f"{engine_path} already exists. Skipping .engine export.")
    elif os.path.exists(pt_path) and not os.path.exists(engine_path) and hasattr(YOLO(pt_path), 'export'):
        print(f"Skipping .engine export for {variant}.pt as no CUDA device found or export failed.")

    # Then try to export to .onnx if .engine wasn't successful or desired
    if os.path.exists(pt_path) and not os.path.exists(onnx_path):
        print(f"Exporting {variant}.pt to {variant}.onnx...")
        try:
            model = YOLO(pt_path)
            model.export(format='onnx', imgsz=640)
            print(f"Successfully exported {onnx_path}")
        except Exception as e:
            print(f"Failed to export {variant}.pt to .onnx: {e}")
    elif os.path.exists(onnx_path):
        print(f"{onnx_path} already exists. Skipping .onnx export.")


print("\nAll desired models (or their optimized versions) should now be in the 'models/' directory.")
