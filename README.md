# yolov8seg-ort-rs

This project provides a Rust implementation of YOLOv8 segmentation using the [ONNX Runtime](https://github.com/microsoft/onnxruntime) inference engine. It is based on the YOLOv8 example from the [pykeio/ort](https://github.com/pykeio/ort) project.

## Requirements

- Rust programming language
- Microsoft ONNX Runtime v1.18.0 or later
- yolov8m-seg pt model

## Install the required packages

Run the following commands to install the required packages:

```bash
pip install onnxruntime-gpu  # Install GPU version
pip install onnxruntime  # Install CPU version
pip install ultralytics
```

## Prepare your YOLOv8 segmentation model in ONNX format.
1. Download the model

    Visit the [Segment Models section](https://docs.ultralytics.com/tasks/segment/#models) in the Ultralytics YOLO documentation to download the desired model. For example, you can download the YOLOv8m-seg model.
2. Convert the model to ONNX format

    Run the following command to convert the `YOLOv8m-seg.pt` model to `onnx` format:

    ```bash
    yolo task=segment mode=export model=yolov8m-seg.pt format=onnx
    ```

    This command will convert the YOLOv8m-seg model from PyTorch format (`.pt`) to ONNX format (`.onnx`).

## Download and install the ONNX Runtime
1. Download and install the ONNX Runtime shared library from the [official releases page](https://github.com/microsoft/onnxruntime/releases/tag/v1.18.0). Make sure to choose the appropriate version for your operating system.

2. Set the `LD_LIBRARY_PATH` environment variable to the directory containing the ONNX Runtime shared library. For example:

    ```bash
    export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:$LD_LIBRARY_PATH
    ```

## Installation

Clone this repository:

```bash
git clone https://github.com/your-username/yolov8seg-ort-rs.git
cd yolov8seg-ort-rs
```


## Usage

1. Edit src/main.rs

Replace `path/to/your/model.onnx` with the path to your YOLOv8 in Line:129.
```rust
const YOLOV8M_FILE_PATH: &str = "/home/ubuntu/projects/ort/yolov8m-seg.onnx";
```

And `image path` in Line:139 with the path to the input image.
```rust
let original_img = image::open(Path::new(env!("CARGO_MANIFEST_DIR")).join("data").join("baseball.jpg")).unwrap();
```

2. Run the YOLOv8 segmentation inference:
```bash
./target/release/yolov8seg-ort-rs
```

Replace `path/to/your/model.onnx` with the path to your YOLOv8 segmentation model and `path/to/your/image.jpg` with the path to the input image.

3. The segmentation results will be saved as an image file named `output.jpg` in the current directory.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [pykeio/ort](https://github.com/pykeio/ort) - Rust wrapper for ONNX Runtime
- [Microsoft ONNX Runtime](https://github.com/microsoft/onnxruntime) - Open-source inference engine for ONNX models