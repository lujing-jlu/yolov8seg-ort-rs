#![allow(clippy::manual_retain)]

use std::{path::Path};
use image::{imageops::FilterType, GenericImageView, ImageBuffer};
use ort::{inputs, CUDAExecutionProvider, Session, SessionOutputs};
use raqote::{DrawOptions, DrawTarget, LineCap, LineJoin, PathBuilder, SolidSource, Source, StrokeStyle};
use show_image::{event, AsImageView, WindowOptions};
use ndarray::{Array2, s, Array, Axis};
use image::imageops::resize;
use imageproc::contours::find_contours;
use imageproc::point::Point;



fn get_mask(row: &[f32], box_: (f32, f32, f32, f32), img_width: u32, img_height: u32) -> Array2<u8> {
    let (x1, y1, x2, y2) = box_;
    let mask = Array2::from_shape_vec((160, 160), row.to_vec()).unwrap();

    let mask_x1 = ((x1 / img_width as f32) * 160.0).round() as usize;
    let mask_y1 = ((y1 / img_height as f32) * 160.0).round() as usize;
    let mask_x2 = ((x2 / img_width as f32) * 160.0).round() as usize;
    let mask_y2 = ((y2 / img_height as f32) * 160.0).round() as usize;


    let mask_cropped = mask.slice(s![mask_y1..mask_y2, mask_x1..mask_x2]);
    let mask_sigmoid = mask_cropped.mapv(sigmoid);

    let mask_resized_width = (x2 - x1).round() as i32 as u32;
    let mask_resized_height = (y2 - y1).round() as i32 as u32;
    
    let mask_resized = resize(
        &ImageBuffer::from_fn(mask_sigmoid.shape()[1] as u32, mask_sigmoid.shape()[0] as u32, |x, y| {
            let value = mask_sigmoid[[y as usize, x as usize]];
            image::Luma([(value * 255.0) as u8])
        }),
        mask_resized_width,
        mask_resized_height,
        image::imageops::FilterType::Triangle,
    );

    Array2::from_shape_fn((mask_resized.height() as usize, mask_resized.width() as usize), |(y, x)| {
        if mask_resized.get_pixel(x as u32, y as u32)[0] > 128 {
            255
        } else {
            0
        }
    })
}


fn get_polygon(mask: &Array2<u8>, x1: f32, y1: f32) -> Vec<(f32, f32)> {
    let mask_image = image::ImageBuffer::from_fn(
        mask.shape()[1] as u32,
        mask.shape()[0] as u32,
        |x, y| {
            if mask[[y as usize, x as usize]] > 0 {
                image::Luma([255u8])
            } else {
                image::Luma([0u8])
            }
        },
    );

    let contours = find_contours::<u8>(&mask_image);

    if contours.is_empty() {
        return Vec::new();
    }

    let largest_contour = contours
        .into_iter()
        .max_by_key(|contour| contour.points.len())
        .unwrap();


    let polygon: Vec<(f32, f32)> = largest_contour
        .points
        .iter()
        .map(|&Point { x, y }| (x as f32 + x1, y as f32 + y1))
        .collect();

    polygon
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}


#[derive(Debug, Clone)]
struct Object {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    label: String,
    prob: f32,
    polygon: Vec<(f32, f32)>,
    mask: Array2<u8>,
}

impl Object {
    fn bbox(&self) -> (f32, f32, f32, f32) {
        (self.x1, self.y1, self.x2, self.y2)
    }
}

fn intersection(box1: (f32, f32, f32, f32), box2: (f32, f32, f32, f32)) -> f32 {
	(box1.2.min(box2.2) - box1.0.max(box2.0)) * (box1.3.min(box2.3) - box1.1.max(box2.1))
}

fn union(box1: (f32, f32, f32, f32), box2: (f32, f32, f32, f32)) -> f32 {
	((box1.2 - box1.0) * (box1.3 - box1.1)) + ((box2.2 - box2.0) * (box2.3 - box2.1)) - intersection(box1, box2)
}


#[rustfmt::skip]
const YOLOV8_CLASS_LABELS: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
	"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
	"bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
	"wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
	"carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
	"tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
	"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
];

const YOLOV8M_FILE_PATH: &str = "/home/ubuntu/projects/ort/yolov8m-seg.onnx";

#[show_image::main]
fn main() -> ort::Result<()> {
    tracing_subscriber::fmt::init();

    ort::init()
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .commit()?;

    let original_img = image::open(Path::new(env!("CARGO_MANIFEST_DIR")).join("data").join("baseball.jpg")).unwrap();
    let (img_width, img_height) = (original_img.width(), original_img.height());
    let img = original_img.resize_exact(640, 640, FilterType::CatmullRom);
    let mut input = Array::zeros((1, 3, 640, 640));
    for pixel in img.pixels() {
        let x = pixel.0 as _;
        let y = pixel.1 as _;
        let [r, g, b, _] = pixel.2.0;
        input[[0, 0, y, x]] = (r as f32) / 255.;
        input[[0, 1, y, x]] = (g as f32) / 255.;
        input[[0, 2, y, x]] = (b as f32) / 255.;
    }

    let model = Session::builder()?.commit_from_file(YOLOV8M_FILE_PATH)?;

    let outputs: SessionOutputs = model.run(inputs!["images" => input.view()]?)?;
    let output0 = outputs["output0"].try_extract_tensor::<f32>()?.t().into_owned();
    let output1 = outputs["output1"].try_extract_tensor::<f32>()?.into_owned();
    let output0 = output0.slice(s![.., .., 0]);
	let output1 = output1.slice(s![0, .., .., ..]);
	// 分离boxes和masks
	let boxes = output0.slice(s![.., 0..84]);
	let masks = output0.slice(s![.., 84..]);

	// 确保masks是一个二维数组
	let masks = masks.into_shape((masks.shape()[0], masks.shape()[1])).unwrap();

	// 重塑output1并执行矩阵乘法
	let output1_reshaped = output1.into_shape((output1.shape()[0], output1.shape()[1] * output1.shape()[2])).unwrap();
	let masks = masks.dot(&output1_reshaped);

	// 沿着第二个轴拼接boxes和masks
	let detections = ndarray::concatenate![Axis(1), boxes, masks];

    let mut objects = Vec::new();
    for row in detections.axis_iter(Axis(0)) {
        let row: Vec<_> = row.iter().copied().collect();
        let prob = row[4..84].iter().cloned().fold(f32::MIN, f32::max);
        if prob < 0.5 {
            continue;
        }
    	let class_id = row[4..84].iter().cloned().position(|x| x == prob).unwrap();
        let label = YOLOV8_CLASS_LABELS[class_id].to_string();
        let xc = row[0];
        let yc = row[1];
        let w = row[2];
        let h = row[3];
        let x1 = (xc - w / 2.0) / 640.0 * img_width as f32;
        let y1 = (yc - h / 2.0) / 640.0 * img_height as f32;
        let x2 = (xc + w / 2.0) / 640.0 * img_width as f32;
        let y2 = (yc + h / 2.0) / 640.0 * img_height as f32;
        let mask = get_mask(&row[84..25684], (x1, y1, x2, y2), img_width, img_height);
    	let polygon = get_polygon(&mask, x1, y1);
        objects.push(Object { x1, y1, x2, y2, label, prob, polygon, mask });
    }

    objects.sort_by(|a, b| b.prob.partial_cmp(&a.prob).unwrap());
    let mut results = Vec::new();
	while !objects.is_empty() {
		results.push(objects[0].clone());
		let bbox = results.last().unwrap().bbox();
		objects.retain(|object| intersection(object.bbox(), bbox) / union(object.bbox(), bbox) < 0.7);
	}


    let mut dt = DrawTarget::new(img_width as _, img_height as _);
	for object in &results {
		let bbox = (object.x1, object.y1, object.x2, object.y2);
		let label = &object.label;
		let mask = &object.mask;

		let mut pb = PathBuilder::new();
		pb.rect(bbox.0, bbox.1, bbox.2 - bbox.0, bbox.3 - bbox.1);
		let path = pb.finish();
		let color = match label.as_str() {
			"baseball bat" => SolidSource { r: 0x00, g: 0x10, b: 0x80, a: 0x80 },
			"baseball glove" => SolidSource { r: 0x20, g: 0x80, b: 0x40, a: 0x80 },
			_ => SolidSource { r: 0x80, g: 0x10, b: 0x40, a: 0x80 }
		};
		dt.stroke(
			&path,
			&Source::Solid(color),
			&StrokeStyle {
				join: LineJoin::Round,
				width: 4.,
				..StrokeStyle::default()
			},
			&DrawOptions::new()
		);

      	for (y, row) in mask.axis_iter(Axis(0)).enumerate() {
			for (x, &value) in row.iter().enumerate() {
				if value > 0 {
					dt.fill_rect(
						bbox.0 + x as f32,
						bbox.1 + y as f32,  // 修正这里
						1.0,
						1.0,
						&Source::Solid(SolidSource { r: 0xFF, g: 0x00, b: 0x00, a: 0x80 }),
						&DrawOptions::new()
					);
				}
			}
		}
		let points: Vec<_> = object.polygon.iter().map(|&(x, y)| (x, y)).collect();
		if points.len() >= 2 {
			let mut pb = PathBuilder::new();
			pb.move_to(points[0].0, points[0].1);
			for &(x, y) in &points[1..] {
				pb.line_to(x, y);
			}
			pb.close();
			let path = pb.finish();
			dt.stroke(
				&path,
				&Source::Solid(SolidSource {
					r: 0x00,
					g: 0xFF,
					b: 0x00,
					a: 0xFF,
				}),
				&StrokeStyle {
					cap: LineCap::Round,
					join: LineJoin::Round,
					width: 2.,
					..StrokeStyle::default()
				},
				&DrawOptions::new(),
			);
		}
    }

    let overlay: show_image::Image = dt.into();

    let window = show_image::context()
        .run_function_wait(move |context| -> Result<_, String> {
            let mut window = context
                .create_window(
                    "ort + YOLOv8",
                    WindowOptions {
                        size: Some([img_width, img_height]),
                        ..WindowOptions::default()
                    }
                )
                .map_err(|e| e.to_string())?;
            window.set_image("baseball", &original_img.as_image_view().map_err(|e| e.to_string())?);
            window.set_overlay("yolo", &overlay.as_image_view().map_err(|e| e.to_string())?, true);
            Ok(window.proxy())
        })
        .unwrap();

    for event in window.event_channel().unwrap() {
        if let event::WindowEvent::KeyboardInput(event) = event {
            if event.input.key_code == Some(event::VirtualKeyCode::Escape) && event.input.state.is_pressed() {
                break;
            }
        }
    }

    Ok(())
}