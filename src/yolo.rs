use anyhow::{Error, Result};
use image::DynamicImage;
use image::GenericImageView;
use std::cmp::Ordering;
use std::cmp::PartialOrd;
use std::collections::HashMap;
use tract_core::plan::SimplePlan;
use tract_ndarray::{s, Axis};
use tract_onnx::prelude::*;

use crate::bbox_struct::{Bbox, Xyxy};

#[allow(clippy::type_complexity)]
pub struct YoloModel {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
}

impl YoloModel {
    pub fn get_bbox(
        &self,
        input_image: &DynamicImage,
        confidence_threshold: f32,
        iou_threshold: f32,
        imgsz: u32,
        class_maps: HashMap<String, String>,
    ) -> Result<Vec<Bbox>, Error> {
        // Preprocess
        let preprocess_image = preprocess(input_image, imgsz);

        // run forward pass and then convert result to f32
        let forward = self.model.run(tvec![preprocess_image.to_owned().into()]).unwrap();
        let output = forward.get(0).unwrap().to_array_view::<f32>().unwrap().view().t().into_owned();

        // process results(reference: https://github.com/AndreyGermanov/yolov8_onnx_rust/blob/main/src/main.rs)
        let mut bboxes: Vec<Bbox> = vec![];
        let output = output.slice(s![..,..,0]);
        for row in output.axis_iter(Axis(0)) {
            let row:Vec<_> = row.iter().map(|x| *x).collect();
            let (class_id, confidence) = row.iter().skip(4).enumerate()
                .map(|(index,value)| (index,*value))
                .reduce(|accum, row| if row.1>accum.1 { row } else {accum}).unwrap();
            if confidence < confidence_threshold {
                continue
            }

            // if confidence >= confidence_threshold {
            let x = row[0] / imgsz as f32;
            let y = row[1] / imgsz as f32;
            let w = row[2] / imgsz as f32;
            let h = row[3] / imgsz as f32;
            let class_name = class_maps
                .get(&class_id.to_string())
                .unwrap_or(&"unknown".to_string())
                .clone();
            // 推論結果は640x640で行われているので、元の画像でやるとズレる
            // xywhnの元になった画像サイズにすること(入力画像とは限らないので)
            let bbox = Bbox::new(
                x, y, w, h,
                confidence, class_name,
                imgsz, imgsz,
            );
            bboxes.push(bbox);
        }
        Ok(nms_boxes(bboxes, iou_threshold))
    }
}

pub fn load_yolo_model(model_path: &str, input_size: (u32, u32)) -> YoloModel {
    let pred_model = tract_onnx::onnx()
        .model_for_path(model_path)
        .unwrap()
        .with_input_fact(0, f32::fact([1, 3, input_size.0 as i32, input_size.1 as i32]).into())
        .unwrap()
        .into_optimized()
        .unwrap()
        .into_runnable()
        .unwrap();
    YoloModel { model: pred_model }
}

pub fn letterbox(input_image: &DynamicImage, target_size: u32) -> DynamicImage {
    let img_width = input_image.width();
    let img_height = input_image.height();

    // スケールは短辺基準ではなく「長辺を target_size にフィット」
    let scale = target_size as f32 / img_width.max(img_height) as f32;
    let new_width = (img_width as f32 * scale).round() as u32;
    let new_height = (img_height as f32 * scale).round() as u32;

    // リサイズされた画像
    let resized = image::imageops::resize(
        &input_image.to_rgb8(),
        new_width,
        new_height,
        image::imageops::FilterType::Triangle,
    );

    // target_size の正方形キャンバスを作成
    let mut canvas = image::RgbImage::from_pixel(target_size, target_size, image::Rgb([0, 0, 0]));

    // 貼り付けオフセット（中央）
    let x_offset = ((target_size as i32 - new_width as i32) / 2).max(0) as u32;
    let y_offset = ((target_size as i32 - new_height as i32) / 2).max(0) as u32;

    // リサイズ画像を中央に貼り付け
    image::imageops::replace(&mut canvas, &resized, x_offset as i64, y_offset as i64);

    DynamicImage::ImageRgb8(canvas)
}


/// Resize the image with black padding for keeping aspect ratio.
/// This preprocess is equal to letterbox processing in ultralytics utils.
fn preprocess(input_image: &DynamicImage, target_size: u32) -> Tensor {
    let padded = letterbox(input_image, target_size);
    
    // Convert tract tensor.
    // (Batch, Channel, Height, Width)
    // Choice c: channel and get pixel [u8; 4], and normalize.
    let image: Tensor = tract_ndarray::Array4::from_shape_fn(
        (1, 3, target_size as usize, target_size as usize),
        |(_, c, y, x)| padded.get_pixel(x as u32, y as u32)[c] as f32 / 255.0,
    ).into();
    image
}

fn nms_boxes(mut boxes: Vec<Bbox>, iou_threshold: f32) -> Vec<Bbox> {
    // Sort confidence as descending.
    boxes.sort_by(|a, b| b.conf.partial_cmp(&a.conf).unwrap_or(Ordering::Equal));

    let mut keep = Vec::new();
    while let Some(current) = boxes.pop() {
        keep.push(current.clone());
        boxes.retain(|b| calculate_iou(&current.xyxy, &b.xyxy) <= iou_threshold);
    }
    keep
}

pub fn calculate_iou(box1: &Xyxy, box2: &Xyxy) -> f32 {
    let x1 = box1.x1.max(box2.x1);
    let y1 = box1.y1.max(box2.y1);
    let x2 = box1.x2.min(box2.x2);
    let y2 = box1.y2.min(box2.y2);

    let intersection_width = x2.saturating_sub(x1);
    let intersection_height = y2.saturating_sub(y1);
    let intersection = (intersection_width as f32) * (intersection_height as f32);

    let area1 = (box1.x2.saturating_sub(box1.x1) as f32) * (box1.y2.saturating_sub(box1.y1) as f32);
    let area2 = (box2.x2.saturating_sub(box2.x1) as f32) * (box2.y2.saturating_sub(box2.y1) as f32);
    let union = area1 + area2 - intersection;

    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, GenericImageView, Rgba};

    /// helper function to generate a dummy image.
    fn create_dummy_image(width: u32, height: u32, color: [u8; 4]) -> DynamicImage {
        let mut img = image::ImageBuffer::from_pixel(width, height, Rgba(color));
        DynamicImage::ImageRgba8(img)
    }

    #[test]
    fn test_letter_box() {
        let dummy = create_dummy_image(128, 32, [100,0,0,200]);
        let target_size = 64;
        let letterboxed = letterbox(&dummy, target_size);

        println!("{:?}", letterboxed);

        assert_eq!(letterboxed.width(), 64);
        assert_eq!(letterboxed.height(), 64);

        // 上部はパディングされてるか
        let top_pixel = letterboxed.get_pixel(0, 0);
        assert_eq!(top_pixel[0], 0);
        assert_eq!(top_pixel[1], 0);
        assert_eq!(top_pixel[2], 0);

        // 中央はパディングされてないはず
        let center_pixel = letterboxed.get_pixel(32, 32);
        println!("{:?}", center_pixel);
        assert_eq!(center_pixel[0], 100);
        assert_eq!(center_pixel[1], 0);
        assert_eq!(center_pixel[2], 0);
    }

    #[test]
    fn test_nms_bboxes() {
        let example1 = [
            Bbox::new(
                0.3, 0.5, 0.1, 0.1,
                0.5, "car".to_string(),
                640, 640,
            ),
            Bbox::new(
                0.3, 0.5, 0.1, 0.1,
                0.5, "car".to_string(),
                640, 640,
            ),
        ].to_vec();
        let results = nms_boxes(example1, 0.7);
        assert_eq!(results.len(), 1);

        let example1 = [
            Bbox::new(
                0.3, 0.5, 0.1, 0.1,
                0.5, "car".to_string(),
                640, 640,
            ),
            Bbox::new(
                0.4, 0.7, 0.1, 0.1,
                0.5, "car".to_string(),
                640, 640,
            ),
        ].to_vec();
        let results = nms_boxes(example1, 0.7);
        assert_eq!(results.len(), 2); 
    }
}
