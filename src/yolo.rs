use anyhow::{Error, Result};
use image::DynamicImage;
//use std::cmp::Ordering;
//use std::cmp::PartialOrd;
use tract_core::plan::SimplePlan;
use tract_ndarray::{s, Axis};
use tract_onnx::prelude::*;

use crate::bbox_struct::Bbox;

#[allow(clippy::type_complexity)]
pub struct YoloModel {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
}

impl YoloModel {
    pub fn get_bbox(
        &self,
        input_image: &DynamicImage,
        confidence_threshold: f32,
        iou_threhold: f32,
        imgsz: u32
    ) -> Result<Vec<Bbox>, Error> {
        // Preprocess
        let preprocess_image = preprocess(input_image, imgsz);

        // run forward pass and then convert result to f32
        let forward = self.model.run(tvec![preprocess_image.to_owned().into()])?;
        let results = forward[0].to_array_view::<f32>()?.view().t().into_owned();

        // process results
        let mut bboxes: Vec<Bbox> = vec![];
        for i in 0..results.len_of(tract_ndarray::Axis(0)) {
            let row = results.slice(s![i, .., ..]);

            let row_1d = row.index_axis(Axis(0), 0);         // (例えば row.shape() == [1, 85] のとき)
            let class_scores = row_1d.slice(s![4..]);
            let (class_id, confidence) = class_scores
                .iter()
                .enumerate()
                .fold((0, 0.0), |(best_idx, best_score), (idx, &score)| {
                    if score > best_score {
                        (idx, score)
                    } else {
                        (best_idx, best_score)
                    }
                });

            if confidence >= confidence_threshold {
                let x = row[[0, 0]];
                let y = row[[1, 0]];
                let w = row[[2, 0]];
                let h = row[[3, 0]];
                let bbox = Bbox::new(x, y, w, h, confidence, class_id.to_string());
                bboxes.push(bbox);
            }
        }
        // Ok(nms_boxes(bboxes))
        Ok(bboxes)
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

/// Resize the image with black padding for keeping aspect ratio.
/// This preprocess is equal to letterbox processing in ultralytics utils.
fn preprocess(input_image: &DynamicImage, target_size: u32) -> Tensor {
    let img_width = input_image.width();
    let img_height = input_image.height();
    let scale = (target_size / (img_width.max(img_height))) as f32;

    // Resize for keeping aspect ratio.
    let update_width = (img_width as f32 * scale) as u32;
    let update_height = (img_height as f32 * scale) as u32;
    let resized = image::imageops::resize(
        &input_image.to_rgb8(),
        update_width,
        update_height,
        image::imageops::FilterType::Triangle,
    );

    // Define black image and replace resized image into that.
    let mut padded = image::RgbImage::new(target_size, target_size);
    image::imageops::replace(
        &mut padded,
        &resized,
        (target_size - update_width) as i64 / 2,
        (target_size - update_height) as i64 / 2,
    );
    
    // Convert tract tensor.
    // (Batch, Channel, Height, Width)
    // Choice c: channel and get pixel [u8; 4], and normalize.
    let image: Tensor = tract_ndarray::Array4::from_shape_fn(
        (1, 3, target_size as usize, target_size as usize),
        |(_, c, y, x)| padded.get_pixel(x as u32, y as u32)[c] as f32 / 255.0,
    ).into();
    image
}



