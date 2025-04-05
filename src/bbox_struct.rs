use anyhow::{Error, Result};
use image::DynamicImage;

#[derive(Debug, Clone)]
pub struct Bbox {
    /// xywhn format.
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub conf: f32,
    pub cls: String,
}

impl Bbox {
    pub fn new(x: f32, y: f32, w: f32, h: f32, conf: f32, cls: String) -> Bbox {
        Bbox { x, y, w, h, conf, cls }
    }

    pub fn xywhn2xyxy(
        &mut self,
        original_image: &DynamicImage,
    ) -> Vec<u32> {
        let img_width = original_image.width() as f32;
        let img_height = original_image.height() as f32;

        let x1 = (self.x - self.w / 2.0) * img_width;
        let y1 = (self.y - self.h / 2.0) * img_height;
        let x2 = (self.x + self.w / 2.0) * img_width;
        let y2 = (self.y + self.h / 2.0) * img_height;

        (&[x1 as u32, y1 as u32, x2 as u32, y2 as u32]).to_vec()
    }

    pub fn crop_bbox(&mut self, original_image: &DynamicImage) -> Result<DynamicImage, Error> {
        let xyxy = self.xywhn2xyxy(original_image);
        let bbox_width = xyxy[3] - xyxy[1];
        let bbox_height = xyxy[4] - xyxy[2];
        Ok(original_image.to_owned().crop_imm(
            xyxy[1],
            xyxy[3],
            bbox_width,
            bbox_height,
        ))
    }
}