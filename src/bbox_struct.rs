use anyhow::{Error, Result};
use image::DynamicImage;

#[derive(Debug, Clone)]
pub struct Bbox {
    pub xywhn: Xywhn,
    pub xyxy: Xyxy,
    pub conf: f32,
    pub cls: String,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Xywhn {
    /// xywhn format.
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Xywhn {
    pub fn is_normalized(&self) -> bool {
        self.x >= 0.0 && self.x <= 1.0 &&
        self.y >= 0.0 && self.y <= 1.0 &&
        self.w >= 0.0 && self.w <= 1.0 &&
        self.h >= 0.0 && self.h <= 1.0
    }
}

#[derive(Debug, Clone)]
pub struct Xyxy {
    /// xyxy format.
    pub x1: u32,
    pub y1: u32,
    pub x2: u32,
    pub y2: u32,
}

impl Bbox {
    pub fn new(x: f32, y: f32, w: f32, h: f32, conf: f32, cls: String, width: u32, height: u32) -> Bbox {
        let bbox_xyxy = Bbox::xywhn2xyxy(x, y, w, h, width as f32, height as f32);
        let xywhn = Xywhn { x, y, w, h };
        if xywhn.is_normalized() {
            Bbox { 
                xywhn,
                xyxy: Xyxy { x1: bbox_xyxy[0], y1: bbox_xyxy[1], x2: bbox_xyxy[2], y2: bbox_xyxy[3]},
                conf,
                cls,
            }
        } else {
            Bbox { 
                xywhn: Xywhn {x: 0.0, y: 0.0, w: 0.0, h: 0.0},
                xyxy: Xyxy { x1: 0, y1: 0, x2: 0, y2: 0},
                conf,
                cls,
            }
        }
    }

    pub fn xywhn2xyxy(
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        img_width: f32,
        img_height: f32,
    ) -> Vec<u32> {
        let x1 = (x - w / 2.0) as f32 * img_width;
        let y1 = (y - h / 2.0) as f32 * img_height;
        let x2 = (x + w / 2.0) as f32 * img_width;
        let y2 = (y + h / 2.0) as f32 * img_height;

        (&[x1 as u32, y1 as u32, x2 as u32, y2 as u32]).to_vec()
    }

    pub fn crop_bbox(&mut self, original_image: &DynamicImage) -> Result<DynamicImage, Error> {
        let bbox_width = self.xyxy.x2 - self.xyxy.x1;
        let bbox_height = self.xyxy.y2 - self.xyxy.y1;
        Ok(original_image.to_owned().crop_imm(
            self.xyxy.x1,
            self.xyxy.y1,
            bbox_width,
            bbox_height,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xywhn2xyxy() {
        let xyxy = Bbox::xywhn2xyxy(0.5, 0.5, 0.2, 0.4, 100.0, 100.0);
        assert_eq!(xyxy, vec![40, 30, 60, 70]);
    }

    #[test]
    fn test_is_normalized() {
        let xywhn = Xywhn {x: 0.2, y: 0.3, w: 0.1, h: 0.1};
        assert!(xywhn.is_normalized());
    }
}
