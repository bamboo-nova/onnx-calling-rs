use clap::Parser;
//use serde_json::{Value, to_writer_pretty};
//use std::fs::File;
use std::error::Error;

mod args;
mod bbox_struct;
mod yolo;
mod onnx_metadata;

use args::Args;
use yolo::YoloModel;
use crate::yolo::load_yolo_model;
use crate::onnx_metadata::get_class_ids;

type MyResult<T> = Result<T, Box<dyn Error>>;

#[derive(Debug)]
pub struct Config {
    source: String,
    model_hash: String,
    conf_threshold: f32,
    iou_threhold: f32,
}

pub fn get_args() -> MyResult<Config> {
    let matches = Args::parse();

    Ok(Config{
        source: matches.source,
        model_hash: matches.yolo_model,
        conf_threshold: matches.conf_threshold,
        iou_threhold: matches.iou_threhold,
    })
}

#[allow(deprecated)]
pub fn run(config: Config) -> MyResult<()> {
    let image = image::open(config.source)?;

    let yolo_model: YoloModel = load_yolo_model(&config.model_hash, (640, 640));
    let class_maps = get_class_ids(config.model_hash)?;
    let results = yolo_model.get_bbox(
        &image,
        config.conf_threshold,
        config.iou_threhold,
        640,
        class_maps,
    )?;
    println!("{:?}", results);
    Ok(())
}
