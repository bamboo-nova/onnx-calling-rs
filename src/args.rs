use clap::Parser;

#[derive(Parser, Clone)]
#[command(name = "onnx-calling", version = "0.1.0", author = "chiikawa", about = "onnx model usage")]
pub struct Args {
    /// ONNX model path
    #[arg(value_name = "MODEL", help = "ONNX model path")]
    pub yolo_model: String,

    /// Image path
    #[arg(value_name = "SOURCE", help = "source image path")]
    pub source: String,

    /// class yaml path
    #[arg(value_name = "CLASSES", help = "yaml config path")]
    pub class_config: String,

    /// Confidence threshold.
    #[arg(short='c', long="conf-threshold", default_value="0.3")]
    pub conf_threshold: f32,

    /// IoU threshold.
    #[arg(short='i', long="iou-threshold", default_value="0.7")]
    pub iou_threhold: f32,
}