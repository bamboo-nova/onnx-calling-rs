use tract_onnx::prelude::*;
use tract_onnx::model::Onnx;
use std::collections::HashMap;

fn parse_classes_map(raw: &str) -> HashMap<u32, String> {
    let json_like = raw
        .replace('\'', "\"")
        .replace(": ", "\": ")
        .replace("{", "{\"")
        .replace(", ", ", \"");

    serde_json::from_str::<HashMap<String, String>>(&json_like)
        .expect("Failed to parse class names")
        .into_iter()
        .map(|(k, v)| (k.parse::<u32>().unwrap(), v))
        .collect()
}

fn main() -> TractResult<()> {
    let model_proto = Onnx::default().proto_model_for_path("yolov8n.onnx")?;
    let classes = parse_classes_map(&model_proto.metadata_props[10].value);

    // Vec にして sort_by_key（効率的・読みやすい）
    let mut sorted: Vec<_> = classes.into_iter().collect();
    sorted.sort_by_key(|(k, _)| *k);

    for (k, v) in sorted {
        println!("{}: {}", k, v);
    }

    // println!("{:?}", &model_proto.metadata_props);

    Ok(())
}
