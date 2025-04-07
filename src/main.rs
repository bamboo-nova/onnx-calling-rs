fn main() {
    if let Err(e) = onnxcalling::get_args().and_then(onnxcalling::run) {
        eprintln!("{}", e);
        std::process::exit(1);
    }
}
