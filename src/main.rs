fn main() {
    if let Err(e) = onnxcalling::get_args().and_then(|args|
        onnxcalling::run(args) {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    )
}
