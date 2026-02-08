use reap::parser::Parser;
use reap::text::{extract_char_bboxes, extract_text_blocks};
use reap::tokenizer::Lexer;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut mode = "chars";
    let mut path = "tests/pdfs/fw2.pdf".to_string();
    let mut password: Option<String> = None;

    let mut i = 0usize;
    while i < args.len() {
        let arg = &args[i];
        match arg.as_str() {
            "--words" => {
                eprintln!("--words is no longer supported; use --blocks");
                std::process::exit(2);
            }
            "--chars" => mode = "chars",
            "--blocks" => mode = "blocks",
            "--password" => {
                if i + 1 >= args.len() {
                    eprintln!("missing value for --password");
                    std::process::exit(2);
                }
                password = Some(args[i + 1].clone());
                i += 1;
            }
            _ if arg.starts_with("--password=") => {
                password = Some(arg["--password=".len()..].to_string());
            }
            _ => {
                path = arg.clone();
            }
        }
        i += 1;
    }

    let lexer = match Lexer::from_file(&path) {
        Ok(lexer) => lexer,
        Err(err) => {
            eprintln!("failed to load {}: {}", path, err);
            std::process::exit(2);
        }
    };
    let doc = match Parser::new(lexer).parse_with_password(password.as_deref().map(str::as_bytes)) {
        Ok(doc) => doc,
        Err(err) => {
            eprintln!("failed to parse {}: {}", path, err);
            std::process::exit(2);
        }
    };

    if mode == "blocks" {
        let blocks = extract_text_blocks(&doc);
        for b in blocks {
            println!(
                "page={} block=\"{}\" bbox=({:.2}, {:.2}, {:.2}, {:.2})",
                b.page_index, b.text, b.bbox.left, b.bbox.top, b.bbox.right, b.bbox.bottom
            );
        }
    } else {
        let chars = extract_char_bboxes(&doc);
        for ch in chars {
            println!(
                "page={} ch='{}' bbox=({:.2}, {:.2}, {:.2}, {:.2})",
                ch.page_index, ch.ch, ch.bbox.left, ch.bbox.top, ch.bbox.right, ch.bbox.bottom
            );
        }
    }
}
