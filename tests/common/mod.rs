#![allow(dead_code)]

use reap::parser::{ParseError, Parser, PdfDoc};
use reap::rtree::TextBlockIndex;
use reap::text::extract_text_blocks;
use reap::tokenizer::Lexer;

pub const EDGE_PDFS_DIR: &str = "tests/pdfs/edge";
pub const PDFS_DIR: &str = "tests/pdfs";

pub fn edge_pdf(name: &str) -> String {
    format!("{EDGE_PDFS_DIR}/{name}")
}

pub fn pdf(name: &str) -> String {
    format!("{PDFS_DIR}/{name}")
}

pub fn load_doc(path: &str) -> PdfDoc {
    let lexer = Lexer::from_file(path).expect("failed to load file");
    Parser::new(lexer).parse().expect("failed to parse PDF")
}

pub fn parse_doc_with_password(path: &str, password: Option<&[u8]>) -> Result<PdfDoc, ParseError> {
    let lexer = Lexer::from_file(path).expect("failed to load fixture");
    Parser::new(lexer).parse_with_password(password)
}

pub fn load_index(path: &str) -> TextBlockIndex {
    let doc = load_doc(path);
    let blocks = extract_text_blocks(&doc);
    TextBlockIndex::new(blocks)
}
