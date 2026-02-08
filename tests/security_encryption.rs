mod common;

use reap::parser::ParseError;
use reap::text::extract_text_blocks;

use crate::common::{edge_pdf, parse_doc_with_password};

#[test]
fn encrypted_fixture_extracts_words_with_default_and_explicit_empty_password() {
    let fixture = edge_pdf("edge_encrypted_empty_password.pdf");
    let cases: [(&str, Option<&[u8]>); 2] = [("default", None), ("explicit empty", Some(b""))];

    for (label, password) in cases {
        let doc = parse_doc_with_password(&fixture, password).expect("parse should succeed");
        let words = extract_text_blocks(&doc);
        assert!(
            !words.is_empty(),
            "expected extracted words from encrypted fixture ({})",
            label
        );
        for value in [
            "Form",
            "W-2",
            "AMOUNT_A",
            "AMOUNT_B",
            "ORG-TAX-ID",
            "ID-ALPHA-0001",
            "ORG_ALPHA",
        ] {
            assert!(
                words.iter().any(|w| w.text == value),
                "expected extracted value '{}' with {} password mode",
                value,
                label
            );
        }
    }
}

#[test]
fn encrypted_fixture_rejects_wrong_password() {
    let err = parse_doc_with_password(
        &edge_pdf("edge_encrypted_empty_password.pdf"),
        Some(b"wrong"),
    )
    .expect_err("wrong password should fail");
    assert!(matches!(err, ParseError::InvalidPassword));
}
