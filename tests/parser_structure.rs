mod common;

use std::fs;

use reap::text::extract_text_blocks;

use crate::common::{edge_pdf, load_doc};

#[test]
fn incremental_update_objects_are_parsed() {
    let doc = load_doc(&edge_pdf("edge_incremental_update_objstm.pdf"));

    for obj_num in [1265, 1266, 1273, 1274] {
        assert!(
            doc.get(obj_num, 0).is_some(),
            "expected incremental object {} 0 to be present",
            obj_num
        );
    }

    let obj_stream_count = doc
        .objects
        .values()
        .filter(|obj| match obj {
            reap::model::Object::Stream { dict, .. } => {
                dict.get("Type").and_then(|v| v.as_name()) == Some("ObjStm")
            }
            _ => false,
        })
        .count();
    assert!(
        obj_stream_count >= 1,
        "expected at least one object stream in synthetic fixture, got {}",
        obj_stream_count
    );
}

#[test]
fn incremental_update_filled_values_are_extracted() {
    let doc = load_doc(&edge_pdf("edge_incremental_update_objstm.pdf"));
    let blocks = extract_text_blocks(&doc);

    for token in ["AMT-INC-A", "AMT-INC-B"] {
        let hits = blocks.iter().filter(|b| b.text == token).count();
        assert!(
            hits >= 4,
            "expected at least 4 extracted hits for {}, got {}",
            token,
            hits
        );
    }
}

#[test]
fn artifact_tagged_text_is_included_in_blocks() {
    let path = edge_pdf("edge_artifact_marked_text.pdf");
    let bytes = fs::read(&path).expect("failed to read edge artifact fixture");
    assert!(
        bytes
            .windows(b"/Artifact BMC".len())
            .any(|w| w == b"/Artifact BMC"),
        "expected /Artifact BMC marker in fixture bytes"
    );
    assert!(
        bytes.windows(b"EMC".len()).any(|w| w == b"EMC"),
        "expected EMC marker in fixture bytes"
    );

    let doc = load_doc(&path);
    let blocks = extract_text_blocks(&doc);

    for token in ["Worker's", "verified", "ID"] {
        let hits = blocks.iter().filter(|b| b.text == token).count();
        assert!(
            hits >= 3,
            "expected at least 3 token hits for '{}', got {}",
            token,
            hits
        );
    }
}

#[test]
fn tagged_invisible_text_is_included_but_artifact_invisible_text_is_excluded() {
    let path = edge_pdf("edge_tagged_invisible_text.pdf");
    let bytes = fs::read(&path).expect("failed to read edge tagged invisible fixture");
    assert!(
        bytes
            .windows(b"/Artifact BMC".len())
            .any(|w| w == b"/Artifact BMC"),
        "expected /Artifact BMC marker in fixture bytes"
    );
    assert!(
        bytes
            .windows(b"/P <</MCID 0>> BDC".len())
            .any(|w| w == b"/P <</MCID 0>> BDC"),
        "expected tagged /P ... BDC marker in fixture bytes"
    );
    assert!(
        bytes.windows(b"3 Tr".len()).any(|w| w == b"3 Tr"),
        "expected invisible text render mode marker in fixture bytes"
    );

    let doc = load_doc(&path);
    let blocks = extract_text_blocks(&doc);

    assert!(
        blocks.iter().any(|b| b.text == "VISIBLE_BASELINE"),
        "expected visible control token to be extracted"
    );
    assert!(
        blocks.iter().any(|b| b.text == "Tagged"),
        "expected tagged invisible token 'Tagged' to be extracted"
    );
    assert!(
        blocks.iter().any(|b| b.text == "hidden"),
        "expected tagged invisible token 'hidden' to be extracted"
    );
    assert!(
        blocks.iter().any(|b| b.text == "sample"),
        "expected tagged invisible token 'sample' to be extracted"
    );
    assert!(
        !blocks.iter().any(|b| b.text.contains("ARTIFACT_HIDDEN")),
        "expected artifact-hidden invisible text to remain excluded"
    );
}
