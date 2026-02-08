mod common;

use std::fs;

use reap::rtree::TextBlockIndex;
use reap::text::{extract_char_bboxes, extract_text_blocks};

use crate::common::{edge_pdf, load_doc};

fn percentile(mut values: Vec<f64>, numer: usize, denom: usize) -> f64 {
    assert!(!values.is_empty(), "expected non-empty values");
    values.sort_by(|a, b| a.total_cmp(b));
    let idx = ((values.len() - 1) * numer) / denom;
    values[idx]
}

#[test]
fn hex_string_text_uses_decoded_bytes() {
    let path = edge_pdf("edge_hex_string_text.pdf");
    let bytes = fs::read(&path).expect("failed to read edge fixture");
    assert!(
        bytes
            .windows(b"<576F726B6572277320696420636F6465206E6F2E> Tj".len())
            .any(|w| w == b"<576F726B6572277320696420636F6465206E6F2E> Tj"),
        "expected hex-string text operator in fixture bytes"
    );

    let doc = load_doc(&path);
    let blocks = extract_text_blocks(&doc);

    assert!(
        !blocks.iter().any(|b| b.text.contains("484558")),
        "found undecoded hex digits in text blocks: {:?}",
        blocks.iter().map(|b| b.text.as_str()).collect::<Vec<_>>()
    );

    let normalized = blocks
        .iter()
        .map(|b| b.text.replace('\u{2019}', "'"))
        .collect::<Vec<_>>()
        .join(" ");

    assert!(
        normalized.contains("Worker's id code no."),
        "expected synthetic label variant in blocks, got: {:?}",
        blocks.iter().map(|b| b.text.as_str()).collect::<Vec<_>>()
    );
}

#[test]
fn mac_roman_encoding_decodes_apostrophe_and_dash() {
    let doc = load_doc(&edge_pdf("edge_mac_roman_encoding.pdf"));
    let blocks = extract_text_blocks(&doc);
    let joined = blocks
        .iter()
        .map(|b| b.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    let has_employee_label = joined.contains("Worker\u{2019}s identity number");
    assert!(
        has_employee_label,
        "expected decoded Worker\u{2019}s label in text blocks"
    );

    let has_bad_apostrophe = joined.contains("Worker\u{00D5}s identity number");
    assert!(
        !has_bad_apostrophe,
        "unexpected WinAnsi-style misdecode Worker\\u{{00D5}}s found in text blocks"
    );

    let has_copy_b_line =
        joined.contains("Copy B\u{2014}To Be Filed With Worker\u{2019}s FEDERAL Tax Return.");
    assert!(
        has_copy_b_line,
        "expected decoded Copy B line with em dash and apostrophe"
    );
}

#[test]
fn win_ansi_differences_nonbreaking_space_decodes_as_ascii_space() {
    let doc = load_doc(&edge_pdf("edge_color_headers_visible_winansi_nbsp.pdf"));
    let blocks = extract_text_blocks(&doc);
    let joined = blocks
        .iter()
        .map(|b| b.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    assert!(
        !joined.contains('€'),
        "unexpected euro sign found in decoded text: {}",
        joined
    );

    for phrase in [
        "EMPLOYEE NAME",
        "WEEK ENDING",
        "WITHHOLDING ALLOWANCES",
        "FILING STATUS",
    ] {
        assert!(
            joined.contains(phrase),
            "expected phrase {:?} in decoded text",
            phrase
        );
    }
}

#[test]
fn type0_tounicode_decodes_human_readable_labels() {
    let doc = load_doc(&edge_pdf("edge_type0_tounicode_cid_width.pdf"));
    let has_type0_identity = doc.objects.values().any(|obj| {
        obj.as_dict().is_some_and(|d| {
            d.get("Subtype").and_then(|v| v.as_name()) == Some("Type0")
                && d.get("Encoding").and_then(|v| v.as_name()) == Some("Identity-H")
                && d.contains_key("ToUnicode")
        })
    });
    assert!(
        has_type0_identity,
        "expected Type0 font with Identity-H and ToUnicode in fixture"
    );

    let has_real_cid_widths = doc.objects.values().any(|obj| {
        obj.as_dict().is_some_and(|d| {
            d.get("Subtype").and_then(|v| v.as_name()) == Some("CIDFontType2")
                && d.get("W").and_then(|w| w.as_array()).is_some_and(|arr| {
                    arr.iter()
                        .any(|v| matches!(v, reap::model::Object::Real(_)))
                })
        })
    });
    assert!(
        has_real_cid_widths,
        "expected CID width array with real values in fixture"
    );

    let blocks = extract_text_blocks(&doc);

    assert!(
        !blocks.is_empty(),
        "expected non-empty text blocks for synthetic Type0 fixture"
    );

    let joined = blocks
        .iter()
        .map(|b| b.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    assert!(
        joined.contains("Disclaimer"),
        "expected top disclaimer text to still be present"
    );
    assert!(
        joined.contains("Tax tip summary"),
        "expected decoded synthetic label text in blocks"
    );

    assert!(
        joined.contains("Federal income tax"),
        "expected decoded federal income tax label text in blocks"
    );
}

#[test]
fn type0_tounicode_avoids_byte_fallback_garble() {
    let doc = load_doc(&edge_pdf("edge_type0_tounicode_cid_width.pdf"));
    let blocks = extract_text_blocks(&doc);
    let joined = blocks
        .iter()
        .map(|b| b.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    assert!(
        !joined.contains("�6�R�F"),
        "found legacy garble marker from byte-level fallback decoding"
    );
    assert!(
        !joined.contains("VHFXULW"),
        "found Caesar-shift marker from incorrect byte-level fallback decoding"
    );
}

#[test]
fn cid_real_widths_keep_key_phrases_contiguous() {
    let doc = load_doc(&edge_pdf("edge_type0_tounicode_cid_width.pdf"));
    let words = extract_text_blocks(&doc);

    let has_header_phrase = words.windows(5).any(|w| {
        w[0].text == "Form"
            && w[1].text == "W-2"
            && w[2].text == "Wage"
            && w[3].text == "and"
            && w[4].text == "Tax"
    });
    assert!(
        has_header_phrase,
        "expected contiguous phrase \"Form W-2 Wage and Tax\" in words"
    );

    let has_disclaimer_phrase = words
        .windows(3)
        .any(|w| w[0].text == "this" && w[1].text == "W-2" && w[2].text == "or");
    assert!(
        has_disclaimer_phrase,
        "expected contiguous phrase \"this W-2 or\" in disclaimer text"
    );
}

#[test]
fn cid_real_widths_keep_regex_index_text_contiguous() {
    let doc = load_doc(&edge_pdf("edge_type0_tounicode_cid_width.pdf"));
    let blocks = extract_text_blocks(&doc);
    let index = TextBlockIndex::new(blocks);
    let text = index.text();
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");

    assert!(
        normalized.contains("Form W-2 Wage and Tax"),
        "expected contiguous \"Form W-2 Wage and Tax\" in regex index text"
    );
    assert!(
        normalized.contains("this W-2 or"),
        "expected contiguous \"this W-2 or\" in regex index text"
    );
}

#[test]
fn cm_concat_order_keeps_bboxes_reasonable() {
    let doc = load_doc(&edge_pdf("edge_type0_tounicode_cid_width.pdf"));
    let chars = extract_char_bboxes(&doc);
    let words = extract_text_blocks(&doc);

    assert!(
        !chars.is_empty(),
        "expected extracted chars for Type0 fixture"
    );
    assert!(
        !words.is_empty(),
        "expected extracted words for Type0 fixture"
    );

    for ch in &chars {
        let bbox = &ch.bbox;
        assert!(bbox.left.is_finite(), "non-finite char left bbox");
        assert!(bbox.top.is_finite(), "non-finite char top bbox");
        assert!(bbox.right.is_finite(), "non-finite char right bbox");
        assert!(bbox.bottom.is_finite(), "non-finite char bottom bbox");

        assert!(
            bbox.left >= -10.0 && bbox.right <= 1000.0,
            "char bbox x extents out of expected range: {:?}",
            bbox
        );
        assert!(
            bbox.top >= -10.0 && bbox.bottom <= 1200.0,
            "char bbox y extents out of expected range: {:?}",
            bbox
        );
    }

    for word in &words {
        let bbox = &word.bbox;
        assert!(bbox.left.is_finite(), "non-finite word left bbox");
        assert!(bbox.top.is_finite(), "non-finite word top bbox");
        assert!(bbox.right.is_finite(), "non-finite word right bbox");
        assert!(bbox.bottom.is_finite(), "non-finite word bottom bbox");

        assert!(
            bbox.left >= -10.0 && bbox.right <= 1000.0,
            "word bbox x extents out of expected range: {:?}",
            bbox
        );
        assert!(
            bbox.top >= -10.0 && bbox.bottom <= 1200.0,
            "word bbox y extents out of expected range: {:?}",
            bbox
        );
    }
}

#[test]
fn page0_type3_labels_have_sane_bbox_height() {
    let doc = load_doc(&edge_pdf("edge_type3_bbox_page0.pdf"));
    let has_type3 = doc.objects.values().any(|obj| {
        obj.as_dict()
            .is_some_and(|d| d.get("Subtype").and_then(|v| v.as_name()) == Some("Type3"))
    });
    assert!(has_type3, "expected a Type3 font in edge fixture");

    let words = extract_text_blocks(&doc);

    let employer_hits: Vec<_> = words
        .iter()
        .filter(|w| w.page_index == 0 && w.text == "EMPLOYERS")
        .collect();
    assert!(
        !employer_hits.is_empty(),
        "expected EMPLOYERS tokens on page 0 in edge fixture"
    );

    let on_page_hits = employer_hits
        .iter()
        .filter(|w| {
            let height = w.bbox.bottom - w.bbox.top;
            w.bbox.top >= 0.0 && w.bbox.top < 792.0 && (4.0..20.0).contains(&height)
        })
        .count();
    assert!(
        on_page_hits >= 1,
        "expected at least one on-page EMPLOYERS token with compact bbox, got {}",
        on_page_hits
    );

    for w in employer_hits {
        let height = w.bbox.bottom - w.bbox.top;
        assert!(
            w.bbox.top >= 0.0,
            "expected non-negative top for EMPLOYERS bbox, got {:?}",
            w.bbox
        );
        assert!(
            height < 50.0,
            "expected EMPLOYER'S bbox height < 50, got {} ({:?})",
            height,
            w.bbox
        );
    }
}

#[test]
fn type3_bbox_inflation_is_clamped() {
    let doc = load_doc(&edge_pdf("edge_type3_bbox_clamp.pdf"));
    let has_type3 = doc.objects.values().any(|obj| {
        obj.as_dict()
            .is_some_and(|d| d.get("Subtype").and_then(|v| v.as_name()) == Some("Type3"))
    });
    assert!(has_type3, "expected a Type3 font in clamp fixture");

    let chars = extract_char_bboxes(&doc);
    assert!(
        !chars.is_empty(),
        "expected non-empty chars for clamp fixture"
    );
    let char_heights: Vec<f64> = chars.iter().map(|c| c.bbox.bottom - c.bbox.top).collect();
    let char_p95 = percentile(char_heights.clone(), 95, 100);
    let char_max = char_heights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    println!(
        "type3 clamp char heights: count={} p95={:.2} max={:.2}",
        char_heights.len(),
        char_p95,
        char_max
    );
    assert!(
        char_p95 < 20.0,
        "expected compact char p95 height (<20), got {}",
        char_p95
    );
    assert!(
        char_max < 30.0,
        "expected compact char max height (<30), got {}",
        char_max
    );

    let words = extract_text_blocks(&doc);
    assert!(
        !words.is_empty(),
        "expected non-empty words for clamp fixture"
    );
    let word_heights: Vec<f64> = words.iter().map(|w| w.bbox.bottom - w.bbox.top).collect();
    let word_p95 = percentile(word_heights.clone(), 95, 100);
    let word_max = word_heights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    println!(
        "type3 clamp word heights: count={} p95={:.2} max={:.2}",
        word_heights.len(),
        word_p95,
        word_max
    );
    assert!(
        word_p95 < 20.0,
        "expected compact word p95 height (<20), got {}",
        word_p95
    );
    assert!(
        word_max < 30.0,
        "expected compact word max height (<30), got {}",
        word_max
    );
}
