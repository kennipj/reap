mod common;

use std::collections::BTreeMap;

use reap::text::{Rectangle, extract_char_bboxes, extract_text_blocks};

use crate::common::{edge_pdf, load_doc, pdf};

const PAGE_WIDTH_TOL: f64 = 613.0;

fn assert_normalized(rect: Rectangle) {
    assert!(
        rect.left < rect.right,
        "expected left < right, got left={} right={}",
        rect.left,
        rect.right
    );
    assert!(
        rect.top < rect.bottom,
        "expected top < bottom, got top={} bottom={}",
        rect.top,
        rect.bottom
    );
}

fn within_page(rect: &Rectangle) -> bool {
    rect.left >= -1.0
        && rect.top >= -1.0
        && rect.right <= 613.0
        && rect.bottom <= 793.0
        && rect.left < rect.right
        && rect.top < rect.bottom
}

#[test]
fn canonical_rectangles_are_normalized() {
    for path in [pdf("0000.pdf"), pdf("0001.pdf"), pdf("fw2.pdf")] {
        let doc = load_doc(&path);

        let chars = extract_char_bboxes(&doc);
        assert!(!chars.is_empty(), "expected chars for {}", path);
        for ch in chars {
            assert_normalized(ch.bbox);
        }

        let words = extract_text_blocks(&doc);
        assert!(!words.is_empty(), "expected words for {}", path);
        for word in words {
            assert_normalized(word.bbox);
        }

        let blocks = extract_text_blocks(&doc);
        assert!(!blocks.is_empty(), "expected blocks for {}", path);
        for block in blocks {
            assert_normalized(block.bbox);
        }
    }
}

#[test]
fn multi_page_y_is_stacked_and_monotonic() {
    let doc = load_doc(&pdf("fw2.pdf"));
    let chars = extract_char_bboxes(&doc);
    assert!(!chars.is_empty(), "expected chars for fw2.pdf");

    let mut per_page: BTreeMap<usize, (f64, f64)> = BTreeMap::new();
    for ch in chars {
        let entry = per_page
            .entry(ch.page_index)
            .or_insert((f64::INFINITY, f64::NEG_INFINITY));
        entry.0 = entry.0.min(ch.bbox.top);
        entry.1 = entry.1.max(ch.bbox.bottom);
    }

    assert!(
        per_page.len() > 1,
        "expected multiple pages in fw2.pdf, got {}",
        per_page.len()
    );

    let page_ranges: Vec<(usize, (f64, f64))> = per_page.into_iter().collect();
    for pair in page_ranges.windows(2) {
        let (prev_page, (_, prev_bottom)) = pair[0];
        let (next_page, (next_top, _)) = pair[1];
        assert!(
            prev_bottom < next_top,
            "expected page {} max bottom ({}) < page {} min top ({})",
            prev_page,
            prev_bottom,
            next_page,
            next_top
        );
    }
}

#[test]
fn form_xobject_words_stay_in_reasonable_bounds() {
    let doc = load_doc(&edge_pdf("edge_form_xobject_transform.pdf"));
    let has_form_xobject = doc.objects.values().any(|obj| {
        if let reap::model::Object::Stream { dict, .. } = obj {
            dict.get("Subtype").and_then(|v| v.as_name()) == Some("Form")
        } else {
            false
        }
    });
    assert!(has_form_xobject, "expected Form XObject stream in fixture");

    let has_annot_ap = doc.objects.values().any(|obj| {
        obj.as_dict().is_some_and(|d| {
            d.get("Subtype").and_then(|v| v.as_name()) == Some("Widget") && d.contains_key("AP")
        })
    });
    assert!(
        has_annot_ap,
        "expected widget annotation appearance stream in fixture"
    );

    let words = extract_text_blocks(&doc);
    assert!(
        !words.is_empty(),
        "expected non-empty words for form fixture"
    );

    for w in &words {
        let coords = [w.bbox.left, w.bbox.top, w.bbox.right, w.bbox.bottom];
        for coord in coords {
            assert!(
                coord.abs() <= 1000.0,
                "found absurd coordinate {} for word '{}' with bbox {:?}",
                coord,
                w.text,
                w.bbox
            );
        }
    }
}

#[test]
fn form_xobject_filled_values_are_found_on_page() {
    let doc = load_doc(&edge_pdf("edge_form_xobject_transform.pdf"));
    let words = extract_text_blocks(&doc);

    for token in ["AMT-ALPHA", "AMT-BETA", "AMT-GAMMA", "ID-XX-7834"] {
        let on_page_hits = words
            .iter()
            .filter(|w| w.text == token && within_page(&w.bbox))
            .count();
        assert!(
            on_page_hits >= 4,
            "expected at least 4 on-page hits for {}, got {}",
            token,
            on_page_hits
        );
    }
}

#[test]
fn rotated_pages_place_control_labels_on_left_side() {
    let doc = load_doc(&edge_pdf("edge_rotation_form_transform.pdf"));
    let words = extract_text_blocks(&doc);

    let control_hits: Vec<_> = words
        .iter()
        .filter(|w| w.page_index <= 1 && w.text == "Control")
        .collect();
    assert!(
        !control_hits.is_empty(),
        "expected Control labels on rotated pages"
    );

    for hit in control_hits {
        assert!(
            hit.bbox.left >= -1e-3 && hit.bbox.right <= PAGE_WIDTH_TOL,
            "expected Control label to remain on page after rotation, got bbox {:?}",
            hit.bbox
        );
        assert!(
            hit.bbox.left < hit.bbox.right,
            "expected normalized x extents for Control label, got bbox {:?}",
            hit.bbox
        );
    }
}

#[test]
fn rotated_pages_keep_form_field_values_on_page() {
    let doc = load_doc(&edge_pdf("edge_rotation_form_transform.pdf"));
    let words = extract_text_blocks(&doc);

    for token in ["AMT-ROT-A", "AMT-ROT-B", "AMT-ROT-C"] {
        let hits: Vec<_> = words
            .iter()
            .filter(|w| w.page_index <= 1 && w.text == token)
            .collect();
        assert!(
            !hits.is_empty(),
            "expected on-page hits for {} on rotated pages",
            token
        );

        for hit in hits {
            assert!(
                hit.bbox.left >= -1e-3,
                "expected non-negative x for {}, got {:?}",
                token,
                hit.bbox
            );
            assert!(
                hit.bbox.left < hit.bbox.right,
                "expected normalized x extents for {}, got {:?}",
                token,
                hit.bbox
            );
            assert!(
                hit.bbox.right <= PAGE_WIDTH_TOL,
                "expected on-page right bound for {}, got {:?}",
                token,
                hit.bbox
            );
        }
    }
}

#[test]
fn type3_fractional_widths_do_not_explode_bboxes() {
    let doc = load_doc(&edge_pdf("edge_type3_fractional_widths.pdf"));
    let has_type3 = doc.objects.values().any(|obj| {
        obj.as_dict()
            .is_some_and(|d| d.get("Subtype").and_then(|v| v.as_name()) == Some("Type3"))
    });
    assert!(has_type3, "expected Type3 font object in fixture");

    let words = extract_text_blocks(&doc);
    assert!(
        !words.is_empty(),
        "expected extracted words for Type3 fractional-width fixture"
    );

    let employee = words
        .iter()
        .find(|w| w.page_index == 0 && w.text == "Employee")
        .expect("expected Employee token on page 0");
    let employee_width = employee.bbox.right - employee.bbox.left;
    assert!(
        employee_width < 120.0,
        "expected compact Employee width, got {} ({:?})",
        employee_width,
        employee.bbox
    );
    assert!(
        employee.bbox.right <= PAGE_WIDTH_TOL,
        "expected Employee token to stay on page, got {:?}",
        employee.bbox
    );

    for w in words.iter().filter(|w| w.page_index == 0) {
        let width = w.bbox.right - w.bbox.left;
        assert!(
            width < 500.0,
            "expected non-exploded width for '{}' on page 0, got {} ({:?})",
            w.text,
            width,
            w.bbox
        );
    }

    let chars = extract_char_bboxes(&doc);
    let l_chars: Vec<_> = chars
        .iter()
        .filter(|ch| {
            ch.page_index == 0
                && ch.ch == 'l'
                && ch.bbox.top > 50.0
                && ch.bbox.top < 90.0
                && ch.bbox.left > 130.0
                && ch.bbox.left < 200.0
        })
        .collect();
    assert!(
        !l_chars.is_empty(),
        "expected at least one header-row 'l' glyph in Type3 fixture"
    );
    assert!(
        l_chars
            .iter()
            .any(|ch| (ch.bbox.right - ch.bbox.left) < 20.0),
        "expected at least one compact 'l' glyph width, got {:?}",
        l_chars
            .iter()
            .map(|ch| (ch.bbox.right - ch.bbox.left, ch.bbox))
            .collect::<Vec<_>>()
    );
}

#[test]
fn cid_widths_are_not_near_zero() {
    let doc = load_doc(&edge_pdf("edge_cid_width_nonzero_bulk.pdf"));
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
    let has_cid_widths = doc.objects.values().any(|obj| {
        obj.as_dict().is_some_and(|d| {
            d.get("Subtype").and_then(|v| v.as_name()) == Some("CIDFontType2")
                && d.get("W").and_then(|v| v.as_array()).is_some()
        })
    });
    assert!(
        has_cid_widths,
        "expected CIDFontType2 with /W widths in fixture"
    );

    let words = extract_text_blocks(&doc);
    assert!(
        !words.is_empty(),
        "expected extracted words for CID width fixture"
    );

    for token in ["Employee's", "Employer"] {
        let hit = words
            .iter()
            .find(|w| w.page_index == 0 && w.text == token)
            .unwrap_or_else(|| panic!("expected '{}' token on page 0", token));
        let width = hit.bbox.right - hit.bbox.left;
        assert!(
            width > 5.0,
            "expected non-degenerate width for '{}', got {} ({:?})",
            token,
            width,
            hit.bbox
        );
    }

    let mut checked = 0usize;
    for w in words.iter().filter(|w| w.page_index == 0) {
        let alpha_num_len = w.text.chars().filter(|ch| ch.is_alphanumeric()).count();
        if alpha_num_len < 3 {
            continue;
        }
        checked += 1;
        let width = w.bbox.right - w.bbox.left;
        assert!(
            width > 1.0,
            "expected meaningful width for page-0 alphanumeric block '{}', got {} ({:?})",
            w.text,
            width,
            w.bbox
        );
    }
    assert!(
        checked > 30,
        "expected to validate many alphanumeric blocks on page 0, checked {}",
        checked
    );
}

#[test]
fn filled_courier_tokens_are_not_truncated() {
    let doc = load_doc(&edge_pdf("edge_courier_fields.pdf"));
    let words = extract_text_blocks(&doc);

    let full: Vec<_> = words.iter().filter(|w| w.text == "ID-AAA-0001").collect();
    assert!(
        full.len() >= 4,
        "expected at least 4 full synthetic ID tokens, got {}",
        full.len()
    );

    assert!(
        !words.iter().any(|w| w.text == "ID-AAA-000"),
        "found truncated synthetic ID token in extracted words"
    );

    for w in full {
        let width = w.bbox.right - w.bbox.left;
        let height = w.bbox.bottom - w.bbox.top;
        assert!(
            (30.0..70.0).contains(&width),
            "unexpected synthetic ID width {} for bbox {:?}",
            width,
            w.bbox
        );
        assert!(
            (4.0..12.0).contains(&height),
            "unexpected synthetic ID height {} for bbox {:?}",
            height,
            w.bbox
        );
    }
}

#[test]
fn filled_courier_tokens_have_compact_height() {
    let doc = load_doc(&edge_pdf("edge_courier_fields.pdf"));
    let words = extract_text_blocks(&doc);

    for token in ["MASK-KEY1", "ORG-ABCD-01"] {
        let hits: Vec<_> = words.iter().filter(|w| w.text == token).collect();
        assert!(
            hits.len() >= 4,
            "expected at least 4 occurrences of {}, got {}",
            token,
            hits.len()
        );

        for w in hits {
            let height = w.bbox.bottom - w.bbox.top;
            assert!(
                (4.0..12.0).contains(&height),
                "expected compact bbox height for {}, got {} ({:?})",
                token,
                height,
                w.bbox
            );
        }
    }
}
