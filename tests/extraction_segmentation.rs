mod common;

use std::fs;

use reap::model::Object;
use reap::rtree::{RegexSearchError, ScopedMergeRules, TextBlockIndex};
use reap::text::{Rectangle, TextBlock, extract_text_blocks};

use crate::common::{edge_pdf, load_doc, load_index, pdf};

#[derive(Debug, serde::Deserialize)]
struct FixtureWord {
    page: usize,
    text: String,
    bbox: [f64; 4],
}

#[derive(Debug, serde::Deserialize)]
struct Fixtures {
    #[serde(rename = "0000")]
    p0000: Vec<FixtureWord>,
    #[serde(rename = "0001")]
    p0001: Vec<FixtureWord>,
    #[serde(rename = "fw2")]
    fw2: Vec<FixtureWord>,
}

fn load_words(path: &str) -> Vec<TextBlock> {
    let doc = load_doc(path);
    extract_text_blocks(&doc)
}

fn find_word<'a>(
    words: &'a [TextBlock],
    page: usize,
    text: &str,
    bbox: &[f64; 4],
) -> Option<&'a TextBlock> {
    let mut best: Option<&TextBlock> = None;
    let mut best_dist = f64::MAX;
    for w in words
        .iter()
        .filter(|w| w.page_index == page && w.text == text)
    {
        let dist = (w.bbox.left - bbox[0]).abs()
            + (w.bbox.top - bbox[1]).abs()
            + (w.bbox.right - bbox[2]).abs()
            + (w.bbox.bottom - bbox[3]).abs();
        if dist < best_dist {
            best_dist = dist;
            best = Some(w);
        }
    }
    best
}

fn assert_bbox_close(actual: &Rectangle, expected: &[f64; 4], tol: f64) {
    let actual = [actual.left, actual.top, actual.right, actual.bottom];
    for i in 0..4 {
        let delta = (actual[i] - expected[i]).abs();
        assert!(
            delta <= tol,
            "bbox mismatch idx={} actual={} expected={} delta={} tol={}",
            i,
            actual[i],
            expected[i],
            delta,
            tol
        );
    }
}

#[test]
fn word_fixtures_match() {
    let data =
        fs::read_to_string("tests/fixtures/words_expected.json").expect("Failed to read fixtures");
    let fixtures: Fixtures = serde_json::from_str(&data).expect("Failed to parse fixtures");

    let words_0000 = load_words(&pdf("0000.pdf"));
    for f in &fixtures.p0000 {
        let w = find_word(&words_0000, f.page, &f.text, &f.bbox)
            .unwrap_or_else(|| panic!("missing word '{}' on page {}", f.text, f.page));
        assert_bbox_close(&w.bbox, &f.bbox, 1.0);
    }

    let words_0001 = load_words(&pdf("0001.pdf"));
    for f in &fixtures.p0001 {
        let w = find_word(&words_0001, f.page, &f.text, &f.bbox)
            .unwrap_or_else(|| panic!("missing word '{}' on page {}", f.text, f.page));
        assert_bbox_close(&w.bbox, &f.bbox, 1.0);
    }

    let words_fw2 = load_words(&pdf("fw2.pdf"));
    for f in &fixtures.fw2 {
        let w = find_word(&words_fw2, f.page, &f.text, &f.bbox)
            .unwrap_or_else(|| panic!("missing word '{}' on page {}", f.text, f.page));
        assert_bbox_close(&w.bbox, &f.bbox, 1.5);
    }
}

#[test]
fn header_phrase_reconstructs_from_words() {
    let words = load_words(&edge_pdf("edge_header_phrase_reconstruction.pdf"));
    let mut header_words: Vec<&TextBlock> = words
        .iter()
        .filter(|w| {
            w.page_index == 0 && w.bbox.top <= 20.0 && w.bbox.left >= 120.0 && w.bbox.left <= 280.0
        })
        .collect();
    header_words.sort_by(|a, b| a.bbox.left.partial_cmp(&b.bbox.left).unwrap());

    let header_tokens: Vec<&str> = header_words.iter().map(|w| w.text.as_str()).collect();
    let expected = ["a", "Worker's", "id", "marker", "code"];
    let mut matched = 0usize;
    for token in &header_tokens {
        if *token == expected[matched] {
            matched += 1;
            if matched == expected.len() {
                break;
            }
        }
    }

    assert_eq!(
        matched,
        expected.len(),
        "failed to reconstruct expected header phrase; header tokens: {:?}",
        header_tokens
    );
}

#[test]
fn id_and_number_are_not_merged() {
    let doc = load_doc(&edge_pdf("edge_spacing_and_regex_labels.pdf"));
    let blocks = extract_text_blocks(&doc);

    assert!(
        !blocks
            .iter()
            .any(|w| w.page_index == 0 && w.text == "IDnumber"),
        "unexpected merged page-0 token IDnumber"
    );

    assert!(
        blocks.iter().any(|w| {
            w.page_index == 0
                && w.text == "ID"
                && w.bbox.left > 140.0
                && w.bbox.left < 190.0
                && w.bbox.top > 180.0
                && w.bbox.top < 240.0
        }),
        "expected standalone ID token in the synthetic header region"
    );

    assert!(
        blocks.iter().any(|w| {
            w.page_index == 0
                && w.text == "number"
                && w.bbox.left > 155.0
                && w.bbox.left < 200.0
                && w.bbox.top > 180.0
                && w.bbox.top < 240.0
        }),
        "expected standalone number token in the synthetic header region"
    );

    assert!(
        !blocks.iter().any(|b| b.text.contains("IDnumber")),
        "unexpected merged block token IDnumber"
    );
}

#[test]
fn cross_column_tokens_do_not_merge() {
    let doc = load_doc(&edge_pdf("edge_cross_column_merge.pdf"));
    let blocks = extract_text_blocks(&doc);

    let merged_block_tokens = ["COLA1234COLA1234", "COLB5678COLB5678"];
    for token in merged_block_tokens {
        assert!(
            !blocks.iter().any(|b| b.text == token),
            "unexpected merged block token found: {}",
            token
        );
    }

    for (token, min_hits, max_width) in [("COLA1234", 4usize, 80.0), ("COLB5678", 4usize, 80.0)] {
        let block_hits: Vec<_> = blocks.iter().filter(|b| b.text == token).collect();
        assert!(
            block_hits.len() >= min_hits,
            "expected at least {} block hits for {}, got {}",
            min_hits,
            token,
            block_hits.len()
        );
        for b in &block_hits {
            let width = b.bbox.right - b.bbox.left;
            assert!(
                width <= max_width,
                "expected compact block width for {} <= {}, got {} ({:?})",
                token,
                max_width,
                width,
                b.bbox
            );
        }
    }
}

#[test]
fn white_deduction_codes_are_hidden_but_labels_remain_visible() {
    let fixture = edge_pdf("edge_hidden_white_deduction_codes.pdf");
    let doc = load_doc(&fixture);
    let blocks = extract_text_blocks(&doc);

    for hidden in ["1100MEDICARE", "2000HLTH", "2010DENTAL"] {
        assert!(
            !blocks.iter().any(|b| b.page_index == 0 && b.text == hidden),
            "expected hidden white deduction code token to be excluded: {}",
            hidden
        );
    }

    for visible in ["MEDICARE", "HLTH", "PPO", "DENTAL"] {
        assert!(
            blocks
                .iter()
                .any(|b| b.page_index == 0 && b.text == visible),
            "expected visible deduction label token to remain: {}",
            visible
        );
    }

    for header in ["Emp", "No", "Employee", "Name"] {
        assert!(
            blocks.iter().any(|b| b.page_index == 0 && b.text == header),
            "expected white-on-dark header token to remain: {}",
            header
        );
    }
}

#[test]
fn separation_colorspace_text_remains_visible() {
    let fixture = edge_pdf("edge_separation_type0_corrupt_xref.pdf");
    let bytes = fs::read(&fixture).expect("failed to read separation fixture");
    assert!(
        bytes
            .windows(b"/Separation".len())
            .any(|w| w == b"/Separation"),
        "expected /Separation colorspace marker in fixture bytes"
    );

    let doc = load_doc(&fixture);
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

    let blocks = extract_text_blocks(&doc);

    let has_phrase = blocks.windows(4).any(|w| {
        let a = w[0].text.replace('\u{2019}', "'").to_lowercase();
        let b = w[1].text.to_lowercase();
        let c = w[2].text.to_lowercase();
        let d = w[3].text.to_lowercase();
        a == "employee's" && b == "social" && c == "security" && d == "no."
    });
    assert!(
        has_phrase,
        "expected \"employee's social security no.\" phrase in separation/type0 fixture output"
    );
}

#[test]
fn inline_image_payload_does_not_block_text_extraction() {
    let fixture = edge_pdf("edge_inline_image_payload_continuation.pdf");
    let bytes = fs::read(&fixture).expect("failed to read inline-image fixture");
    for marker in [b"BI".as_slice(), b"ID".as_slice(), b"EI".as_slice()] {
        assert!(
            bytes.windows(marker.len()).any(|w| w == marker),
            "expected {:?} marker in inline-image fixture bytes",
            marker
        );
    }

    let doc = load_doc(&fixture);
    let blocks = extract_text_blocks(&doc);
    assert!(
        !blocks.is_empty(),
        "expected non-empty extraction for inline-image continuation fixture"
    );

    for token in ["DIRECT", "DEPOSIT", "Earnings", "Rockdale"] {
        assert!(
            blocks.iter().any(|b| {
                b.page_index == 0
                    && b.text
                        .to_ascii_lowercase()
                        .contains(&token.to_ascii_lowercase())
            }),
            "expected token containing '{}' in inline-image continuation extraction",
            token
        );
    }

    let page0_count = blocks.iter().filter(|b| b.page_index == 0).count();
    assert!(
        page0_count > 50,
        "expected substantial page-0 text extraction in inline-image continuation fixture, got {} blocks",
        page0_count
    );
}

#[test]
fn contents_stream_with_indirect_filter_extracts_text() {
    let fixture = edge_pdf("edge_contents_indirect_filter_flate.pdf");
    let bytes = fs::read(&fixture).expect("failed to read indirect-filter fixture");
    assert!(
        bytes.windows(b"/Filter ".len()).any(|w| w == b"/Filter ")
            && bytes.windows(b" 0 R".len()).any(|w| w == b" 0 R")
            && bytes
                .windows(b"/FlateDecode".len())
                .any(|w| w == b"/FlateDecode"),
        "expected indirect /Filter reference and Flate marker in fixture bytes"
    );

    let doc = load_doc(&fixture);
    let has_indirect_filter = doc.objects.values().any(|obj| match obj {
        Object::Stream { dict, .. } => matches!(dict.get("Filter"), Some(Object::Reference { .. })),
        _ => false,
    });
    assert!(
        has_indirect_filter,
        "expected at least one stream with indirect /Filter reference"
    );

    let blocks = extract_text_blocks(&doc);
    assert!(
        !blocks.is_empty(),
        "expected non-empty extraction for indirect-filter content stream fixture"
    );

    for token in ["INDIRECT", "FILTER", "OK"] {
        assert!(
            blocks.iter().any(|b| b.page_index == 0 && b.text == token),
            "expected token '{}' on page 0",
            token
        );
    }
}

#[test]
fn white_headers_on_green_background_remain_visible() {
    let doc = load_doc(&edge_pdf("edge_color_headers_visible_winansi_nbsp.pdf"));
    let blocks = extract_text_blocks(&doc);

    for token in ["DEDUCTIONS", "EE", "CURRENT", "YTD", "EMPLR"] {
        assert!(
            blocks.iter().any(|b| b.page_index == 0 && b.text == token),
            "expected header token to remain visible in color-header fixture: {}",
            token
        );
    }

    let has_ee_current = blocks
        .windows(2)
        .any(|w| w[0].text == "EE" && w[1].text == "CURRENT");
    assert!(
        has_ee_current,
        "expected contiguous header phrase \"EE CURRENT\""
    );

    let has_ee_ytd = blocks
        .windows(2)
        .any(|w| w[0].text == "EE" && w[1].text == "YTD");
    assert!(has_ee_ytd, "expected contiguous header phrase \"EE YTD\"");

    let has_emplr_ytd = blocks
        .windows(2)
        .any(|w| w[0].text == "EMPLR" && w[1].text == "YTD");
    assert!(
        has_emplr_ytd,
        "expected contiguous header phrase \"EMPLR YTD\""
    );
}

#[test]
fn underscore_only_blocks_are_discarded() {
    let doc = load_doc(&edge_pdf("edge_underscore_noise_filter.pdf"));
    let blocks = extract_text_blocks(&doc);

    assert!(
        !blocks
            .iter()
            .any(|b| !b.text.is_empty() && b.text.chars().all(|ch| ch == '_')),
        "expected underscore-only blocks to be discarded in underscore-noise fixture"
    );
    for token in ["Employee", "Advice"] {
        assert!(
            blocks.iter().any(|b| b.text == token),
            "expected neighboring non-noise token to remain: {}",
            token
        );
    }
}

#[test]
fn white_on_gray_headers_remain_visible() {
    let doc = load_doc(&edge_pdf("edge_color_headers_visible_winansi_nbsp.pdf"));
    let blocks = extract_text_blocks(&doc);

    for token in [
        "EARNINGS",
        "BEGIN",
        "END",
        "HOURS",
        "CURRENT",
        "YTD",
        "DEDUCTIONS",
    ] {
        assert!(
            blocks.iter().any(|b| b.page_index == 0 && b.text == token),
            "expected white-on-gray header token to remain visible in color-header fixture: {}",
            token
        );
    }
}

#[test]
fn inline_image_termination_corrupt_xref_regression() {
    let fixture = edge_pdf("edge_inline_image_termination_corrupt_xref.pdf");
    let bytes = fs::read(&fixture).expect("failed to read inline-image termination fixture");
    for marker in [b"BI".as_slice(), b"ID".as_slice(), b"EI".as_slice()] {
        assert!(
            bytes.windows(marker.len()).any(|w| w == marker),
            "expected {:?} marker in inline-image termination fixture bytes",
            marker
        );
    }
    assert!(
        bytes
            .windows(b"%CORRUPT_STARTXREF_SYNTHETIC".len())
            .any(|w| w == b"%CORRUPT_STARTXREF_SYNTHETIC"),
        "expected deterministic corrupted startxref marker in fixture bytes"
    );
    assert!(
        bytes
            .windows(b"startxref\n0".len())
            .any(|w| w == b"startxref\n0"),
        "expected malformed startxref offset in fixture bytes"
    );

    let doc = load_doc(&fixture);
    let blocks = extract_text_blocks(&doc);

    for token in ["LONGVIEW", "WORLDGATE", "HERNDON", "20170"] {
        assert!(
            blocks
                .iter()
                .any(|b| b.page_index == 0 && b.text.eq_ignore_ascii_case(token)),
            "expected page-0 token to be extracted in inline-image termination fixture: {}",
            token
        );
    }

    let has_key_id = blocks
        .iter()
        .any(|b| b.page_index == 0 && b.text == "ID-KEY-7788");
    assert!(
        has_key_id,
        "expected key synthetic ID token on page 0 in inline-image termination fixture"
    );
}

#[test]
fn search_regex_finds_compact_label_phrases() {
    let spacing_path = edge_pdf("edge_spacing_and_regex_labels.pdf");
    let bytes = fs::read(&spacing_path).expect("failed to read spacing fixture");
    assert!(
        bytes.windows(b"TJ".len()).any(|w| w == b"TJ"),
        "expected TJ operator in spacing fixture bytes"
    );

    let mut index = load_index(&spacing_path);

    let id_label_hits = index
        .search_regex(r"Worker.?s\s+id\s+marker")
        .expect("regex should compile");
    assert!(
        !id_label_hits.is_empty(),
        "expected at least one match, got {}",
        id_label_hits.len()
    );
    for hit in id_label_hits {
        let width = hit.bbox.right - hit.bbox.left;
        assert!(
            width <= 90.0,
            "expected compact ID label width <= 90, got {} ({:?})",
            width,
            hit.bbox
        );
        let normalized = hit.text.replace('\u{2019}', "'").to_lowercase();
        assert!(
            normalized.contains("id marker"),
            "expected phrase to contain id marker, got {:?}",
            hit.text
        );
    }

    let control_hits = index
        .search_regex(r"Control code")
        .expect("regex should compile");
    assert!(
        !control_hits.is_empty(),
        "expected at least one match, got {}",
        control_hits.len()
    );
    for hit in control_hits {
        let width = hit.bbox.right - hit.bbox.left;
        assert!(
            width <= 70.0,
            "expected compact control label width <= 70, got {} ({:?})",
            width,
            hit.bbox
        );
        assert_eq!(hit.text, "Control code");
    }
}

#[test]
fn search_regex_finds_expected_spaced_labels() {
    let mut index = load_index(&edge_pdf("edge_spacing_and_regex_labels.pdf"));

    let gross_pay_hits = index
        .search_regex(r"Gross\s+Pay")
        .expect("regex should compile");
    assert!(
        !gross_pay_hits.is_empty(),
        "expected at least one Gross Pay match, got {}",
        gross_pay_hits.len()
    );
    for hit in gross_pay_hits {
        let width = hit.bbox.right - hit.bbox.left;
        assert!(
            width <= 70.0,
            "expected compact Gross Pay width <= 70, got {} ({:?})",
            width,
            hit.bbox
        );
        let normalized = hit.text.replace('\u{2019}', "'").to_lowercase();
        assert!(
            normalized.contains("gross pay"),
            "expected phrase to contain gross pay, got {:?}",
            hit.text
        );
    }

    let tax_tips_hits = index
        .search_regex(r"Tax\.Tips\s+Reported\s+in\s+Box\s*7")
        .expect("regex should compile");
    assert!(
        !tax_tips_hits.is_empty(),
        "expected at least one Tax.Tips Reported in Box 7 match, got {}",
        tax_tips_hits.len()
    );
    for hit in tax_tips_hits {
        let width = hit.bbox.right - hit.bbox.left;
        assert!(
            width <= 110.0,
            "expected compact Tax.Tips Reported in Box 7 width <= 110, got {} ({:?})",
            width,
            hit.bbox
        );
        let normalized = hit.text.replace('\u{2019}', "'").to_lowercase();
        assert!(
            normalized.contains("tax.tips reported in box 7"),
            "expected phrase to contain tax.tips reported in box 7, got {:?}",
            hit.text
        );
    }

    let reported_w2_wages_hits = index
        .search_regex(r"Reported\s+W-2\s+Wages")
        .expect("regex should compile");
    assert!(
        !reported_w2_wages_hits.is_empty(),
        "expected at least one Reported W-2 Wages match, got {}",
        reported_w2_wages_hits.len()
    );
    for hit in reported_w2_wages_hits {
        let width = hit.bbox.right - hit.bbox.left;
        assert!(
            width <= 95.0,
            "expected compact Reported W-2 Wages width <= 95, got {} ({:?})",
            width,
            hit.bbox
        );
        let normalized = hit.text.replace('\u{2019}', "'").to_lowercase();
        assert!(
            normalized.contains("reported w-2 wages"),
            "expected phrase to contain reported w-2 wages, got {:?}",
            hit.text
        );
    }
}

#[test]
fn scoped_index_regex_limits_matches_to_area() {
    let mut index = load_index(&edge_pdf("edge_spacing_and_regex_labels.pdf"));
    let scope = Rectangle {
        top: 90.0,
        left: 40.0,
        bottom: 105.0,
        right: 85.0,
    };

    let mut scoped = index.scoped(scope, 1.0, 0.0, false, ScopedMergeRules::default());
    let control_hits = scoped
        .search_regex(r"Control code")
        .expect("regex should compile");
    assert!(
        !control_hits.is_empty(),
        "expected scoped index to match Control code"
    );

    let gross_hits = scoped
        .search_regex(r"Gross\s+Pay")
        .expect("regex should compile");
    assert!(
        gross_hits.is_empty(),
        "did not expect Gross Pay inside scoped region, got {} hits",
        gross_hits.len()
    );

    let base_gross_hits = index
        .search_regex(r"Gross\s+Pay")
        .expect("regex should compile");
    assert!(
        !base_gross_hits.is_empty(),
        "expected base index to still match Gross Pay"
    );
}

#[test]
fn search_regex_invalid_pattern_returns_error() {
    let mut index = load_index(&edge_pdf("edge_spacing_and_regex_labels.pdf"));
    let err = index
        .search_regex("(")
        .expect_err("pattern should be invalid");
    assert!(matches!(err, RegexSearchError::InvalidPattern(_)));
}

#[test]
fn search_regex_does_not_span_pages() {
    let blocks = vec![
        TextBlock {
            page_index: 0,
            text: "alpha".to_string(),
            bbox: Rectangle {
                top: 10.0,
                left: 10.0,
                bottom: 12.0,
                right: 20.0,
            },
        },
        TextBlock {
            page_index: 1,
            text: "beta".to_string(),
            bbox: Rectangle {
                top: 20.0,
                left: 10.0,
                bottom: 22.0,
                right: 18.0,
            },
        },
    ];
    let mut index = TextBlockIndex::new(blocks);
    let hits = index
        .search_regex(r"alpha\s+beta")
        .expect("regex should compile");
    assert!(hits.is_empty(), "expected no cross-page matches");
}

#[test]
fn scoped_merge_threshold_merges_adjacent_same_line_blocks() {
    let rect = Rectangle {
        top: 0.0,
        left: 0.0,
        bottom: 30.0,
        right: 30.0,
    };
    let blocks = vec![
        TextBlock {
            page_index: 0,
            text: "word1".to_string(),
            bbox: Rectangle {
                top: 10.0,
                left: 10.0,
                bottom: 12.0,
                right: 14.0,
            },
        },
        TextBlock {
            page_index: 0,
            text: "word2".to_string(),
            bbox: Rectangle {
                top: 10.0,
                left: 16.0,
                bottom: 12.0,
                right: 20.0,
            },
        },
    ];
    let index = TextBlockIndex::new(blocks);

    let unmerged = index.scoped(rect, 1.0, 1.9, false, ScopedMergeRules::default());
    assert_eq!(unmerged.blocks().len(), 2, "expected no merge below gap");

    let merged = index.scoped(rect, 1.0, 2.0, false, ScopedMergeRules::default());
    let merged_blocks = merged.blocks();
    assert_eq!(merged_blocks.len(), 1, "expected merge at threshold");
    assert_eq!(merged_blocks[0].text, "word1 word2");
    assert_bbox_close(&merged_blocks[0].bbox, &[10.0, 10.0, 20.0, 12.0], 0.0001);
}

#[test]
fn scoped_merge_threshold_does_not_merge_across_lines() {
    let rect = Rectangle {
        top: 0.0,
        left: 0.0,
        bottom: 40.0,
        right: 40.0,
    };
    let blocks = vec![
        TextBlock {
            page_index: 0,
            text: "line1".to_string(),
            bbox: Rectangle {
                top: 10.0,
                left: 10.0,
                bottom: 12.0,
                right: 14.0,
            },
        },
        TextBlock {
            page_index: 0,
            text: "line2".to_string(),
            bbox: Rectangle {
                top: 20.0,
                left: 14.5,
                bottom: 22.0,
                right: 18.5,
            },
        },
    ];
    let index = TextBlockIndex::new(blocks);
    let scoped = index.scoped(rect, 1.0, 100.0, false, ScopedMergeRules::default());
    let merged_blocks = scoped.blocks();
    assert_eq!(merged_blocks.len(), 2, "expected no cross-line merge");
}

#[test]
fn scoped_merge_threshold_merges_transitively() {
    let rect = Rectangle {
        top: 0.0,
        left: 0.0,
        bottom: 40.0,
        right: 40.0,
    };
    let blocks = vec![
        TextBlock {
            page_index: 0,
            text: "A".to_string(),
            bbox: Rectangle {
                top: 10.0,
                left: 10.0,
                bottom: 12.0,
                right: 13.0,
            },
        },
        TextBlock {
            page_index: 0,
            text: "B".to_string(),
            bbox: Rectangle {
                top: 10.0,
                left: 14.0,
                bottom: 12.0,
                right: 17.0,
            },
        },
        TextBlock {
            page_index: 0,
            text: "C".to_string(),
            bbox: Rectangle {
                top: 10.0,
                left: 18.0,
                bottom: 12.0,
                right: 21.0,
            },
        },
    ];
    let index = TextBlockIndex::new(blocks);
    let scoped = index.scoped(rect, 1.0, 1.0, false, ScopedMergeRules::default());
    let merged_blocks = scoped.blocks();
    assert_eq!(merged_blocks.len(), 1, "expected transitive merge");
    assert_eq!(merged_blocks[0].text, "A B C");
    assert_bbox_close(&merged_blocks[0].bbox, &[10.0, 10.0, 21.0, 12.0], 0.0001);
}

#[test]
fn scoped_merge_threshold_merges_transitively_with_unsorted_input() {
    let rect = Rectangle {
        top: 0.0,
        left: 0.0,
        bottom: 40.0,
        right: 40.0,
    };
    let blocks = vec![
        TextBlock {
            page_index: 0,
            text: "C".to_string(),
            bbox: Rectangle {
                top: 10.0,
                left: 18.0,
                bottom: 12.0,
                right: 21.0,
            },
        },
        TextBlock {
            page_index: 0,
            text: "A".to_string(),
            bbox: Rectangle {
                top: 10.0,
                left: 10.0,
                bottom: 12.0,
                right: 13.0,
            },
        },
        TextBlock {
            page_index: 0,
            text: "B".to_string(),
            bbox: Rectangle {
                top: 10.0,
                left: 14.0,
                bottom: 12.0,
                right: 17.0,
            },
        },
    ];
    let index = TextBlockIndex::new(blocks);
    let scoped = index.scoped(rect, 1.0, 1.0, false, ScopedMergeRules::default());
    let merged_blocks = scoped.blocks();
    assert_eq!(
        merged_blocks.len(),
        1,
        "expected transitive merge regardless of input order"
    );
    assert_eq!(merged_blocks[0].text, "A B C");
    assert_bbox_close(&merged_blocks[0].bbox, &[10.0, 10.0, 21.0, 12.0], 0.0001);
}

#[test]
fn scoped_merge_threshold_does_not_merge_across_pages() {
    let rect = Rectangle {
        top: 0.0,
        left: 0.0,
        bottom: 40.0,
        right: 40.0,
    };
    let blocks = vec![
        TextBlock {
            page_index: 0,
            text: "alpha".to_string(),
            bbox: Rectangle {
                top: 10.0,
                left: 10.0,
                bottom: 12.0,
                right: 14.0,
            },
        },
        TextBlock {
            page_index: 1,
            text: "beta".to_string(),
            bbox: Rectangle {
                top: 10.0,
                left: 14.5,
                bottom: 12.0,
                right: 18.5,
            },
        },
    ];
    let index = TextBlockIndex::new(blocks);
    let scoped = index.scoped(rect, 1.0, 100.0, false, ScopedMergeRules::default());
    let merged_blocks = scoped.blocks();
    assert_eq!(merged_blocks.len(), 2, "expected no cross-page merge");
    assert!(
        merged_blocks
            .iter()
            .any(|b| b.page_index == 0 && b.text == "alpha"),
        "missing page 0 block after scoped merge"
    );
    assert!(
        merged_blocks
            .iter()
            .any(|b| b.page_index == 1 && b.text == "beta"),
        "missing page 1 block after scoped merge"
    );
}

#[test]
fn scoped_normalize_aligns_line_and_preserves_without_flag() {
    let rect = Rectangle {
        top: 0.0,
        left: 0.0,
        bottom: 60.0,
        right: 60.0,
    };
    let blocks = vec![
        TextBlock {
            page_index: 0,
            text: "left".to_string(),
            bbox: Rectangle {
                top: 10.0,
                left: 10.0,
                bottom: 14.0,
                right: 14.0,
            },
        },
        TextBlock {
            page_index: 0,
            text: "right".to_string(),
            bbox: Rectangle {
                top: 11.0,
                left: 16.0,
                bottom: 15.0,
                right: 20.0,
            },
        },
    ];
    let index = TextBlockIndex::new(blocks);

    let unnormalized = index
        .scoped(rect, 1.0, 0.0, false, ScopedMergeRules::default())
        .blocks();
    assert_bbox_close(&unnormalized[0].bbox, &[10.0, 10.0, 14.0, 14.0], 0.0001);
    assert_bbox_close(&unnormalized[1].bbox, &[16.0, 11.0, 20.0, 15.0], 0.0001);

    let normalized = index
        .scoped(rect, 1.0, 0.0, true, ScopedMergeRules::default())
        .blocks();
    assert_bbox_close(&normalized[0].bbox, &[10.0, 10.5, 14.0, 14.5], 0.0001);
    assert_bbox_close(&normalized[1].bbox, &[16.0, 10.5, 20.0, 14.5], 0.0001);
}

#[test]
fn scoped_normalize_uses_first_block_anchor_and_stays_page_local() {
    let rect = Rectangle {
        top: 0.0,
        left: 0.0,
        bottom: 80.0,
        right: 80.0,
    };
    let blocks = vec![
        TextBlock {
            page_index: 0,
            text: "A".to_string(),
            bbox: Rectangle {
                top: 10.0,
                left: 10.0,
                bottom: 14.0,
                right: 13.0,
            },
        },
        TextBlock {
            page_index: 0,
            text: "B".to_string(),
            bbox: Rectangle {
                top: 11.0,
                left: 14.0,
                bottom: 15.0,
                right: 17.0,
            },
        },
        TextBlock {
            page_index: 0,
            text: "C".to_string(),
            bbox: Rectangle {
                top: 12.0,
                left: 18.0,
                bottom: 16.0,
                right: 21.0,
            },
        },
        TextBlock {
            page_index: 1,
            text: "P1-left".to_string(),
            bbox: Rectangle {
                top: 30.0,
                left: 10.0,
                bottom: 34.0,
                right: 14.0,
            },
        },
        TextBlock {
            page_index: 1,
            text: "P1-right".to_string(),
            bbox: Rectangle {
                top: 31.0,
                left: 16.0,
                bottom: 35.0,
                right: 20.0,
            },
        },
    ];
    let index = TextBlockIndex::new(blocks);
    let normalized = index
        .scoped(rect, 1.0, 0.0, true, ScopedMergeRules::default())
        .blocks();

    let a = normalized
        .iter()
        .find(|b| b.text == "A")
        .expect("missing A");
    let b = normalized
        .iter()
        .find(|b| b.text == "B")
        .expect("missing B");
    let c = normalized
        .iter()
        .find(|b| b.text == "C")
        .expect("missing C");
    let p1_left = normalized
        .iter()
        .find(|b| b.text == "P1-left")
        .expect("missing P1-left");
    let p1_right = normalized
        .iter()
        .find(|b| b.text == "P1-right")
        .expect("missing P1-right");

    assert_bbox_close(&a.bbox, &[10.0, 10.5, 13.0, 14.5], 0.0001);
    assert_bbox_close(&b.bbox, &[14.0, 10.5, 17.0, 14.5], 0.0001);
    assert_bbox_close(&c.bbox, &[18.0, 12.0, 21.0, 16.0], 0.0001);

    assert_bbox_close(&p1_left.bbox, &[10.0, 30.5, 14.0, 34.5], 0.0001);
    assert_bbox_close(&p1_right.bbox, &[16.0, 30.5, 20.0, 34.5], 0.0001);
}

#[test]
fn scoped_normalize_runs_before_merge_threshold() {
    let rect = Rectangle {
        top: 0.0,
        left: 0.0,
        bottom: 40.0,
        right: 40.0,
    };
    let blocks = vec![
        TextBlock {
            page_index: 0,
            text: "a".to_string(),
            bbox: Rectangle {
                top: 9.5,
                left: 10.0,
                bottom: 15.5,
                right: 12.0,
            },
        },
        TextBlock {
            page_index: 0,
            text: "b".to_string(),
            bbox: Rectangle {
                top: 10.0,
                left: 13.0,
                bottom: 14.0,
                right: 15.0,
            },
        },
        TextBlock {
            page_index: 0,
            text: "c".to_string(),
            bbox: Rectangle {
                top: 12.0,
                left: 16.0,
                bottom: 16.0,
                right: 18.0,
            },
        },
    ];
    let index = TextBlockIndex::new(blocks);

    let no_normalize = index
        .scoped(rect, 1.0, 1.0, false, ScopedMergeRules::default())
        .blocks();
    assert_eq!(
        no_normalize.len(),
        2,
        "expected no merge when same-line overlap is below threshold"
    );

    let normalized_then_merged = index
        .scoped(rect, 1.0, 1.0, true, ScopedMergeRules::default())
        .blocks();
    assert_eq!(
        normalized_then_merged.len(),
        1,
        "expected merge after normalization aligns line boxes"
    );
    assert_eq!(normalized_then_merged[0].text, "a b c");
    assert_bbox_close(
        &normalized_then_merged[0].bbox,
        &[10.0, 10.5, 18.0, 15.166666666666666],
        0.0001,
    );
}

#[test]
fn scoped_no_numeric_pair_merge_blocks_adjacent_numbers() {
    let rect = Rectangle {
        top: 0.0,
        left: 0.0,
        bottom: 30.0,
        right: 40.0,
    };
    let blocks = vec![
        TextBlock {
            page_index: 0,
            text: "-1624.57".to_string(),
            bbox: Rectangle {
                top: 10.0,
                left: 10.0,
                bottom: 12.0,
                right: 16.0,
            },
        },
        TextBlock {
            page_index: 0,
            text: "234.52".to_string(),
            bbox: Rectangle {
                top: 10.0,
                left: 16.5,
                bottom: 12.0,
                right: 22.5,
            },
        },
    ];
    let index = TextBlockIndex::new(blocks);

    let default_scoped = index.scoped(rect, 1.0, 1.0, false, ScopedMergeRules::default());
    assert_eq!(default_scoped.blocks().len(), 1, "expected default merge");

    let blocked = index.scoped(
        rect,
        1.0,
        1.0,
        false,
        ScopedMergeRules {
            no_numeric_pair_merge: true,
            no_date_pair_merge: false,
        },
    );
    let blocked_blocks = blocked.blocks();
    assert_eq!(
        blocked_blocks.len(),
        2,
        "expected numeric pair merge to be blocked"
    );
}

#[test]
fn scoped_no_date_pair_merge_blocks_adjacent_dates() {
    let rect = Rectangle {
        top: 0.0,
        left: 0.0,
        bottom: 30.0,
        right: 50.0,
    };
    let blocks = vec![
        TextBlock {
            page_index: 0,
            text: "01/31/24".to_string(),
            bbox: Rectangle {
                top: 10.0,
                left: 10.0,
                bottom: 12.0,
                right: 18.0,
            },
        },
        TextBlock {
            page_index: 0,
            text: "02/01/24".to_string(),
            bbox: Rectangle {
                top: 10.0,
                left: 18.2,
                bottom: 12.0,
                right: 26.2,
            },
        },
    ];
    let index = TextBlockIndex::new(blocks);

    let blocked = index.scoped(
        rect,
        1.0,
        1.0,
        false,
        ScopedMergeRules {
            no_numeric_pair_merge: false,
            no_date_pair_merge: true,
        },
    );
    let blocked_blocks = blocked.blocks();
    assert_eq!(
        blocked_blocks.len(),
        2,
        "expected date pair merge to be blocked"
    );
}

#[test]
fn scoped_pairwise_rules_still_allow_numeric_with_text_merge() {
    let rect = Rectangle {
        top: 0.0,
        left: 0.0,
        bottom: 30.0,
        right: 40.0,
    };
    let blocks = vec![
        TextBlock {
            page_index: 0,
            text: "$".to_string(),
            bbox: Rectangle {
                top: 10.0,
                left: 10.0,
                bottom: 12.0,
                right: 11.0,
            },
        },
        TextBlock {
            page_index: 0,
            text: "234.52".to_string(),
            bbox: Rectangle {
                top: 10.0,
                left: 11.2,
                bottom: 12.0,
                right: 17.2,
            },
        },
    ];
    let index = TextBlockIndex::new(blocks);
    let merged = index.scoped(
        rect,
        1.0,
        1.0,
        false,
        ScopedMergeRules {
            no_numeric_pair_merge: true,
            no_date_pair_merge: false,
        },
    );
    let merged_blocks = merged.blocks();
    assert_eq!(
        merged_blocks.len(),
        1,
        "expected numeric-text merge to remain"
    );
    assert_eq!(merged_blocks[0].text, "$ 234.52");
}

#[test]
fn scoped_pairwise_rules_keep_transitive_merge_through_text() {
    let rect = Rectangle {
        top: 0.0,
        left: 0.0,
        bottom: 40.0,
        right: 50.0,
    };
    let blocks = vec![
        TextBlock {
            page_index: 0,
            text: "100".to_string(),
            bbox: Rectangle {
                top: 10.0,
                left: 10.0,
                bottom: 12.0,
                right: 13.0,
            },
        },
        TextBlock {
            page_index: 0,
            text: "USD".to_string(),
            bbox: Rectangle {
                top: 10.0,
                left: 13.5,
                bottom: 12.0,
                right: 17.0,
            },
        },
        TextBlock {
            page_index: 0,
            text: "200".to_string(),
            bbox: Rectangle {
                top: 10.0,
                left: 17.5,
                bottom: 12.0,
                right: 20.5,
            },
        },
    ];
    let index = TextBlockIndex::new(blocks);
    let merged = index.scoped(
        rect,
        1.0,
        1.0,
        false,
        ScopedMergeRules {
            no_numeric_pair_merge: true,
            no_date_pair_merge: false,
        },
    );
    let merged_blocks = merged.blocks();
    assert_eq!(merged_blocks.len(), 1, "expected transitive merge via text");
    assert_eq!(merged_blocks[0].text, "100 USD 200");
}
