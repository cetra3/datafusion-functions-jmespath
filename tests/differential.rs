//! Differential test for the scan/interpret split.
//!
//! `jmespath()` doesn't hand whole documents to JMESPath's interpreter: it
//! peels a literal prefix off the expression, walks that with jiter, and only
//! interprets what's left against the subtree it landed on. That's only sound
//! if it produces exactly what the unmodified interpreter would have.
//!
//! So run both, over a corpus chosen to cover every way the peeling can stall,
//! and compare.

mod utils;

use datafusion::arrow::datatypes::DataType;
use utils::{create_test_table, run_query_typed_on, LOCATIONS};

const DOCUMENTS: &[&str] = &[
    r#"{"a": {"b": {"c": [10, 20, 30]}}}"#,
    r#"{"a": [{"b": 1}, {"b": 2}, {"c": 3}]}"#,
    r#"{"a": {"x": {"n": 1}, "y": {"n": 2}}, "b": [[1, 2], [3, 4]]}"#,
    r#"{"a": 1, "b": 2, "c": {"d": 3}}"#,
    r#"{"a": [1, 2, 3, 4, 5]}"#,
    r#"{"a": null, "b": false, "c": "", "d": [], "e": {}}"#,
    r#"{"a": {"b": {"c": {"d": {"e": {"f": "deep"}}}}}}"#,
    r#"[{"a": 1}, {"a": 2}]"#,
    r#""just a string""#,
    // code points vs bytes for `length`, and duplicate keys for the map the
    // interpreter would have built
    r#"{"a": "h\u00e9llo \ud83e\udd80"}"#,
    r#"{"a": {"z": 1, "A": 2, "m": 3}}"#,
    "42",
    "null",
    LOCATIONS,
];

const EXPRESSIONS: &[&str] = &[
    // Fully lowered to a jiter scan.
    "@",
    "a",
    "a.b",
    "a.b.c",
    "a.b.c[0]",
    "a.b.c[2]",
    "a.b.c[9]",
    "a.b.c.d.e.f",
    "a.missing.b",
    "[0]",
    "[0].a",
    "[9].a",
    "a.b | c",
    // Prefix scanned, remainder interpreted.
    "a[*]",
    "a[*].b",
    "a[].b",
    "a.*",
    "a.*.n",
    "a[-1]",
    "a[1:3]",
    "a[::2]",
    "a[?b]",
    "a[?@ > `2`]",
    "b[]",
    "a.b.c[*]",
    "a.b.c[-1]",
    "locations[?state == 'WA'].name",
    "locations[?state == 'WA'].name | sort(@)",
    "locations[?state == 'WA'].name | sort(@) | {WashingtonCities: join(', ', @)}",
    // A pipe stops the projection, so the trailing index applies to the list.
    "locations[?state == 'WA'] | [0]",
    "locations[?state == 'WA'].name | [0]",
    "locations[?state == 'WA'] | [0].name",
    "locations[?state == 'WA'] | length(@)",
    "a[?b] | [0]",
    "a[*].b | [1]",
    "a | b | c",
    // Nothing to lower at all.
    "length(a)",
    "keys(@)",
    "values(@)",
    "[a, b]",
    "{x: a, y: b}",
    "a || b",
    "a && b",
    "!a",
    "to_string(a)",
    "sort(a)",
    "max(a)",
    "sum(a)",
    "reverse(a)",
    "not_null(a, b, c)",
    // Expressions that turn a missing value into a present one, so a failed
    // scan can't just report NULL and stop.
    "!a",
    "!a.b",
    "!(a.b)",
    "!missing",
    "to_array(a)",
    "to_array(a.b)",
    "type(a)",
    "type(a.missing)",
    "length(a.b)",
    "length(a)",
    "length(@)",
    "length(a.b.c)",
    "keys(a.b)",
    "keys(a)",
    "keys(@)",
    "type(@)",
    "type(a.b)",
    "sort(a.b.c)",
    "not_null(a.missing, b)",
    "`\"literal\"`",
];

/// What the UDF produced, rendered the same way for both sides.
async fn via_udf(json: &str, expression: &str) -> String {
    let ctx = create_test_table(&DataType::Utf8).unwrap();
    let sql = format!(
        "select jmespath('{}', '{}')",
        json.replace('\'', "''"),
        expression.replace('\'', "''")
    );
    let (_, rows) = run_query_typed_on(&ctx, &sql).await.unwrap();
    rows[0][0].clone()
}

/// What JMESPath's own interpreter produces from a fully materialised document.
fn via_jmespath(json: &str, expression: &str) -> String {
    let value: serde_json::Value =
        serde_json::from_str(json).expect("test documents are valid JSON");
    let compiled = jmespath::compile(expression).expect("test expressions are valid JMESPath");

    // A runtime type error — `sort(@)` on a null, say — is NULL to SQL, which is
    // how the UDF reports it too.
    let Ok(result) = compiled.search(value) else {
        return "null".to_string();
    };

    // Match how the UDF's union renders: strings bare, everything else as JSON.
    match &*result {
        jmespath::Variable::String(s) => s.clone(),
        other => serde_json::to_string(other).expect("a searchable result is serialisable"),
    }
}

#[tokio::test]
async fn scan_split_matches_the_interpreter() {
    for expression in EXPRESSIONS {
        for json in DOCUMENTS {
            let expected = via_jmespath(json, expression);
            let actual = via_udf(json, expression).await;

            // The UDF slices nested results out of the original bytes instead of
            // re-serialising them, so compare those as parsed JSON.
            let equal = if actual == expected {
                true
            } else {
                match (
                    serde_json::from_str::<serde_json::Value>(&actual),
                    serde_json::from_str::<serde_json::Value>(&expected),
                ) {
                    (Ok(a), Ok(b)) => a == b,
                    _ => false,
                }
            };

            assert!(
                equal,
                "'{expression}' on {json}\n  udf:       {actual}\n  jmespath:  {expected}"
            );
        }
    }
}
