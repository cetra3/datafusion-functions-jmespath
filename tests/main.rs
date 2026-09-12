mod utils;

use datafusion::arrow::datatypes::DataType;
use utils::{
    create_test_table, run_query, run_query_typed, run_query_typed_on, single_value, LOCATIONS,
};

#[tokio::test]
async fn jmespath_homepage_example() {
    // The worked example from https://jmespath.org — a filter, a pipe, a
    // function call and a multi-hash, none of which the scan can lower, all
    // hanging off a `locations` prefix that it can.
    let expression = "locations[?state == 'WA'].name | sort(@) | {WashingtonCities: join(', ', @)}";
    let sql = format!(
        "select jmespath('{LOCATIONS}', '{}')",
        expression.replace('\'', "''")
    );
    assert_eq!(
        single_value(&sql).await.unwrap(),
        r#"{"WashingtonCities":"Bellevue, Olympia, Seattle"}"#
    );
}

#[tokio::test]
async fn scalar_types_round_trip() {
    for (expression, expected) in [
        ("foo", "abc"),
        ("bar", "null"),
        ("foo[0]", "null"),
        ("length(foo)", "3"),
    ] {
        let sql = format!(r#"select jmespath('{{"foo": "abc"}}', '{expression}')"#);
        assert_eq!(
            single_value(&sql).await.unwrap(),
            expected,
            "for {expression}"
        );
    }
}

#[tokio::test]
async fn preserves_json_types() {
    for (json, expression, expected) in [
        (r#"{"a": "s"}"#, "a", "s"),
        (r#"{"a": 42}"#, "a", "42"),
        (r#"{"a": 4.5}"#, "a", "4.5"),
        (r#"{"a": true}"#, "a", "true"),
        (r#"{"a": null}"#, "a", "null"),
        (r#"{"a": [1, 2]}"#, "a", "[1, 2]"),
        (r#"{"a": {"b": 1}}"#, "a", r#"{"b": 1}"#),
    ] {
        let sql = format!("select jmespath('{json}', '{expression}')");
        assert_eq!(
            single_value(&sql).await.unwrap(),
            expected,
            "for {json} / {expression}"
        );
    }
}

#[tokio::test]
async fn nested_objects_keep_their_original_text() {
    // The scan path slices the raw bytes rather than re-serialising, so the
    // original spacing survives.
    let value = single_value(r#"select jmespath('{"a": {"b":   1}}', 'a')"#)
        .await
        .unwrap();
    assert_eq!(value, r#"{"b":   1}"#);
}

#[tokio::test]
async fn interpreter_results_are_serialised() {
    // Anything the interpreter builds has no original text to slice.
    let value = single_value(r#"select jmespath('{"a": [{"b": 1}, {"b": 2}]}', 'a[*].b')"#)
        .await
        .unwrap();
    assert_eq!(value, "[1,2]");
}

#[tokio::test]
async fn full_scan_path() {
    // Expressions the plan lowers completely, so the interpreter never runs.
    for (expression, expected) in [
        ("@", r#"{"a": {"b": {"c": [10, 20]}}}"#),
        ("a", r#"{"b": {"c": [10, 20]}}"#),
        ("a.b", r#"{"c": [10, 20]}"#),
        ("a.b.c", "[10, 20]"),
        ("a.b.c[0]", "10"),
        ("a.b.c[1]", "20"),
        ("a.b.c[2]", "null"),
        ("a.b.missing", "null"),
        ("a.b.c.nope", "null"),
    ] {
        let sql =
            format!(r#"select jmespath('{{"a": {{"b": {{"c": [10, 20]}}}}}}', '{expression}')"#);
        assert_eq!(
            single_value(&sql).await.unwrap(),
            expected,
            "for {expression}"
        );
    }
}

#[tokio::test]
async fn hybrid_scan_then_interpret() {
    // `a.b.c[0].d` is scanned; `[*].e` is interpreted against just that subtree.
    let sql = "select jmespath(json_data, 'a.b.c[0].d[*].e') from test where name = 'nested'";
    assert_eq!(
        run_query(sql).await.unwrap(),
        vec![vec!["[1,2]".to_string()]]
    );
}

#[tokio::test]
async fn negative_index_falls_back_to_interpreter() {
    let value = single_value(r#"select jmespath('{"a": [1, 2, 3]}', 'a[-1]')"#)
        .await
        .unwrap();
    assert_eq!(value, "3");
}

#[tokio::test]
async fn slices_and_projections() {
    for (expression, expected) in [
        ("a[1:3]", "[2,3]"),
        ("a[::2]", "[1,3,5]"),
        ("a[?@ > `3`]", "[4,5]"),
        ("max(a)", "5"),
        // JMESPath's `sum` is defined to return a number, and returns a float.
        ("sum(a)", "15.0"),
        ("reverse(a)", "[5,4,3,2,1]"),
    ] {
        let sql = format!("select jmespath('{{\"a\": [1, 2, 3, 4, 5]}}', '{expression}')");
        assert_eq!(
            single_value(&sql).await.unwrap(),
            expected,
            "for {expression}"
        );
    }
}

#[tokio::test]
async fn flatten_and_wildcards() {
    let json = r#"{"a": {"x": {"n": 1}, "y": {"n": 2}}, "b": [[1, 2], [3, 4]]}"#;
    for (expression, expected) in [
        ("a.*.n", "[1,2]"),
        ("b[]", "[1,2,3,4]"),
        ("a.*", r#"[{"n":1},{"n":2}]"#),
    ] {
        let sql = format!("select jmespath('{json}', '{expression}')");
        assert_eq!(
            single_value(&sql).await.unwrap(),
            expected,
            "for {expression}"
        );
    }
}

#[tokio::test]
async fn multi_select_and_pipes() {
    let json = r#"{"a": 1, "b": 2, "c": {"d": 3}}"#;
    for (expression, expected) in [
        ("[a, b]", "[1,2]"),
        ("{first: a, second: b}", r#"{"first":1,"second":2}"#),
        ("c | d", "3"),
        ("a || b", "1"),
        ("missing || b", "2"),
        ("keys(@)", r#"["a","b","c"]"#),
    ] {
        let sql = format!("select jmespath('{json}', '{expression}')");
        assert_eq!(
            single_value(&sql).await.unwrap(),
            expected,
            "for {expression}"
        );
    }
}

#[tokio::test]
async fn pipes_stop_projections() {
    // A pipe is what stops a projection from carrying on, so `[0]` after one
    // indexes the filtered list rather than being folded into the scan prefix.
    let json = r#"{"people": [
        {"general": {"id": 100, "name": "a"}},
        {"general": {"id": 101, "name": "b"}},
        {"general": {"id": 100, "name": "c"}}
    ]}"#;
    for (expression, expected) in [
        (
            "people[?general.id==`100`].general | [0]",
            r#"{"id":100,"name":"a"}"#,
        ),
        (
            "people[?general.id==`100`].general | [1]",
            r#"{"id":100,"name":"c"}"#,
        ),
        ("people[?general.id==`100`].general | [0].name", "a"),
        ("people[?general.id==`100`].general | length(@)", "2"),
        ("people[?general.id==`100`].general[].name", r#"["a","c"]"#),
        // Without the pipe, `[0]` is part of the projection, so it applies to
        // each matched element instead of to the list.
        ("people[?general.id==`100`].general[0]", "[]"),
    ] {
        let sql = format!(
            "select jmespath('{}', '{}')",
            json.replace('\'', "''"),
            expression
        );
        assert_eq!(
            single_value(&sql).await.unwrap(),
            expected,
            "for {expression}"
        );
    }
}

#[tokio::test]
async fn a_missing_value_still_reaches_the_expression() {
    // JMESPath calls a missing value `null`, and an expression can turn that
    // into something — so a scan that finds nothing can't just stop at NULL.
    for (json, expression, expected) in [
        (r#"{"a": 1}"#, "!b", "true"),
        (r#"{"a": 1}"#, "!a", "false"),
        // `a` on an array is null, so `!a` is true.
        ("[1, 2]", "!a", "true"),
        (r#"{"a": 1}"#, "to_array(b)", "[null]"),
        (r#"{"a": 1}"#, "type(b)", "null"),
        (r#"{"a": {"b": 1}}"#, "!(a.c)", "true"),
    ] {
        let sql = format!("select jmespath('{json}', '{expression}')");
        assert_eq!(
            single_value(&sql).await.unwrap(),
            expected,
            "for {json} / {expression}"
        );
    }
}

#[tokio::test]
async fn a_missing_value_from_invalid_json_stays_null() {
    // Garbage input is SQL NULL, not a null the expression gets to reinterpret.
    for expression in ["!b", "to_array(b)", "type(b)"] {
        let sql = format!(
            "select jmespath(json_data, '{expression}') from test where name = 'invalid_json'"
        );
        assert_eq!(
            run_query(&sql).await.unwrap(),
            vec![vec!["null".to_string()]],
            "for {expression}"
        );
    }
}

#[tokio::test]
async fn single_argument_functions_scan_their_argument() {
    // `length(a.b.c)` pre-navigates `a.b.c` rather than materialising the
    // whole document; the answers must not move.
    for (expression, expected) in [
        ("length(a.b.c)", "2"),
        ("keys(a.b)", r#"["c"]"#),
        ("sort(a.b.c)", "[10,20]"),
        ("length(a.missing)", "null"),
        ("max(a.b.c)", "20"),
    ] {
        let sql =
            format!(r#"select jmespath('{{"a": {{"b": {{"c": [20, 10]}}}}}}', '{expression}')"#);
        assert_eq!(
            single_value(&sql).await.unwrap(),
            expected,
            "for {expression}"
        );
    }
}

#[tokio::test]
async fn duplicate_keys_take_the_first() {
    // Legal but pathological JSON. Scanning forward stops at the first match,
    // which is what `json_get` does too; a parser that builds a map would keep
    // the last. Counting keys still dedupes, because that side does build a map.
    let json = r#"{"a": {"b": 1, "b": 2, "c": 3}}"#;
    for (expression, expected) in [
        ("a.b", "1"),
        ("length(a)", "2"),
        ("keys(a)", r#"["b","c"]"#),
    ] {
        let sql = format!("select jmespath('{json}', '{expression}')");
        assert_eq!(
            single_value(&sql).await.unwrap(),
            expected,
            "for {expression}"
        );
    }
}

#[tokio::test]
async fn scanned_functions_match_the_interpreter() {
    // `length(@)`, `keys(@)` and `type(@)` are answered from the jiter cursor
    // rather than a materialised tree, so their edge cases need pinning.
    for (json, expression, expected) in [
        // `length` counts code points, not bytes.
        (r#"{"a": "héllo"}"#, "length(a)", "5"),
        (r#"{"a": "🦀"}"#, "length(a)", "1"),
        (r#"{"a": ""}"#, "length(a)", "0"),
        (r#"{"a": []}"#, "length(a)", "0"),
        (r#"{"a": {}}"#, "length(a)", "0"),
        (r#"{"a": [1, [2, 3], {"b": 4}]}"#, "length(a)", "3"),
        // `length` is undefined for numbers, booleans and null.
        (r#"{"a": 1}"#, "length(a)", "null"),
        (r#"{"a": true}"#, "length(a)", "null"),
        (r#"{"a": null}"#, "length(a)", "null"),
        // `keys` comes back sorted, not in document order.
        (
            r#"{"a": {"z": 1, "A": 2, "m": 3}}"#,
            "keys(a)",
            r#"["A","m","z"]"#,
        ),
        (r#"{"a": {}}"#, "keys(a)", "[]"),
        (r#"{"a": [1]}"#, "keys(a)", "null"),
        // `type` names the JSON type.
        (r#"{"a": "s"}"#, "type(a)", "string"),
        (r#"{"a": 1}"#, "type(a)", "number"),
        (r#"{"a": 1.5}"#, "type(a)", "number"),
        (r#"{"a": true}"#, "type(a)", "boolean"),
        (r#"{"a": null}"#, "type(a)", "null"),
        (r#"{"a": []}"#, "type(a)", "array"),
        (r#"{"a": {}}"#, "type(a)", "object"),
    ] {
        let sql = format!(
            "select jmespath('{}', '{expression}')",
            json.replace('\'', "''")
        );
        assert_eq!(
            single_value(&sql).await.unwrap(),
            expected,
            "for {json} / {expression}"
        );
    }
}

#[tokio::test]
async fn invalid_json_is_null() {
    let sql = "select name, jmespath(json_data, 'foo') from test where name = 'invalid_json'";
    assert_eq!(
        run_query(sql).await.unwrap(),
        vec![vec!["invalid_json".to_string(), "null".to_string()]]
    );
}

#[tokio::test]
async fn null_expression_is_null() {
    assert_eq!(
        single_value("select jmespath('{\"a\": 1}', null)")
            .await
            .unwrap(),
        "null"
    );
}

#[tokio::test]
async fn invalid_expression_is_an_error() {
    let err = run_query("select jmespath('{\"a\": 1}', 'a..b')")
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("invalid JMESPath expression"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn wrong_argument_count_is_a_plan_error() {
    let err = run_query("select jmespath('{}')").await.unwrap_err();
    assert!(
        err.to_string()
            .contains("expected 2 arguments but received 1"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn non_string_expression_is_a_plan_error() {
    let err = run_query("select jmespath('{}', 1)").await.unwrap_err();
    assert!(
        err.to_string().contains("at position 2"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn column_of_expressions() {
    // A per-row expression, which defeats the compile-once path but must still
    // give the same answers.
    let sql = "select name, jmespath(json_data, expr) from (
        select name, json_data, 'foo' as expr from test where name = 'object_foo'
        union all
        select name, json_data, 'foo[1]' as expr from test where name = 'object_foo_array'
        union all
        select name, json_data, 'foo.bar' as expr from test where name = 'object_foo_obj'
    ) order by name";
    assert_eq!(
        run_query(sql).await.unwrap(),
        vec![
            vec!["object_foo".to_string(), "abc".to_string()],
            vec!["object_foo_array".to_string(), "2".to_string()],
            vec!["object_foo_obj".to_string(), "1".to_string()],
        ]
    );
}

#[tokio::test]
async fn nested_calls() {
    // The inner call returns a JSON union, which the outer call reads back.
    let sql = r#"select jmespath(jmespath('{"a": {"b": {"c": 7}}}', 'a'), 'b.c')"#;
    assert_eq!(single_value(sql).await.unwrap(), "7");
}

#[tokio::test]
async fn nested_call_into_array() {
    let sql = r#"select jmespath(jmespath('{"a": [{"b": 1}, {"b": 2}]}', 'a'), '[1].b')"#;
    assert_eq!(single_value(sql).await.unwrap(), "2");
}

#[tokio::test]
async fn works_across_string_types() {
    for json_type in [DataType::Utf8, DataType::LargeUtf8, DataType::Utf8View] {
        let ctx = create_test_table(&json_type).unwrap();
        let sql = "select jmespath(json_data, 'foo') from test where name = 'object_foo'";
        let (_, rows) = run_query_typed_on(&ctx, sql).await.unwrap();
        assert_eq!(rows, vec![vec!["abc".to_string()]], "for {json_type}");
    }
}

#[tokio::test]
async fn dictionary_input_keeps_dictionary_output() {
    let dict_type = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
    let ctx = create_test_table(&dict_type).unwrap();
    let sql = "select name, jmespath(json_data, 'foo') from test order by name";
    let (type_name, rows) = run_query_typed_on(&ctx, sql).await.unwrap();
    assert!(
        type_name.starts_with("Dictionary"),
        "unexpected type: {type_name}"
    );
    assert_eq!(
        rows,
        vec![
            vec!["invalid_json".to_string(), "null".to_string()],
            vec!["list_foo".to_string(), "null".to_string()],
            vec!["nested".to_string(), "null".to_string()],
            vec!["object_bar".to_string(), "null".to_string()],
            vec!["object_foo".to_string(), "abc".to_string()],
            vec!["object_foo_array".to_string(), "[1, 2, 3]".to_string()],
            vec!["object_foo_null".to_string(), "null".to_string()],
            vec!["object_foo_obj".to_string(), r#"{"bar": 1}"#.to_string()],
        ]
    );
}

#[tokio::test]
async fn returns_a_json_union() {
    let (type_name, _) = run_query_typed("select jmespath('{\"a\": 1}', 'a')")
        .await
        .unwrap();
    assert!(
        type_name.starts_with("Union"),
        "unexpected type: {type_name}"
    );
}

#[tokio::test]
async fn deeply_nested_json_does_not_overflow_on_the_scan_path() {
    // The scan path never builds a tree of its own, but it does ask jiter to
    // skip over one; jiter caps its own recursion, so this is an error, not a
    // crash.
    let json = format!("{{\"a\": {}1{}}}", "[".repeat(500), "]".repeat(500));
    let sql = format!("select jmespath('{json}', 'a')");
    assert_eq!(single_value(&sql).await.unwrap(), "null");

    // Shallow enough for jiter, and returned as raw bytes without recursing.
    let json = format!("{{\"a\": {}1{}}}", "[".repeat(10), "]".repeat(10));
    let sql = format!("select jmespath('{json}', 'a')");
    assert_eq!(
        single_value(&sql).await.unwrap(),
        format!("{}1{}", "[".repeat(10), "]".repeat(10))
    );
}

#[tokio::test]
async fn deeply_nested_json_does_not_overflow() {
    // The interpreter path materialises a tree, so it needs a depth guard.
    let json = format!("{}1{}", "[".repeat(500), "]".repeat(500));
    let sql = format!("select jmespath('{json}', '[0][?@]')");
    // Too deep to materialise: NULL rather than a stack overflow.
    assert_eq!(single_value(&sql).await.unwrap(), "null");
}
