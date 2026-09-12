//! What `jmespath()` costs over a full batch, by expression shape.
//!
//! Expressions that lower entirely to a scan should cost a single jiter walk
//! over the raw bytes: the expression is parsed once and cached, and nothing is
//! materialised. The rest show what the interpreter adds once the scan stops.
//!
//! One known cost worth watching: on a dictionary-encoded column the output
//! path (key cast, null masking, rebuilding the `DictionaryArray`) is about
//! 0.6ns/row. Because the dictionary short-circuit removes nearly all the
//! parsing work, that small absolute cost reads as a large percentage of a very
//! small number.

use std::sync::Arc;

use codspeed_criterion_compat::{
    criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use datafusion::arrow::array::{
    ArrayRef, DictionaryArray, Int32Array, StringArray, StringViewArray,
};
use datafusion::arrow::datatypes::{DataType, Field, Int32Type};
use datafusion::common::ScalarValue;
use datafusion::config::ConfigOptions;
use datafusion::logical_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDF};

/// A full DataFusion batch, so per-row cost dominates the call overhead.
const ROWS: usize = 8192;

/// Distinct documents behind the dictionary-encoded column.
const DISTINCT: usize = 16;

/// A document shaped like something you'd actually store: the interesting
/// fields are neither first nor last, and there's bulk to skip past.
fn document(i: usize) -> String {
    format!(
        r#"{{"id":{i},"padding":[{padding}],"meta":{{"labels":{{"env":"prod","team":"core"}},"tags":["a","b","c"],"ok":true,"ratio":0.75}},"events":[{{"name":"start","n":1}},{{"name":"stop","n":2}}],"deep":{{"a":{{"b":{{"c":{{"d":"leaf"}}}}}}}},"trailing":"{i}"}}"#,
        padding = (0..40).map(|n| n.to_string()).collect::<Vec<_>>().join(",")
    )
}

fn utf8_column() -> ArrayRef {
    Arc::new(StringArray::from_iter_values((0..ROWS).map(document)))
}

fn utf8_view_column() -> ArrayRef {
    Arc::new(StringViewArray::from_iter_values((0..ROWS).map(document)))
}

/// The same documents dictionary-encoded, which is short-circuited by
/// evaluating against the dictionary values rather than once per row.
fn dictionary_column() -> ArrayRef {
    let values = StringArray::from_iter_values((0..DISTINCT).map(document));
    let keys = Int32Array::from_iter_values(
        (0..ROWS).map(|i| i32::try_from(i % DISTINCT).expect("fits in i32")),
    );
    Arc::new(DictionaryArray::<Int32Type>::new(keys, Arc::new(values)))
}

/// Invoke `udf` over `json_array` with `expression` as its second argument.
fn invoke(udf: &ScalarUDF, json_array: &ArrayRef, expression: &ScalarValue) {
    let args = vec![
        ColumnarValue::Array(json_array.clone()),
        ColumnarValue::Scalar(expression.clone()),
    ];
    let arg_fields = vec![
        Arc::new(Field::new("json", json_array.data_type().clone(), false)),
        Arc::new(Field::new("expression", expression.data_type(), false)),
    ];

    let arg_types: Vec<DataType> = arg_fields.iter().map(|f| f.data_type().clone()).collect();
    let return_type = udf.return_type(&arg_types).expect("valid arguments");

    udf.invoke_with_args(ScalarFunctionArgs {
        args,
        number_rows: ROWS,
        arg_fields,
        return_field: Arc::new(Field::new("result", return_type, true)),
        config_options: Arc::new(ConfigOptions::default()),
    })
    .expect("invoke should succeed");
}

/// Benchmark each `(case, expression)` over `json_array` as one group.
fn bench_group(c: &mut Criterion, name: &str, json_array: &ArrayRef, cases: &[(&str, &str)]) {
    let udf = datafusion_functions_jmespath::udfs::jmespath_udf();

    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Elements(ROWS as u64));

    for (case, expression) in cases {
        let expression = ScalarValue::Utf8(Some((*expression).to_string()));
        group.bench_with_input(BenchmarkId::new("jmespath", case), &expression, |b, expression| {
            b.iter(|| invoke(&udf, json_array, expression));
        });
    }
    group.finish();
}

/// Plain paths, which lower entirely to a scan.
const PATHS: &[(&str, &str)] = &[
    ("first_key", "id"),
    ("nested_key", "meta.labels.env"),
    ("deep_key", "deep.a.b.c.d"),
    ("array_index", "events[1]"),
    ("index_then_key", "events[1].name"),
    ("into_padding", "padding[30]"),
    ("returns_object", "meta.labels"),
    ("returns_array", "meta.tags"),
    ("returns_float", "meta.ratio"),
    ("returns_bool", "meta.ok"),
    ("missing_key", "meta.nope"),
    ("last_key", "trailing"),
];

fn bench_paths_utf8(c: &mut Criterion) {
    bench_group(c, "paths_utf8", &utf8_column(), PATHS);
}

fn bench_paths_utf8_view(c: &mut Criterion) {
    bench_group(c, "paths_utf8_view", &utf8_view_column(), PATHS);
}

fn bench_paths_dictionary(c: &mut Criterion) {
    bench_group(c, "paths_dictionary", &dictionary_column(), PATHS);
}

/// Single-argument functions, answered by scanning the lifted argument rather
/// than materialising it.
const FUNCTIONS: &[(&str, &str)] = &[
    ("length_array", "length(meta.tags)"),
    ("length_object", "length(meta.labels)"),
    ("object_keys", "keys(meta.labels)"),
    ("type", "type(meta.labels)"),
];

fn bench_functions(c: &mut Criterion) {
    bench_group(c, "functions", &utf8_column(), FUNCTIONS);
}

/// Shapes that need the interpreter: the scan stops early, or can't start at
/// all when the root is a multi-select.
const INTERPRETED: &[(&str, &str)] = &[
    ("projection", "events[*].name"),
    ("filter", "events[?n > `1`].name"),
    ("negative_index", "events[-1].name"),
    ("slice", "padding[0:5]"),
    ("multi_select", "{env: meta.labels.env, tags: meta.tags}"),
];

fn bench_interpreted(c: &mut Criterion) {
    bench_group(c, "interpreted", &utf8_column(), INTERPRETED);
}

criterion_group!(
    benches,
    bench_paths_utf8,
    bench_paths_utf8_view,
    bench_paths_dictionary,
    bench_functions,
    bench_interpreted
);
criterion_main!(benches);
