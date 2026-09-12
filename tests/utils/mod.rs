#![allow(dead_code)]
use std::sync::Arc;

use datafusion::arrow::array::{
    Array, ArrayRef, AsArray, DictionaryArray, Int32Array, LargeStringArray, StringArray,
    StringViewArray, UnionArray,
};
use datafusion::arrow::datatypes::{DataType, Field, Int32Type, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::arrow::util::display::{ArrayFormatter, FormatOptions};
use datafusion::error::Result;
use datafusion::execution::context::SessionContext;
use datafusion::prelude::SessionConfig;
use datafusion_functions_jmespath::{register_all, JsonUnionEncoder, JsonUnionValue};

/// The example from the JMESPath homepage.
pub const LOCATIONS: &str = r#"{
  "locations": [
    {"name": "Seattle", "state": "WA"},
    {"name": "New York", "state": "NY"},
    {"name": "Bellevue", "state": "WA"},
    {"name": "Olympia", "state": "WA"}
  ]
}"#;

/// `(name, json)` pairs loaded into the `test` table.
pub const TEST_DATA: [(&str, &str); 8] = [
    ("object_foo", r#"{"foo": "abc"}"#),
    ("object_foo_array", r#"{"foo": [1, 2, 3]}"#),
    ("object_foo_obj", r#"{"foo": {"bar": 1}}"#),
    ("object_foo_null", r#"{"foo": null}"#),
    ("object_bar", r#"{"bar": true}"#),
    ("list_foo", r#"["foo", "bar"]"#),
    (
        "nested",
        r#"{"a": {"b": {"c": [{"d": [{"e": 1}, {"e": 2}]}]}}}"#,
    ),
    ("invalid_json", "is not json"),
];

pub fn create_context() -> Result<SessionContext> {
    let config = SessionConfig::new().set_str("datafusion.sql_parser.dialect", "postgres");
    let mut ctx = SessionContext::new_with_config(config);
    register_all(&mut ctx)?;
    Ok(ctx)
}

/// Build a context with a `test` table whose `json_data` column has the given type.
pub fn create_test_table(json_data_type: &DataType) -> Result<SessionContext> {
    let ctx = create_context()?;
    let values = TEST_DATA.iter().map(|(_, json)| *json);

    let json_array: ArrayRef = match json_data_type {
        DataType::Utf8 => Arc::new(StringArray::from_iter_values(values)),
        DataType::LargeUtf8 => Arc::new(LargeStringArray::from_iter_values(values)),
        DataType::Utf8View => Arc::new(StringViewArray::from_iter_values(values)),
        DataType::Dictionary(_, child) if child.as_ref() == &DataType::Utf8 => {
            Arc::new(DictionaryArray::<Int32Type>::new(
                Int32Array::from_iter_values(0..i32::try_from(TEST_DATA.len()).unwrap()),
                Arc::new(StringArray::from_iter_values(values)),
            ))
        }
        other => panic!("unsupported JSON data type: {other}"),
    };

    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8View, false),
            Field::new("json_data", json_data_type.clone(), false),
        ])),
        vec![
            Arc::new(StringViewArray::from(
                TEST_DATA.iter().map(|(name, _)| *name).collect::<Vec<_>>(),
            )),
            json_array,
        ],
    )?;
    ctx.register_batch("test", batch)?;
    Ok(ctx)
}

/// Run `sql` and render the result as `(type, value)` rows of display strings.
pub async fn run_query_typed(sql: &str) -> Result<(String, Vec<Vec<String>>)> {
    let ctx = create_test_table(&DataType::Utf8)?;
    run_query_typed_on(&ctx, sql).await
}

pub async fn run_query_typed_on(
    ctx: &SessionContext,
    sql: &str,
) -> Result<(String, Vec<Vec<String>>)> {
    let batches = ctx.sql(sql).await?.collect().await?;
    let schema = batches
        .first()
        .map(RecordBatch::schema)
        .expect("query returned no batches");
    let type_name = schema
        .field(schema.fields().len() - 1)
        .data_type()
        .to_string();

    let mut rows = Vec::new();
    for batch in &batches {
        let columns = batch
            .columns()
            .iter()
            .map(render_column)
            .collect::<Result<Vec<_>>>()?;
        for row in 0..batch.num_rows() {
            rows.push(columns.iter().map(|c| c[row].clone()).collect());
        }
    }
    Ok((type_name, rows))
}

/// Render a column as display strings, unwrapping the JSON union so tests can
/// assert on the values rather than on the union's debug shape.
fn render_column(column: &ArrayRef) -> Result<Vec<String>> {
    if let Some(union) = column.as_any().downcast_ref::<UnionArray>() {
        let encoder = JsonUnionEncoder::from_union(union.clone()).expect("a JSON union");
        return Ok((0..encoder.len())
            .map(|i| render_union_value(&encoder.get_value(i)))
            .collect());
    }

    if let DataType::Dictionary(_, value_type) = column.data_type() {
        if matches!(value_type.as_ref(), DataType::Union(_, _)) {
            let dict = column.as_any_dictionary();
            let values = render_column(dict.values())?;
            let keys = dict.normalized_keys();
            return Ok((0..column.len())
                .map(|i| {
                    if column.is_null(i) {
                        "null".to_string()
                    } else {
                        values[keys[i]].clone()
                    }
                })
                .collect());
        }
    }

    let options = FormatOptions::default().with_null("null");
    let formatter = ArrayFormatter::try_new(column.as_ref(), &options)?;
    Ok((0..column.len())
        .map(|i| formatter.value(i).to_string())
        .collect())
}

fn render_union_value(value: &JsonUnionValue) -> String {
    match value {
        JsonUnionValue::JsonNull => "null".to_string(),
        JsonUnionValue::Bool(b) => b.to_string(),
        JsonUnionValue::Int(i) => i.to_string(),
        // Format floats the way serde_json does, so `15.0` doesn't render as
        // `15` and read as an integer.
        JsonUnionValue::Float(f) => {
            serde_json::Number::from_f64(*f).map_or_else(|| f.to_string(), |n| n.to_string())
        }
        JsonUnionValue::Str(s) | JsonUnionValue::Array(s) | JsonUnionValue::Object(s) => {
            (*s).to_string()
        }
    }
}

/// Run `sql` against the default `Utf8` table and return the rendered rows.
pub async fn run_query(sql: &str) -> Result<Vec<Vec<String>>> {
    run_query_typed(sql).await.map(|(_, rows)| rows)
}

/// Run `sql` and return the single value it produces.
pub async fn single_value(sql: &str) -> Result<String> {
    let rows = run_query(sql).await?;
    assert_eq!(rows.len(), 1, "expected exactly one row from: {sql}");
    assert_eq!(rows[0].len(), 1, "expected exactly one column from: {sql}");
    Ok(rows[0][0].clone())
}
