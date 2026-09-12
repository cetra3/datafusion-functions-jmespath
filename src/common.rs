//! Argument handling: getting from DataFusion's columnar arguments to
//! `(json, plan)` pairs, and back to an array.

use std::sync::Arc;

use datafusion::arrow::array::{
    downcast_array, Array, ArrayAccessor, ArrayRef, AsArray, DictionaryArray, PrimitiveArray,
    PrimitiveBuilder, StringArray, UnionArray,
};
use datafusion::arrow::compute::kernels::cast;
use datafusion::arrow::datatypes::{ArrowNativeType, DataType, Int64Type};
use datafusion::common::{exec_err, plan_err, Result as DataFusionResult, ScalarValue};
use datafusion::logical_expr::ColumnarValue;

use crate::common_union::{
    is_json_union, json_from_union_scalar, nested_json_array, JsonUnion, JsonUnionField,
    TYPE_ID_NULL,
};
use crate::plan::{cached_plan, ScanPlan};
use crate::scan::eval;

/// Validate the argument types of `jmespath(json, expression)` and work out the
/// return type.
///
/// # Errors
///
/// Returns an error if the arguments aren't a JSON-ish value and a string.
pub(crate) fn return_type_check(args: &[DataType], fn_name: &str) -> DataFusionResult<DataType> {
    let [json_type, expr_type] = args else {
        return plan_err!(
            "The '{fn_name}' function takes two arguments: a JSON value and a JMESPath expression."
        );
    };

    if !(is_str(json_type)
        || is_json_union(json_type)
        || dict_of_json(json_type)
        || json_type.is_null())
    {
        return plan_err!(
            "Unexpected argument type to '{fn_name}' at position 1, expected a string, got {json_type:?}."
        );
    }
    if !(is_str(expr_type) || dict_of_str(expr_type) || expr_type.is_null()) {
        return plan_err!(
            "Unexpected argument type to '{fn_name}' at position 2, expected a JMESPath expression string, got {expr_type:?}."
        );
    }

    if dict_of_json(json_type) {
        // Keep the dictionary encoding of the input, so a column with few
        // distinct documents stays cheap downstream.
        Ok(DataType::Dictionary(
            Box::new(DataType::Int64),
            Box::new(JsonUnion::data_type()),
        ))
    } else {
        Ok(JsonUnion::data_type())
    }
}

fn is_str(d: &DataType) -> bool {
    matches!(d, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View)
}

fn dict_of_str(d: &DataType) -> bool {
    matches!(d, DataType::Dictionary(_, value) if is_str(value))
}

fn dict_of_json(d: &DataType) -> bool {
    matches!(d, DataType::Dictionary(_, value) if is_str(value) || is_json_union(value))
}

/// Evaluate `jmespath(json, expression)` over its columnar arguments.
///
/// # Errors
///
/// Returns an error if the arguments have unexpected types, or if a JMESPath
/// expression fails to parse.
pub(crate) fn invoke(args: &[ColumnarValue]) -> DataFusionResult<ColumnarValue> {
    let [json_arg, expr_arg] = args else {
        return exec_err!("expected two arguments: a JSON value and a JMESPath expression");
    };

    match (json_arg, expr_arg) {
        // The overwhelmingly common shape: one literal expression applied down
        // a column, so the expression is parsed exactly once per batch.
        (ColumnarValue::Array(json_array), ColumnarValue::Scalar(expr)) => {
            let plan = scalar_plan(expr)?;
            json_array_scalar_expr(json_array, plan.as_deref()).map(ColumnarValue::Array)
        }
        (ColumnarValue::Scalar(json), ColumnarValue::Scalar(expr)) => {
            let plan = scalar_plan(expr)?;
            let value = match plan {
                Some(plan) => eval(extract_json_scalar(json)?, &plan).ok(),
                None => None,
            };
            Ok(ColumnarValue::Scalar(JsonUnionField::scalar_value(value)))
        }
        (ColumnarValue::Array(json_array), ColumnarValue::Array(expr_array)) => {
            zip_apply(json_array, expr_array).map(ColumnarValue::Array)
        }
        (ColumnarValue::Scalar(json), ColumnarValue::Array(expr_array)) => {
            let json = extract_json_scalar(json)?.map(ToOwned::to_owned);
            let broadcast = match &json {
                Some(json) => StringArray::from_iter_values(std::iter::repeat_n(
                    json.as_str(),
                    expr_array.len(),
                )),
                None => StringArray::new_null(expr_array.len()),
            };
            zip_apply(&(Arc::new(broadcast) as ArrayRef), expr_array).map(ColumnarValue::Array)
        }
    }
}

/// Parse the plan for a scalar expression argument. `None` means the expression
/// was NULL, so every row is NULL.
fn scalar_plan(expr: &ScalarValue) -> DataFusionResult<Option<Arc<ScanPlan>>> {
    match extract_str_scalar(expr)? {
        Some(expression) => cached_plan(expression).map(Some),
        None => Ok(None),
    }
}

/// Apply a single plan down a JSON column.
fn json_array_scalar_expr(
    json_array: &ArrayRef,
    plan: Option<&ScanPlan>,
) -> DataFusionResult<ArrayRef> {
    #[allow(clippy::needless_pass_by_value)] // ArrayAccessor is implemented on references
    fn inner<'j>(
        json_array: impl ArrayAccessor<Item = &'j str>,
        plan: Option<&ScanPlan>,
    ) -> ArrayRef {
        let mut union = JsonUnion::new(json_array.len());
        for i in 0..json_array.len() {
            let json = if json_array.is_null(i) {
                None
            } else {
                Some(json_array.value(i))
            };
            match plan.and_then(|plan| eval(json, plan).ok()) {
                Some(field) => union.push(field),
                None => union.push_none(),
            }
        }
        finish(union)
    }

    match json_array.data_type() {
        DataType::Dictionary(_, _) => {
            // Evaluate against the dictionary values only — one evaluation per
            // distinct document rather than per row.
            let dict = json_array.as_any_dictionary();
            let values = json_array_scalar_expr(dict.values(), plan)?;
            let keys = downcast_array(&cast(dict.keys(), &DataType::Int64)?);
            let keys = mask_dictionary_keys(&keys, values.as_union().type_ids());
            Ok(Arc::new(remap_dictionary_key_nulls(keys, values)))
        }
        DataType::Utf8 => Ok(inner(json_array.as_string::<i32>(), plan)),
        DataType::LargeUtf8 => Ok(inner(json_array.as_string::<i64>(), plan)),
        DataType::Utf8View => Ok(inner(json_array.as_string_view(), plan)),
        other => {
            // A JSON union from a nested call, e.g. jmespath(jmespath(x, 'a'), 'b').
            let object_lookup = plan.is_none_or(ScanPlan::is_object_lookup);
            if let Some(string_array) = nested_json_array(json_array, object_lookup) {
                Ok(inner(string_array, plan))
            } else {
                exec_err!("unexpected JSON argument type {other:?}")
            }
        }
    }
}

/// Apply a per-row expression down a JSON column.
fn zip_apply(json_array: &ArrayRef, expr_array: &ArrayRef) -> DataFusionResult<ArrayRef> {
    #[allow(clippy::needless_pass_by_value)] // ArrayAccessor is implemented on references
    fn inner<'j, 'e>(
        json_array: impl ArrayAccessor<Item = &'j str>,
        expr_array: impl ArrayAccessor<Item = &'e str>,
    ) -> DataFusionResult<ArrayRef> {
        let mut union = JsonUnion::new(json_array.len());
        for i in 0..json_array.len() {
            if expr_array.is_null(i) {
                union.push_none();
                continue;
            }
            let plan = cached_plan(expr_array.value(i))?;
            let json = if json_array.is_null(i) {
                None
            } else {
                Some(json_array.value(i))
            };
            match eval(json, &plan).ok() {
                Some(field) => union.push(field),
                None => union.push_none(),
            }
        }
        Ok(finish(union))
    }

    /// Resolve the expression array, then dispatch on the JSON array.
    fn with_expr<'e>(
        json_array: &ArrayRef,
        expr_array: impl ArrayAccessor<Item = &'e str> + Copy,
    ) -> DataFusionResult<ArrayRef> {
        match json_array.data_type() {
            DataType::Utf8 => inner(json_array.as_string::<i32>(), expr_array),
            DataType::LargeUtf8 => inner(json_array.as_string::<i64>(), expr_array),
            DataType::Utf8View => inner(json_array.as_string_view(), expr_array),
            // Per-row expressions defeat the dictionary short-circuit, so just
            // unpack to the values each row points at.
            DataType::Dictionary(_, _) => {
                let unpacked = cast(json_array, &unpacked_dict_type(json_array.data_type()))?;
                with_expr(&unpacked, expr_array)
            }
            other => {
                // A per-row expression could want either union member, and we
                // can only read one, so assume the common object case.
                if let Some(string_array) = nested_json_array(json_array, true) {
                    inner(string_array, expr_array)
                } else {
                    exec_err!("unexpected JSON argument type {other:?}")
                }
            }
        }
    }

    match expr_array.data_type() {
        DataType::Utf8 => with_expr(json_array, expr_array.as_string::<i32>()),
        DataType::LargeUtf8 => with_expr(json_array, expr_array.as_string::<i64>()),
        DataType::Utf8View => with_expr(json_array, expr_array.as_string_view()),
        DataType::Dictionary(_, value_type) => {
            let flattened = cast(expr_array, value_type)?;
            zip_apply(json_array, &flattened)
        }
        other => {
            exec_err!("unexpected expression argument type, expected a string array, got {other:?}")
        }
    }
}

/// The type a dictionary array unpacks to.
fn unpacked_dict_type(data_type: &DataType) -> DataType {
    match data_type {
        DataType::Dictionary(_, value) => value.as_ref().clone(),
        other => other.clone(),
    }
}

fn finish(union: JsonUnion) -> ArrayRef {
    let array: UnionArray = union.try_into().expect("JsonUnion is always well formed");
    Arc::new(array)
}

fn extract_json_scalar(scalar: &ScalarValue) -> DataFusionResult<Option<&str>> {
    match scalar {
        ScalarValue::Dictionary(_, inner) => extract_json_scalar(inner.as_ref()),
        ScalarValue::Utf8(s) | ScalarValue::Utf8View(s) | ScalarValue::LargeUtf8(s) => {
            Ok(s.as_deref())
        }
        ScalarValue::Null => Ok(None),
        ScalarValue::Union(type_id_value, union_fields, _) => {
            Ok(json_from_union_scalar(type_id_value.as_ref(), union_fields))
        }
        _ => exec_err!("unexpected first argument type, expected a string or JSON union"),
    }
}

fn extract_str_scalar(scalar: &ScalarValue) -> DataFusionResult<Option<&str>> {
    match scalar {
        ScalarValue::Dictionary(_, inner) => extract_str_scalar(inner.as_ref()),
        ScalarValue::Utf8(s) | ScalarValue::Utf8View(s) | ScalarValue::LargeUtf8(s) => {
            Ok(s.as_deref())
        }
        ScalarValue::Null => Ok(None),
        _ => exec_err!("unexpected second argument type, expected a JMESPath expression string"),
    }
}

/// Set keys to null where the union member they point at is null.
///
/// Arrow is happiest expressing dictionary nulls through the keys; see
/// <https://github.com/apache/arrow-rs/issues/6017#issuecomment-2352756753>.
fn mask_dictionary_keys(
    keys: &PrimitiveArray<Int64Type>,
    type_ids: &[i8],
) -> PrimitiveArray<Int64Type> {
    let mut null_mask = vec![true; keys.len()];
    for (i, key) in keys.iter().enumerate() {
        match key {
            Some(k)
                if type_ids.get(k.as_usize()).copied().unwrap_or(TYPE_ID_NULL) != TYPE_ID_NULL => {}
            _ => null_mask[i] = false,
        }
    }
    PrimitiveArray::new(keys.values().clone(), Some(null_mask.into()))
}

/// Move nulls out of dictionary values and into the keys.
///
/// Arrow and DataFusion assume dictionary values contain no nulls; breaking
/// that invariant produces invalid arrays once batches get concatenated.
fn remap_dictionary_key_nulls(
    keys: PrimitiveArray<Int64Type>,
    values: ArrayRef,
) -> DictionaryArray<Int64Type> {
    if values.null_count() == 0 {
        return DictionaryArray::new(keys, values);
    }

    let length = i64::try_from(values.len()).unwrap_or(i64::MAX);
    let mut builder = PrimitiveBuilder::<Int64Type>::new();
    for key in &keys {
        match key {
            // Keys can point past the values after a slice, take or concat.
            Some(k) if k < length && !values.is_null(k.as_usize()) => builder.append_value(k),
            _ => builder.append_null(),
        }
    }
    DictionaryArray::new(builder.finish(), values)
}
