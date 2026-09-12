//! The `jmespath()` scalar UDF.

use std::sync::{Arc, OnceLock};

use datafusion::arrow::datatypes::DataType;
use datafusion::common::Result as DataFusionResult;
use datafusion::logical_expr::expr::ScalarFunction;
use datafusion::logical_expr::{
    ColumnarValue, Expr, ExpressionPlacement, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl,
    Signature, Volatility,
};

use crate::common::{invoke, return_type_check};

/// Query a JSON string with a JMESPath expression.
#[must_use]
pub fn jmespath(json_data: Expr, expression: Expr) -> Expr {
    Expr::ScalarFunction(ScalarFunction::new_udf(
        jmespath_udf(),
        vec![json_data, expression],
    ))
}

/// The [`ScalarUDF`] for [`jmespath`], built once and shared.
pub fn jmespath_udf() -> Arc<ScalarUDF> {
    static UDF: OnceLock<Arc<ScalarUDF>> = OnceLock::new();
    UDF.get_or_init(|| Arc::new(ScalarUDF::new_from_impl(JmesPath::default())))
        .clone()
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub(super) struct JmesPath {
    signature: Signature,
    aliases: [String; 1],
}

impl Default for JmesPath {
    fn default() -> Self {
        Self {
            signature: Signature::any(2, Volatility::Immutable),
            aliases: ["jmespath".to_string()],
        }
    }
}

impl ScalarUDFImpl for JmesPath {
    fn name(&self) -> &str {
        self.aliases[0].as_str()
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> DataFusionResult<DataType> {
        return_type_check(arg_types, self.name())
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DataFusionResult<ColumnarValue> {
        invoke(&args.args)
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }

    fn placement(&self, args: &[ExpressionPlacement]) -> ExpressionPlacement {
        // A column and a literal expression can be evaluated at the scan, before
        // anything else has to carry the JSON around.
        if matches!(
            args,
            [ExpressionPlacement::Column, ExpressionPlacement::Literal]
        ) {
            ExpressionPlacement::MoveTowardsLeafNodes
        } else {
            ExpressionPlacement::KeepInPlace
        }
    }
}
