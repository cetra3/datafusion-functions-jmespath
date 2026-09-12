//! JMESPath functions for DataFusion.
//!
//! Provides a single scalar UDF, [`functions::jmespath`], which queries a JSON
//! string with a [JMESPath](https://jmespath.org) expression:
//!
//! ```sql
//! select jmespath(payload, 'locations[?state == ''WA''].name | sort(@)') from events;
//! ```
//!
//! JMESPath supplies the query language; jiter does the reading. Every
//! expression is split into a literal prefix that jiter walks straight over the
//! raw JSON bytes and a residual expression for JMESPath's interpreter — see
//! [`plan`] for how, and why it's worth it.
//!
//! Results keep their JSON type via a sparse union, so a string comes back as a
//! string and an object comes back as JSON text tagged as such.

use datafusion::common::Result;
use datafusion::execution::FunctionRegistry;

mod common;
mod common_union;
mod jmespath_udf;
mod plan;
mod scan;

pub use common_union::{
    json_field_metadata, JsonUnionEncoder, JsonUnionValue, JSON_UNION_DATA_TYPE,
};

pub mod functions {
    pub use crate::jmespath_udf::jmespath;
}

pub mod udfs {
    pub use crate::jmespath_udf::jmespath_udf;
}

/// Register the JMESPath UDFs with the provided [`FunctionRegistry`].
///
/// # Errors
///
/// Returns an error if the UDFs cannot be registered.
pub fn register_all(registry: &mut dyn FunctionRegistry) -> Result<()> {
    registry.register_udf(jmespath_udf::jmespath_udf())?;
    Ok(())
}
