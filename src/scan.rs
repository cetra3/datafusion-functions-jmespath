//! Executing a [`ScanPlan`] against raw JSON bytes with jiter.

use std::collections::{BTreeMap, BTreeSet};
use std::str::Utf8Error;

use jiter::{Jiter, JiterError, NumberAny, NumberInt, Peek};
use jmespath::ast::Ast;
use jmespath::{interpret, Context, Rcvar, Variable, DEFAULT_RUNTIME};
use serde_json::Number;

use crate::common_union::JsonUnionField;
use crate::plan::{ScanPlan, Step};

/// Any failure to produce a value: a parse error, a path that doesn't exist, a
/// JMESPath runtime error. They all mean the same thing to SQL — NULL.
pub(crate) struct GetError;

impl From<JiterError> for GetError {
    fn from(_: JiterError) -> Self {
        Self
    }
}

impl From<Utf8Error> for GetError {
    fn from(_: Utf8Error) -> Self {
        Self
    }
}

/// Cap on how deep we'll recurse when materialising a `Variable`, so a
/// pathologically nested document can't overflow the stack.
const MAX_DEPTH: usize = 128;

/// Evaluate `plan` against a JSON document.
pub(crate) fn eval(opt_json: Option<&str>, plan: &ScanPlan) -> Result<JsonUnionField, GetError> {
    let json = opt_json.ok_or(GetError)?;

    let Some((mut jiter, peek)) = scan_prefix(json, plan.prefix()) else {
        // The prefix led nowhere. With nothing else to do that's SQL NULL, but
        // a residual still has to see it: JMESPath calls a missing value `null`,
        // and plenty of expressions turn a null into something — `!missing` is
        // `true`, `to_array(missing)` is `[null]`.
        let Some(residual) = plan.residual() else {
            return Err(GetError);
        };
        // Only for a document that actually parses, though. Garbage in is SQL
        // NULL, not a null the expression gets to reinterpret.
        validate(json)?;
        return interpret_residual(plan, residual, &Rcvar::new(Variable::Null));
    };

    match plan.residual() {
        // The prefix covered the whole expression, so read the value straight
        // out of the JSON bytes — no tree is ever built.
        None => build_union(&mut jiter, peek),
        // A handful of residuals only need to walk the value, not own it, so
        // they can be answered from the cursor the prefix left behind. This is
        // what makes `length(a.b)` cost about what a plain lookup costs.
        Some(residual) => {
            if let Some(direct) = Direct::of(residual) {
                return direct.eval(&mut jiter, peek);
            }
            // Otherwise materialise the subtree the prefix landed on and let
            // the interpreter finish the job.
            let subtree = Rcvar::new(build_variable(&mut jiter, peek, 0)?);
            interpret_residual(plan, residual, &subtree)
        }
    }
}

/// A residual of the form `f(@)` that can be answered by scanning.
///
/// These only exist because `peel` lifts a single-argument function's argument
/// into the prefix, leaving the call rooted at the value itself.
#[derive(Debug, Clone, Copy)]
enum Direct {
    Length,
    Keys,
    Type,
}

impl Direct {
    /// Recognise a residual that doesn't need a materialised tree.
    fn of(residual: &Ast) -> Option<Self> {
        let Ast::Function { name, args, .. } = residual else {
            return None;
        };
        if !matches!(args.as_slice(), [Ast::Identity { .. }]) {
            return None;
        }
        match name.as_str() {
            "length" => Some(Self::Length),
            "keys" => Some(Self::Keys),
            "type" => Some(Self::Type),
            _ => None,
        }
    }

    fn eval(self, jiter: &mut Jiter, peek: Peek) -> Result<JsonUnionField, GetError> {
        match self {
            Self::Length => Self::length(jiter, peek),
            Self::Keys => Self::keys(jiter, peek),
            Self::Type => Self::json_type(jiter, peek),
        }
    }

    fn length(jiter: &mut Jiter, peek: Peek) -> Result<JsonUnionField, GetError> {
        let count = match peek {
            // JMESPath counts code points, not bytes.
            Peek::String => jiter.known_str()?.chars().count(),
            Peek::Array => {
                let mut count = 0;
                let mut element = jiter.known_array()?;
                while let Some(element_peek) = element {
                    jiter.known_skip(element_peek)?;
                    count += 1;
                    element = jiter.array_step()?;
                }
                count
            }
            // The interpreter would have collected the object into a map, so
            // what counts is the number of *distinct* keys.
            Peek::Object => object_keys(jiter)?.len(),
            // `length` is defined only for strings, arrays and objects; anything
            // else is a type error, which this crate reports as NULL.
            _ => return Err(GetError),
        };
        i64::try_from(count)
            .map(JsonUnionField::Int)
            .map_err(|_| GetError)
    }

    fn keys(jiter: &mut Jiter, peek: Peek) -> Result<JsonUnionField, GetError> {
        if peek != Peek::Object {
            // `keys` is defined only for objects.
            return Err(GetError);
        }
        let keys = object_keys(jiter)?;
        serde_json::to_string(&keys)
            .map(JsonUnionField::Array)
            .map_err(|_| GetError)
    }

    fn json_type(jiter: &mut Jiter, peek: Peek) -> Result<JsonUnionField, GetError> {
        let name = match peek {
            Peek::Null => "null",
            Peek::True | Peek::False => "boolean",
            Peek::String => "string",
            Peek::Array => "array",
            Peek::Object => "object",
            _ => "number",
        };
        // Walk the value even though the name came from its first byte, so
        // malformed input is NULL rather than confidently mistyped.
        jiter.known_skip(peek)?;
        Ok(JsonUnionField::Str(name.to_owned()))
    }
}

/// Collect an object's keys the way the interpreter's map would hold them:
/// sorted and deduplicated.
fn object_keys(jiter: &mut Jiter) -> Result<BTreeSet<String>, GetError> {
    let mut keys = BTreeSet::new();
    let mut key = jiter.known_object()?.map(ToOwned::to_owned);
    while let Some(name) = key {
        keys.insert(name);
        jiter.next_skip()?;
        key = jiter.next_key()?.map(ToOwned::to_owned);
    }
    Ok(keys)
}

/// Run the residual expression against the value the prefix landed on.
fn interpret_residual(
    plan: &ScanPlan,
    residual: &Ast,
    data: &Rcvar,
) -> Result<JsonUnionField, GetError> {
    let mut ctx = Context::new(plan.expression(), &DEFAULT_RUNTIME);
    let result = interpret(data, residual, &mut ctx).map_err(|_| GetError)?;
    variable_to_union(&result)
}

/// Check that the whole document is well-formed JSON.
///
/// Only used on the path-not-found branch, to tell "this document has no such
/// field" apart from "this isn't JSON at all".
fn validate(json: &str) -> Result<(), GetError> {
    let mut jiter = Jiter::new(json.as_bytes());
    let peek = jiter.peek()?;
    jiter.known_skip(peek)?;
    jiter.finish()?;
    Ok(())
}

/// Walk `prefix` over the raw JSON, leaving the jiter positioned at the value
/// it names.
///
/// Returns `None` if the path doesn't exist or the document doesn't have the
/// shape the path expects — both of which JMESPath defines as `null`.
fn scan_prefix<'j>(json: &'j str, prefix: &[Step]) -> Option<(Jiter<'j>, Peek)> {
    let mut jiter = Jiter::new(json.as_bytes());
    let mut peek = jiter.peek().ok()?;

    for step in prefix {
        match step {
            Step::Key(key) if peek == Peek::Object => {
                let mut next_key = jiter.known_object().ok()??;
                while next_key != key.as_str() {
                    jiter.next_skip().ok()?;
                    next_key = jiter.next_key().ok()??;
                }
                peek = jiter.peek().ok()?;
            }
            Step::Index(index) if peek == Peek::Array => {
                let mut element = jiter.known_array().ok()??;
                for _ in 0..*index {
                    jiter.known_skip(element).ok()?;
                    element = jiter.array_step().ok()??;
                }
                peek = element;
            }
            // A field access on a non-object, or an index into a non-array.
            _ => return None,
        }
    }

    Some((jiter, peek))
}

/// Read the value the jiter is positioned at into a union field.
///
/// Arrays and objects are taken as a slice of the original bytes rather than
/// re-serialised, so nested results cost one skip and one copy.
fn build_union(jiter: &mut Jiter, peek: Peek) -> Result<JsonUnionField, GetError> {
    match peek {
        Peek::Null => {
            jiter.known_null()?;
            Ok(JsonUnionField::JsonNull)
        }
        Peek::True | Peek::False => Ok(JsonUnionField::Bool(jiter.known_bool(peek)?)),
        Peek::String => Ok(JsonUnionField::Str(jiter.known_str()?.to_owned())),
        Peek::Array => Ok(JsonUnionField::Array(raw_slice(jiter, peek)?)),
        Peek::Object => Ok(JsonUnionField::Object(raw_slice(jiter, peek)?)),
        _ => match jiter.known_number(peek)? {
            NumberAny::Int(NumberInt::Int(value)) => Ok(JsonUnionField::Int(value)),
            NumberAny::Float(value) => Ok(JsonUnionField::Float(value)),
            // The union has no home for an integer wider than i64, so fall back
            // to the float member and accept the loss of precision.
            NumberAny::Int(big @ NumberInt::BigInt(_)) => Ok(JsonUnionField::Float(f64::from(big))),
        },
    }
}

/// Skip over the value the jiter is positioned at and return its raw text.
fn raw_slice(jiter: &mut Jiter, peek: Peek) -> Result<String, GetError> {
    let start = jiter.current_index();
    jiter.known_skip(peek)?;
    Ok(std::str::from_utf8(jiter.slice_to_current(start))?.to_owned())
}

/// Build a JMESPath [`Variable`] from the value the jiter is positioned at.
///
/// This is the fallback path, and it's why jiter still does the parsing even
/// when the interpreter has to run: we go straight from JSON bytes to
/// `Variable` without a `serde_json::Value` in between.
fn build_variable(jiter: &mut Jiter, peek: Peek, depth: usize) -> Result<Variable, GetError> {
    if depth > MAX_DEPTH {
        return Err(GetError);
    }

    match peek {
        Peek::Null => {
            jiter.known_null()?;
            Ok(Variable::Null)
        }
        Peek::True | Peek::False => Ok(Variable::Bool(jiter.known_bool(peek)?)),
        Peek::String => Ok(Variable::String(jiter.known_str()?.to_owned())),
        Peek::Array => {
            let mut elements = Vec::new();
            let mut element = jiter.known_array()?;
            while let Some(element_peek) = element {
                elements.push(Rcvar::new(build_variable(jiter, element_peek, depth + 1)?));
                element = jiter.array_step()?;
            }
            Ok(Variable::Array(elements))
        }
        Peek::Object => {
            let mut entries = BTreeMap::new();
            let mut key = jiter.known_object()?.map(ToOwned::to_owned);
            while let Some(next_key) = key {
                let value_peek = jiter.peek()?;
                let value = build_variable(jiter, value_peek, depth + 1)?;
                entries.insert(next_key, Rcvar::new(value));
                key = jiter.next_key()?.map(ToOwned::to_owned);
            }
            Ok(Variable::Object(entries))
        }
        _ => match jiter.known_number(peek)? {
            NumberAny::Int(NumberInt::Int(value)) => Ok(Variable::Number(Number::from(value))),
            NumberAny::Float(value) => Number::from_f64(value)
                .map(Variable::Number)
                .ok_or(GetError),
            // See the note in `build_union` about integers wider than i64.
            NumberAny::Int(big @ NumberInt::BigInt(_)) => Number::from_f64(f64::from(big))
                .map(Variable::Number)
                .ok_or(GetError),
        },
    }
}

/// Convert an interpreter result back into a union field.
fn variable_to_union(variable: &Variable) -> Result<JsonUnionField, GetError> {
    match variable {
        Variable::Null => Ok(JsonUnionField::JsonNull),
        Variable::Bool(value) => Ok(JsonUnionField::Bool(*value)),
        Variable::String(value) => Ok(JsonUnionField::Str(value.clone())),
        Variable::Number(number) => number
            .as_i64()
            .map(JsonUnionField::Int)
            .or_else(|| number.as_f64().map(JsonUnionField::Float))
            .ok_or(GetError),
        Variable::Array(_) => serde_json::to_string(variable)
            .map(JsonUnionField::Array)
            .map_err(|_| GetError),
        Variable::Object(_) => serde_json::to_string(variable)
            .map(JsonUnionField::Object)
            .map_err(|_| GetError),
        // An expression reference can't escape the expression that made it.
        Variable::Expref(_) => Err(GetError),
    }
}
