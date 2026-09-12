//! Lowering a JMESPath expression into a "scan plan".
//!
//! JMESPath's own interpreter works over a fully materialised tree of
//! [`jmespath::Variable`]s, so using it naively means parsing every byte of
//! every JSON document into an allocated tree before the expression gets to
//! throw almost all of it away.
//!
//! Most expressions used in practice start with a run of plain field accesses
//! and non-negative indices — `metadata.labels.env`, `events[0].id`. That
//! prefix is pure navigation: it needs no interpreter state, and jiter can walk
//! it directly over the raw bytes, skipping whole subtrees without allocating.
//!
//! So we split the parsed AST in two:
//!
//! * a [`Step`] prefix that jiter walks over the raw JSON, and
//! * the residual AST, which the JMESPath interpreter evaluates against
//!   whatever the prefix landed on.
//!
//! When the prefix covers the whole expression the interpreter never runs at
//! all. When it doesn't, we still only materialise the subtree the prefix
//! landed on rather than the whole document.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, PoisonError};

use datafusion::common::{exec_err, Result as DataFusionResult};
use jmespath::ast::Ast;
use jmespath::DEFAULT_RUNTIME;

/// A single literal navigation step, the part of a JMESPath expression that
/// jiter can resolve by itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Step {
    Key(String),
    Index(usize),
}

/// A JMESPath expression split into a jiter-walkable prefix and the residual
/// expression left for the interpreter.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ScanPlan {
    /// The original expression text, kept for JMESPath's error reporting.
    expression: String,
    prefix: Vec<Step>,
    /// `None` means the prefix covered the whole expression, so jiter alone can
    /// produce the result.
    residual: Option<Ast>,
}

impl ScanPlan {
    /// Parse `expression` and split it into a prefix and a residual.
    ///
    /// # Errors
    ///
    /// Returns an error if `expression` is not a valid JMESPath expression.
    pub(crate) fn parse(expression: &str) -> DataFusionResult<Self> {
        let ast = match jmespath::parse(expression) {
            Ok(ast) => ast,
            Err(e) => return exec_err!("invalid JMESPath expression '{expression}': {e}"),
        };
        check_functions(&ast, expression)?;

        let mut prefix = Vec::new();
        let residual = peel(ast, &mut prefix);
        Ok(Self {
            expression: expression.to_owned(),
            prefix,
            residual,
        })
    }

    pub(crate) fn expression(&self) -> &str {
        &self.expression
    }

    pub(crate) fn prefix(&self) -> &[Step] {
        &self.prefix
    }

    pub(crate) fn residual(&self) -> Option<&Ast> {
        self.residual.as_ref()
    }

    /// Which member of a JSON union input this plan reads from: `true` for the
    /// object member, `false` for the array member.
    ///
    /// Only the first step can tell us, so a plan that starts with something
    /// the interpreter has to handle (a projection, a function call) falls back
    /// to the object member, which is the far more common shape.
    pub(crate) fn is_object_lookup(&self) -> bool {
        !matches!(self.prefix.first(), Some(Step::Index(_)))
    }
}

/// Reject calls to functions that don't exist.
///
/// JMESPath resolves function names when it evaluates, where an unknown name is
/// a runtime error — and this crate reports runtime errors as SQL NULL, so a
/// typo would otherwise produce a silent column of nulls instead of a
/// complaint.
fn check_functions(ast: &Ast, expression: &str) -> DataFusionResult<()> {
    if let Ast::Function { name, .. } = ast {
        if DEFAULT_RUNTIME.get_function(name).is_none() {
            return exec_err!("unknown JMESPath function '{name}' in expression '{expression}'");
        }
    }
    for child in children(ast) {
        check_functions(child, expression)?;
    }
    Ok(())
}

/// The sub-expressions of an AST node.
fn children(ast: &Ast) -> Vec<&Ast> {
    match ast {
        Ast::And { lhs, rhs, .. }
        | Ast::Comparison { lhs, rhs, .. }
        | Ast::Or { lhs, rhs, .. }
        | Ast::Projection { lhs, rhs, .. }
        | Ast::Subexpr { lhs, rhs, .. } => vec![lhs.as_ref(), rhs.as_ref()],
        Ast::Condition {
            predicate, then, ..
        } => vec![predicate.as_ref(), then.as_ref()],
        Ast::Expref { ast, .. } => vec![ast.as_ref()],
        Ast::Flatten { node, .. } | Ast::Not { node, .. } | Ast::ObjectValues { node, .. } => {
            vec![node.as_ref()]
        }
        Ast::Function { args, .. } | Ast::MultiList { elements: args, .. } => args.iter().collect(),
        Ast::MultiHash { elements, .. } => elements.iter().map(|pair| &pair.value).collect(),
        Ast::Field { .. }
        | Ast::Identity { .. }
        | Ast::Index { .. }
        | Ast::Literal { .. }
        | Ast::Slice { .. } => Vec::new(),
    }
}

/// Consume literal navigation steps from the head of `ast` into `prefix`,
/// returning whatever is left for the interpreter.
///
/// The invariant this maintains is:
///
/// ```text
/// interpret(ast, data) == interpret(residual, navigate(data, prefix))
/// ```
///
/// which holds because `Subexpr`, `Projection`, `Flatten` and `ObjectValues`
/// all evaluate their left child against the current node and feed the result
/// onwards — so pre-navigating that child and replacing it with `Identity` is
/// exactly equivalent.
fn peel(ast: Ast, prefix: &mut Vec<Step>) -> Option<Ast> {
    match ast {
        Ast::Identity { .. } => None,
        Ast::Field { name, .. } => {
            prefix.push(Step::Key(name));
            None
        }
        Ast::Index { offset, idx } => match usize::try_from(idx) {
            Ok(index) => {
                prefix.push(Step::Index(index));
                None
            }
            // A negative index counts back from the end of the array, which a
            // forward-only scan can't resolve without knowing its length.
            Err(_) => Some(Ast::Index { offset, idx }),
        },
        Ast::Subexpr { offset, lhs, rhs } => match peel(*lhs, prefix) {
            // The left side lowered completely, so the right side is still
            // rooted at a position the prefix can reach.
            None => peel(*rhs, prefix),
            Some(lhs) => Some(Ast::Subexpr {
                offset,
                lhs: Box::new(lhs),
                rhs,
            }),
        },
        // These three stall the prefix — the interpreter has to fan out over
        // the result — but their left child is still plain navigation we can
        // hand to jiter.
        Ast::Projection { offset, lhs, rhs } => Some(Ast::Projection {
            offset,
            lhs: Box::new(peel_to_identity(*lhs, prefix, offset)),
            rhs,
        }),
        Ast::Flatten { offset, node } => Some(Ast::Flatten {
            offset,
            node: Box::new(peel_to_identity(*node, prefix, offset)),
        }),
        Ast::ObjectValues { offset, node } => Some(Ast::ObjectValues {
            offset,
            node: Box::new(peel_to_identity(*node, prefix, offset)),
        }),
        Ast::Not { offset, node } => Some(Ast::Not {
            offset,
            node: Box::new(peel_to_identity(*node, prefix, offset)),
        }),
        // A function's arguments are each evaluated against the current node,
        // so a single-argument call — `length(a.b)`, `keys(meta)`, `sort(xs)` —
        // can have that argument pre-navigated and rewritten to `f(@)`. With
        // more than one argument they navigate independently and there's no
        // single prefix to lift.
        Ast::Function {
            offset,
            name,
            mut args,
        } if args.len() == 1 => {
            let arg = args.pop().expect("length checked");
            Some(Ast::Function {
                offset,
                name,
                args: vec![peel_to_identity(arg, prefix, offset)],
            })
        }
        other => Some(other),
    }
}

/// Peel `ast` into `prefix`, standing in `Identity` if it lowered completely.
fn peel_to_identity(ast: Ast, prefix: &mut Vec<Step>, offset: usize) -> Ast {
    peel(ast, prefix).unwrap_or(Ast::Identity { offset })
}

/// Upper bound on cached plans, so a query with an expression column of high
/// cardinality can't grow the cache without limit.
const CACHE_CAPACITY: usize = 256;

static PLAN_CACHE: LazyLock<Mutex<HashMap<String, Arc<ScanPlan>>>> = LazyLock::new(Mutex::default);

/// Get the plan for `expression`, parsing it only if it isn't already cached.
///
/// The expression is nearly always a literal, so this keeps parsing off the
/// per-batch path entirely.
///
/// # Errors
///
/// Returns an error if `expression` is not a valid JMESPath expression.
pub(crate) fn cached_plan(expression: &str) -> DataFusionResult<Arc<ScanPlan>> {
    let mut cache = PLAN_CACHE.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(plan) = cache.get(expression) {
        return Ok(plan.clone());
    }
    // Parse errors aren't cached: they're rare, and a plan error fails the
    // query anyway.
    let plan = Arc::new(ScanPlan::parse(expression)?);
    if cache.len() >= CACHE_CAPACITY {
        cache.clear();
    }
    cache.insert(expression.to_owned(), plan.clone());
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(expression: &str) -> ScanPlan {
        ScanPlan::parse(expression).unwrap()
    }

    fn key(name: &str) -> Step {
        Step::Key(name.to_owned())
    }

    #[test]
    fn lowers_whole_expression() {
        for (expression, expected) in [
            ("@", vec![]),
            ("foo", vec![key("foo")]),
            ("foo.bar.baz", vec![key("foo"), key("bar"), key("baz")]),
            ("foo[0]", vec![key("foo"), Step::Index(0)]),
            (
                "foo.bar[2].baz",
                vec![key("foo"), key("bar"), Step::Index(2), key("baz")],
            ),
            (
                "foo[0][1]",
                vec![key("foo"), Step::Index(0), Step::Index(1)],
            ),
            ("foo.bar | baz", vec![key("foo"), key("bar"), key("baz")]),
        ] {
            let plan = plan(expression);
            assert_eq!(plan.prefix(), expected, "prefix for {expression}");
            assert_eq!(plan.residual(), None, "residual for {expression}");
        }
    }

    #[test]
    fn lowers_prefix_of_partial_expression() {
        for (expression, expected) in [
            ("foo[*].bar", vec![key("foo")]),
            ("foo[].bar", vec![key("foo")]),
            ("foo.*.bar", vec![key("foo")]),
            ("foo[?a=='b'].c", vec![key("foo")]),
            ("foo[-1]", vec![key("foo")]),
            ("foo[1:3]", vec![key("foo")]),
            (
                "a.b.c[0].d[*].e",
                vec![key("a"), key("b"), key("c"), Step::Index(0), key("d")],
            ),
            ("a.b[*].c[*].d", vec![key("a"), key("b")]),
            // A single-argument function pre-navigates its argument.
            ("length(a.b)", vec![key("a"), key("b")]),
            ("keys(meta)", vec![key("meta")]),
            ("sort(a.b.c)", vec![key("a"), key("b"), key("c")]),
            ("length(a[0].b)", vec![key("a"), Step::Index(0), key("b")]),
            // `!` binds tighter than `.`, so this is `(!a).b` and only `a`
            // is reachable by the scan.
            ("!a.b", vec![key("a")]),
            ("!(a.b)", vec![key("a"), key("b")]),
            // With more than one argument each navigates from the same root, so
            // there's no single prefix to lift.
            ("not_null(a, b)", vec![]),
            ("sort_by(a, &b)", vec![]),
        ] {
            let plan = plan(expression);
            assert_eq!(plan.prefix(), expected, "prefix for {expression}");
            assert!(plan.residual().is_some(), "residual for {expression}");
        }
    }

    #[test]
    fn lowers_nothing_when_head_is_not_navigation() {
        for expression in ["{a: foo, b: bar}", "[foo, bar]", "'literal'", "a || b"] {
            let plan = plan(expression);
            assert!(plan.prefix().is_empty(), "prefix for {expression}");
            assert!(plan.residual().is_some(), "residual for {expression}");
        }
    }

    #[test]
    fn union_member_hint_follows_first_step() {
        assert!(plan("foo.bar").is_object_lookup());
        assert!(!plan("[0].foo").is_object_lookup());
        assert!(plan("length(@)").is_object_lookup());
    }

    #[test]
    fn rejects_invalid_expression() {
        assert!(ScanPlan::parse("foo..bar").is_err());
    }

    #[test]
    fn rejects_unknown_functions() {
        for expression in [
            "items(@)",
            "length(itmes(@))",
            "a[?nosuch(@)]",
            "{a: nosuch(@)}",
            "[nosuch(@)]",
            "sort_by(a, &nosuch(@))",
            "!nosuch(@)",
        ] {
            let err = ScanPlan::parse(expression).unwrap_err().to_string();
            assert!(
                err.contains("unknown JMESPath function"),
                "for {expression}: {err}"
            );
        }
    }

    #[test]
    fn accepts_known_functions() {
        for expression in [
            "length(@)",
            "sort_by(a, &b)",
            "map(&c, a)",
            "not_null(a, b)",
        ] {
            assert!(ScanPlan::parse(expression).is_ok(), "for {expression}");
        }
    }

    #[test]
    fn caches_by_expression() {
        let first = cached_plan("cache.test.expr").unwrap();
        let second = cached_plan("cache.test.expr").unwrap();
        assert!(Arc::ptr_eq(&first, &second));
    }
}
