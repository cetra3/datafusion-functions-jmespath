# datafusion-functions-jmespath

A [JMESPath](https://jmespath.org) UDF for [DataFusion](https://datafusion.apache.org),
built on [jmespath.rs](https://github.com/jmespath/jmespath.rs) and [jiter](https://github.com/pydantic/jiter).

```sql
select jmespath(payload, 'locations[?state == ''WA''].name | sort(@)') from events;
```

## Usage

Register the UDF with a session:

```rust
let mut ctx = SessionContext::new();
datafusion_functions_jmespath::register_all(&mut ctx)?;
```

Or build the expression directly with `datafusion_functions_jmespath::functions::jmespath(json, expression)`.

## Return type

Results keep their JSON type, using the same sparse union as
`datafusion-functions-json`: strings come back as strings, numbers as ints or
floats, and nested JSON as text tagged with Arrow's canonical JSON extension type.

| Situation | Result |
| --- | --- |
| Path doesn't exist | `NULL` |
| Document doesn't parse | `NULL` |
| JMESPath runtime type error | `NULL` |
| Expression doesn't parse | query error |
| Unknown function | query error |

**Duplicate keys:** a forward scan returns the *first* match, whereas JMESPath's
interpreter (which builds a map) keeps the last. This matches `json_get`. Key
counting (`length`, `keys`) still deduplicates.

## How it works

JMESPath's interpreter operates on a fully materialised tree, so using it
directly means parsing every document in full before the expression discards
almost all of it.

Most expressions, though, start with plain field accesses and non-negative
indices (`meta.labels.env`, `events[0].id`). That prefix is pure navigation, and
jiter can walk it over the raw bytes without allocating, skipping whole subtrees
as it goes.

So each expression is parsed once, cached, and split in two:

| Expression | Scanned by jiter | Interpreted by JMESPath |
| --- | --- | --- |
| `meta.labels.env` | `meta.labels.env` | — |
| `a.b.c[0].d[*].e` | `a.b.c[0].d` | `[*].e` |
| `locations[?state == 'WA'].name` | `locations` | `[?state == 'WA'].name` |
| `length(meta.tags)` | `meta.tags` | `length(@)` |
| `{a: x, b: y}` | — | all of it |

- **Prefix covers everything:** the interpreter never runs; the result is read
  (or, for nested JSON, sliced) straight from the original bytes.
- **Residual remains:** only the subtree the prefix landed on is materialised,
  parsed by jiter directly into JMESPath values with no `serde_json::Value` in
  between.
- **`length`, `keys`, `type`:** answered by scanning rather than materialising,
  since the prefix lifts a single-argument function's argument.

`tests/differential.rs` checks that the split returns exactly what the
unmodified interpreter would, across every expression shape that can stop the
prefix.

## Performance

Benchmarked against [`datafusion-functions-json`](https://github.com/datafusion-contrib/datafusion-functions-json)
over an 8192-row batch (`cargo bench`).

**Summary:** for an equivalent path with the same return type, the two crates
are the same speed. Differences come from the return type or from expressions
that can't be scanned.

The `datafusion-functions-json` timings are from an earlier run; the `jmespath`
timings were rerun against jiter 0.17, so treat deltas within a few percent as
noise.

### Same return type

`json_get` returns the same JSON union, so this isolates the scan.

| Expression | `json_get` | `jmespath` | Delta |
| --- | ---: | ---: | ---: |
| `id` | 287µs | 298µs | +3.8% |
| `meta.labels.env` | 2645µs | 2570µs | −2.8% |
| `deep.a.b.c.d` | 4030µs | 4155µs | +3.1% |
| `events[1]` | 3779µs | 3971µs | +5.1% |
| `events[1].name` | 3579µs | 3641µs | +1.7% |
| `padding[30]` | 2213µs | 2132µs | −3.7% |
| `meta.labels` (object) | 2985µs | 2811µs | −5.8% |
| `meta.tags` (array) | 3208µs | 3109µs | −3.1% |
| `meta.nope` (missing) | 2964µs | 3082µs | +4.0% |
| `trailing` (last key) | 4242µs | 4325µs | +2.0% |

### Typed return

The json crate's typed functions build a typed column; `jmespath` always builds
the seven-child union.

| Expression | json crate | `jmespath` | Delta |
| --- | ---: | ---: | ---: |
| `meta.labels.env` | 2471µs (`json_get_str`) | 2570µs | +4.0% |
| `meta.ratio` | 3073µs (`json_get_float`) | 3045µs | −0.9% |
| `meta.ok` | 2903µs (`json_get_bool`) | 2927µs | +0.8% |
| `meta.labels.env` | 2590µs (`json_as_text`) | 2570µs | −0.8% |
| `meta.labels` | 2802µs (`json_get_json`) | 2811µs | +0.3% |
| `id` | 160µs (`json_get_int`) | 298µs | +86% |

The union costs about **17ns/row**. That's lost in the noise when scanning dominates, but
`id` is the first key, so there's almost no scanning and the union is most of
the cost. It's the price of preserving the JSON type, not a scan cost.

### Function calls

| Expression | json crate | `jmespath` | Delta |
| --- | ---: | ---: | ---: |
| `length(meta.tags)` (array) | 2830µs (`json_length`) | 2843µs | +0.5% |
| `length(meta.labels)` (object) | 2571µs (`json_length`) | 3114µs | +21% |
| `keys(meta.labels)` | 3032µs (`json_object_keys`) | 3462µs | +14% |

Array `length` is an allocation-free skip loop. Object `length` and `keys` build
a `BTreeSet` of keys to deduplicate them as the interpreter would, and that
allocation accounts for the extra cost.

### Dictionary-encoded columns

About 0.6ns/row slower, all in the dictionary output path (key cast, null
masking, rebuilding the array). It shows up as 6–10% only because the dictionary
short-circuit makes the baseline tiny (≈35µs per batch). Plain `Utf8` columns
show no difference.

### Expressions with no equivalent

| Expression | Time |
| --- | ---: |
| `events[-1].name` | 6.7ms |
| `events[*].name` | 7.3ms |
| ``events[?n > `1`].name`` | 8.1ms |
| `padding[0:5]` | 11.2ms |
| `{env: meta.labels.env, tags: meta.tags}` | 25.7ms |

The most expensive are expressions rooted in a multi-select or multi-argument
function: there's no prefix to scan, so the whole document is materialised.
