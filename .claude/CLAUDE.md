# i2

The middleware toolbox: meta-programming tools for building declarative frameworks
— function signatures as data, decorators, wrapping/routing, multi-object
composition. Packaged with `pyproject.toml` (hatchling) and heavily depended upon
across the fleet — see Dependents below.

## Module map (`i2/`)

- `signatures.py` — **the core**: `Sig` (extended `Signature`), `Param`,
  `call_forgivingly`/`call_somewhat_forgivingly`, `name_of_obj`. Signature calculus
  underlies most of the rest of the package.
- `deco.py` — decorator tools: `FuncFactory`, `preprocess`/`postprocess`,
  `preprocess_arguments`, `input_output_decorator`,
  `wrap_class_methods_input_and_output`, `double_up_as_factory`.
- `wrapper.py` — the `Wrap` class: wraps a function while controlling its
  ingress/egress and signature.
- `multi_object.py` — operating on a fixed collection of functions: `MultiObj`,
  `MultiFunc`, `Pipe`, `FuncFanout`, `FlexFuncFanout`, `ParallelFuncs`, `ContextFanout`.
  Skill: `i2-multi-object`.
- `routing_forest.py` — specifying functions through trees/forests of conditions.
- `castgraph.py` — a transformation service for the "stable role, unstable
  representation" problem. Skill: `i2-castgraph`.
- `base.py` / `doc_mint.py` — "mints": `Mapping` views of an object's interface.
- `footprints.py` — what attributes of an input object a function actually uses.
- `key_path.py` — flattening maps and manipulating key paths.
- `io_trans.py` — building input/output-transforming decorators.
- `itypes.py`, `util.py`, `errors.py` — types, misc utilities, error objects.
- `chain_map.py` — merging mappings; **marked for deprecation**, don't build on it.

## Tests (verified)

```bash
uv venv .venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest
# 782 passed, 2 xfailed
```
`[tool.pytest.ini_options]` supplies `--doctest-modules` and the `i2/examples`,
`i2/scrap` ignores, so plain `pytest` matches CI. `i2/tests/test_readme.py` runs
every python block of `README.md` in order, so README examples must run.

CI is the wads **inline** uv workflow (`.github/workflows/ci.yml`), configured by
`[tool.wads.ci]`. It is inline rather than the reusable-workflow stub only because
the stub's Pages job can't pass the epythet v2 pilot pin (`epythet-spec`). The
lint gate is `ruff check i2` with only `D100` (module docstrings) selected; don't
widen it without fixing what it finds. A merge to `master` bumps the version and
publishes to PyPI, so never edit the version by hand.

## Invariant: this package has no safety net for its own breaking changes

i2#88/i2#89 (2026-09-22): a `NotSet` sentinel default landed in `FuncFactory`
signatures, shipped to PyPI (0.1.71), and broke `front` (number-input TypeError,
text inputs prefilled with the literal string `'NotSet'`) and `py2http` (OpenAPI
JSON) — i2's own test suite didn't catch it because nothing in it exercises a
`FuncFactory` signature end-to-end through a consumer. Reverted; re-land plan is
on the reopened [i2mint/i2#48](https://github.com/i2mint/i2/issues/48). **Before
changing any public signature, default, or return type in `signatures.py`,
`deco.py`, or `wrapper.py`, check dependents' tests, not just this repo's.**

## Docs & skills

- `.claude/skills/`: `i2-signatures`, `i2-sig-arithmetic`, `i2-wrapper`,
  `i2-multi-object`, `i2-castgraph`.
- `misc/docs/wrapper_improvement_docs/`; notebooks under `misc/`.

## Dependents

50 packages import `i2` (fleet dependency graph), including `allude`, `chromadol`,
`config2py`, `crude`, `dagapp`, `front`, `guided`, `http2py`, `meshed`, `mongodol`,
`py2http`, `py2json`, `tabled`, `taped`, `uf` — see
`fleet_dependents.json` for the full list. This is the widest blast radius in the
fleet; treat any signature/default/return-type change as breaking until proven
otherwise.
