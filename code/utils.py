import ast
from collections import OrderedDict

def extract_imports(code: str):
    """
    Extract top-level imported libraries from a Python code string.
    Returns a sorted, comma-separated string of unique imports.
    """
    imports = set()

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""

    for node in ast.walk(tree):
        # Handle: import a, b.c
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])

        # Handle: from a.b import c
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])

    return ", ".join(sorted(imports))


def extract_imports_features_v2(code: str):
    """
    Returns dict where:
      - keys are main library names (or alias key for 'import X as y' when X has no submodule)
      - values are list of imported "functionalities" per your examples

    Rules (per user examples):
      import a                  -> {'a': []}
      import h.j                -> {'h': ['j']}
      import n as nn            -> {'n': ['nn']}
      import tt as t            -> {'t': ['tt']}          # alias becomes key, original becomes value
      import u.v as uv          -> {'u': ['v']}           # key is main lib, value is submodule remainder
      from b import c           -> {'b': ['c']}
      from d.e import f.g, k.m  -> {'d': ['e.f.g','e.k.m']}
      from r.s import y         -> {'r': ['s.y']}
      from u.p import l.m as lm -> {'u': ['p.l.m']}       # ignore alias, keep full path
      from s.a import (aa,bb)   -> {'s': ['a.aa','a.bb']}
    """
    if not isinstance(code, str) or not code.strip():
        return {}

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}

    out = OrderedDict()

    def ensure_key(k: str):
        if k not in out:
            out[k] = []

    def add_unique(k: str, item: str):
        if item and item not in out[k]:
            out[k].append(item)

    for node in ast.walk(tree):

        # ---- import ... ----
        if isinstance(node, ast.Import):
            for alias in node.names:
                full = alias.name               # e.g., "a", "h.j", "u.v", "tt"
                parts = full.split(".")
                main = parts[0]
                rest = ".".join(parts[1:])      # e.g., "j", "v", "" if none

                if alias.asname:
                    if rest == "":
                        # import tt as t  -> key=t, value=['tt']
                        ensure_key(alias.asname)
                        add_unique(alias.asname, main)
                    else:
                        # import u.v as uv -> key=u, value=['v'] (ignore alias)
                        ensure_key(main)
                        add_unique(main, rest)
                else:
                    # no alias
                    ensure_key(main)
                    if rest:
                        # import h.j -> key=h, value=['j']
                        add_unique(main, rest)
                    # import a -> []

        # ---- from ... import ... ----
        elif isinstance(node, ast.ImportFrom):
            if not node.module:
                continue

            module_full = node.module           # e.g., "d.e", "r.s", "u.p", "s.a"
            module_parts = module_full.split(".")
            main = module_parts[0]
            module_remainder = ".".join(module_parts[1:])  # e.g., "e", "s", "p", "a"

            ensure_key(main)

            for alias in node.names:
                # IMPORTANT per your new spec:
                # - ignore alias.asname (lm)
                # - keep alias.name (l.m)
                imported_name = alias.name       # e.g., "c", "f.g", "k.m", "y", "l.m", "aa"

                if module_remainder:
                    add_unique(main, f"{module_remainder}.{imported_name}")
                else:
                    add_unique(main, imported_name)

    return dict(out)


def extract_import_aliases(code: str) -> dict:
    """
    Return dict: { 'module[.submodule][.name]': 'alias' } for imports that use 'as'.
    Only entries with an alias are included.

    Examples:
      import a as aaa              -> {'a': 'aaa'}
      import h.j as lm             -> {'h.j': 'lm'}
      from b import c as kd        -> {'b.c': 'kd'}
      from r.s import y as uwo     -> {'r.s.y': 'uwo'}
      from u.p import l.m as qw    -> {'u.p.l.m': 'qw'}
    """
    if not isinstance(code, str) or not code.strip():
        return {}

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}

    out = {}

    for node in ast.walk(tree):
        # import x.y as z
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:  # only keep aliased ones
                    out[alias.name] = alias.asname

        # from a.b import c.d as e
        elif isinstance(node, ast.ImportFrom):
            if not node.module:
                continue
            for alias in node.names:
                if alias.asname:  # only keep aliased ones
                    out[f"{node.module}.{alias.name}"] = alias.asname

    return out



import ast

def extract_call_params(code: str, target: str) -> dict:
    """
    Detect whether `target` is called anywhere in `code` and extract its keyword parameters.

    target examples:
      - "Engine"
      - "sgl.Engine"
      - "sglang.Engine"
      - "TurbomindEngineConfig"
      - "tensor.TurbomindEngineConfig"

    Return:
      {} if not found or code empty/invalid
      {target: {kw: value, ...}} if found

    Notes:
      - Keyword names are lowercased (X -> x).
      - Repeated keywords keep the last value (dict behavior).
      - Values:
          * literals -> Python values (False, "lru", 3, ...)
          * variables -> name string ("A")
          * calls -> "FuncName()" (e.g., "B()")
    """
    if not isinstance(code, str) or not code.strip():
        return {}

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}

    target_parts = target.split(".")

    def get_func_parts(func_node):
        """Return dotted name parts for a call func, e.g. sgl.Engine -> ['sgl','Engine']."""
        if isinstance(func_node, ast.Name):
            return [func_node.id]
        if isinstance(func_node, ast.Attribute):
            parts = []
            cur = func_node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
                return list(reversed(parts))
        return None

    def norm_value(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Call):
            # represent calls as CalleeName()
            f = node.func
            if isinstance(f, ast.Name):
                return f"{f.id}()"
            if isinstance(f, ast.Attribute):
                return f"{f.attr}()"
            return "call()"
        return ast.unparse(node) if hasattr(ast, "unparse") else "<expr>"

    # Walk entire AST to catch nested calls (examples 3,7,8)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            parts = get_func_parts(node.func)
            if parts == target_parts:
                params = {}
                for kw in node.keywords:
                    if kw.arg is not None:  # ignore **kwargs
                        params[kw.arg.lower()] = norm_value(kw.value)
                return {target: params}

    return {}


import ast
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple


def extract_models_by_framework(code: str) -> dict:
    """
    Extract model identifiers (HF IDs / local paths / URIs) from an LLM-serving Python script,
    grouped by framework: vLLM, SGLang, TensorRT-LLM, LMDeploy, FlashInfer, Unknown.

    Fixes over the previous version:
    - Never turns numeric/bool/None constants into EXPR-strings (prevents EXPR:1024 etc.).
    - Filters out obvious non-model strings (sentences, messages, test placeholders).
    - Produces meaningful placeholders for non-literal model values:
        VAR:<name>  (Name / Attribute / Subscript)
        EXPR:<expr> (only when context strongly indicates model and it can't be reduced)
    - Better resolves variables and dict kwargs (including **kwargs).
    - Recognizes HF download helpers (hf_hub_download / snapshot_download / safe_snapshot_download).
    - Better framework attribution and avoids mis-attributing random expressions.
    """

    # ---------------------------
    # Required output skeleton
    # ---------------------------
    result_keys = ["vLLM", "SGLang", "TensorRT-LLM", "LMDeploy", "FlashInfer", "Unknown"]
    out_store: Dict[str, OrderedDict] = {k: OrderedDict() for k in result_keys}

    def _add(framework: str, model_id: str) -> None:
        if framework not in out_store:
            framework = "Unknown"
        if model_id and model_id not in out_store[framework]:
            out_store[framework][model_id] = None

    # ---------------------------
    # AST utilities
    # ---------------------------
    def _safe_unparse(node: ast.AST) -> str:
        try:
            return ast.unparse(node)
        except Exception:
            return node.__class__.__name__

    def _short(node: ast.AST, n: int = 90) -> str:
        s = " ".join(_safe_unparse(node).split())
        return s if len(s) <= n else s[:n] + "..."

    def _qualname(call: ast.Call) -> str:
        fn = call.func
        if isinstance(fn, ast.Name):
            return fn.id
        if isinstance(fn, ast.Attribute):
            parts = []
            cur = fn
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            parts.reverse()
            return ".".join(parts)
        return _short(fn)

    def _node_to_varref(node: ast.AST) -> Optional[str]:
        """
        Convert Name/Attribute/Subscript chains into a stable placeholder.
        Examples:
          args.model_name_or_path -> VAR:args.model_name_or_path
          self.model_path         -> VAR:self.model_path
          BASE_CONFIG['model']    -> VAR:BASE_CONFIG['model']
        """
        if isinstance(node, ast.Name):
            return f"VAR:{node.id}"
        if isinstance(node, ast.Attribute):
            base = _node_to_varref(node.value)
            if base:
                return f"{base}.{node.attr}".replace("VAR:", "VAR:")
            # fallback: still represent attr
            return f"VAR:{_short(node)}"
        if isinstance(node, ast.Subscript):
            # keep it readable
            return f"VAR:{_short(node)}"
        return None

    # ---------------------------
    # "Model identifier" heuristics
    # ---------------------------
    PATHISH = ("/", "./", "../", "s3://", "gs://", "hf://", "file://", "http://", "https://")
    # Detect HF-style repo id "org/name" (no spaces), or path/uri.
    def _looks_like_model_id(s: str) -> bool:
        if not isinstance(s, str):
            return False
        t = s.strip()
        if len(t) < 3:
            return False

        # Filter common false positives: sentences / messages
        if " " in t and not any(p in t for p in PATHISH):
            # allow "org/name" without spaces; disallow long sentences
            return False

        # Filter test placeholders like DEFAULT_*_MODEL_NAME_FOR_TEST
        up = t.upper()
        if ("MODEL_NAME_FOR_TEST" in up) or ("DEFAULT_" in up and "TEST" in up):
            # if it also looks like an actual model id/path, keep it; otherwise drop
            if not ("/" in t or any(p in t for p in PATHISH)):
                return False

        if any(p in t for p in PATHISH):
            return True

        # HF repo style
        if "/" in t and "://" not in t and t.count("/") >= 1:
            a, b = t.split("/", 1)
            if a and b and (" " not in a) and (" " not in b) and len(a) > 1 and len(b) > 1:
                return True

        return False

    # Context-based key heuristic: don't rely on a fixed list, but do use semantic tokens.
    MODELISH_TOKENS = (
        "model", "checkpoint", "ckpt", "pretrained", "repo", "hf", "weights", "engine", "artifact", "snapshot"
    )
    def _is_modelish_key(k: str) -> bool:
        if not isinstance(k, str) or not k:
            return False
        kk = k.lower().replace("-", "_")
        return any(tok in kk for tok in MODELISH_TOKENS)

    # Reject placeholder EXPR if it's clearly non-model (numbers, None, function names, internal objects)
    def _is_bad_placeholder(s: str) -> bool:
        if not isinstance(s, str):
            return True
        t = s.strip()
        if not t:
            return True
        # Reject messages / error strings / logs
        if any(phrase in t.lower() for phrase in ["not initialized", "cannot", "failed", "exiting", "please increase"]):
            return True
        # Reject pure numeric-like
        if t.replace(".", "", 1).isdigit():
            return True
        # Reject obvious config field references that are not resolved (unless they look like paths)
        if t.startswith("EXPR:"):
            expr = t[len("EXPR:"):].strip()
            # if expression includes arithmetic or formatting, likely not a model id
            if any(op in expr for op in ["+", "*", "-", "/", "%", "{", "}", "len(", "expected", "parameters"]):
                return True
            # if expression is a function call, often not directly a model id
            if "(" in expr and ")" in expr:
                return True
            # if it's a property access, allow only as VAR:... placeholders instead
            if "." in expr or "[" in expr:
                return True
        return False

    # ---------------------------
    # Imports + framework flags
    # ---------------------------
    imports: Dict[str, str] = {}

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {k: [] for k in result_keys}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                asname = a.asname or a.name.split(".")[0]
                imports[asname] = a.name
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for a in node.names:
                asname = a.asname or a.name
                imports[asname] = (mod + "." + a.name).strip(".")

    mods = set(imports.values()) | set(imports.keys())
    mods_l = {m.lower() for m in mods if isinstance(m, str)}

    import_flags = {
        "vllm": any(m == "vllm" or m.startswith("vllm.") for m in mods_l),
        "sglang": any(m == "sglang" or m.startswith("sglang.") or m == "sgl" or m.startswith("sgl.") for m in mods_l),
        "lmdeploy": any(m == "lmdeploy" or m.startswith("lmdeploy.") for m in mods_l),
        "tensorrt_llm": any(m == "tensorrt_llm" or m.startswith("tensorrt_llm.") or "trtllm" in m for m in mods_l),
        "flashinfer": any(m == "flashinfer" or m.startswith("flashinfer.") for m in mods_l),
        "transformers": any(m == "transformers" or m.startswith("transformers.") for m in mods_l),
        "huggingface_hub": any(m == "huggingface_hub" or m.startswith("huggingface_hub.") for m in mods_l),
    }

    # ---------------------------
    # Value resolution environment
    # ---------------------------
    env: Dict[str, Any] = {}

    def _resolve(node: ast.AST) -> Any:
        """
        Return a best-effort resolved object:
          - str/int/float/bool/None for constants
          - dict/list for literals
          - resolved env[name] for Names
          - VAR:... for Name/Attribute/Subscript when not resolvable
          - EXPR:... only when necessary (and only if caller decides to use it)
        """
        if isinstance(node, ast.Constant):
            return node.value  # <-- key fix: keep numbers/bools/None as their types
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            return _node_to_varref(node) or f"VAR:{node.id}"
        if isinstance(node, ast.Attribute) or isinstance(node, ast.Subscript):
            return _node_to_varref(node) or f"VAR:{_short(node)}"
        if isinstance(node, ast.JoinedStr):
            # f-string: keep as EXPR; may still be meaningful if it contains a path.
            return f"EXPR:{_short(node)}"
        if isinstance(node, ast.Dict):
            d = {}
            for k, v in zip(node.keys, node.values):
                if k is None:
                    continue
                kk = _resolve(k)
                kk = kk if isinstance(kk, str) else str(kk)
                d[kk] = _resolve(v)
            return d
        if isinstance(node, (ast.List, ast.Tuple)):
            return [_resolve(e) for e in node.elts]
        # If it's a call and it's clearly a "download model" helper, try to extract model id.
        if isinstance(node, ast.Call):
            q = _qualname(node).lower()

            # huggingface_hub.hf_hub_download(repo_id=..., ...) or hf_hub_download("repo", filename=...)
            if q.endswith("hf_hub_download"):
                # repo_id may be first positional or keyword
                if node.args:
                    v0 = _resolve(node.args[0])
                    return v0
                for kw in node.keywords:
                    if kw.arg and kw.arg.lower() in {"repo_id", "model", "model_id", "hf_model", "repo"}:
                        return _resolve(kw.value)
                return f"EXPR:{_short(node)}"

            # snapshot_download(model_id/model_name/repo_id=...) or snapshot_download("repo")
            if q.endswith("snapshot_download") or q.endswith("safe_snapshot_download"):
                if node.args:
                    return _resolve(node.args[0])
                for kw in node.keywords:
                    if kw.arg and any(tok in kw.arg.lower() for tok in ["model", "repo", "id", "path", "dir"]):
                        return _resolve(kw.value)
                return f"EXPR:{_short(node)}"

            # transformers.from_pretrained("model")
            if q.endswith("from_pretrained") and node.args:
                return _resolve(node.args[0])

            # otherwise keep expression (caller may ignore)
            return f"EXPR:{_short(node)}"

        return f"EXPR:{_short(node)}"

    def _merge_kwargs(call: ast.Call) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for kw in call.keywords:
            if kw.arg is None:
                # **kwargs
                v = _resolve(kw.value)
                if isinstance(v, dict):
                    merged.update(v)
                else:
                    # keep a marker, but don't treat as model by itself
                    merged["**kwargs"] = v
            else:
                merged[kw.arg] = _resolve(kw.value)
        return merged

    # First pass: capture simple assignments (strings/dicts/lists/constants)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            val = _resolve(node.value)
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    env[tgt.id] = val
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            env[node.target.id] = _resolve(node.value)

    # ---------------------------
    # Framework inference from call
    # ---------------------------
    def _framework_from_call(call: ast.Call) -> Optional[str]:
        q = _qualname(call).lower()

        # vLLM
        if "vllm" in q or q.endswith(".llm") or q.endswith(".asyncllmengine") or "engineargs" in q:
            return "vLLM"
        if import_flags["vllm"] and _qualname(call) in {"LLM", "AsyncLLMEngine", "EngineArgs", "AsyncEngineArgs"}:
            return "vLLM"

        # SGLang
        if "sglang" in q or q.startswith("sgl.") or q.split(".")[0] == "sgl":
            return "SGLang"
        if import_flags["sglang"] and _qualname(call) in {"Runtime", "Engine"}:
            return "SGLang"

        # TensorRT-LLM
        if "tensorrt_llm" in q or "trtllm" in q:
            return "TensorRT-LLM"
        if import_flags["tensorrt_llm"] and any(x in q for x in ["builder", "build", "engine", "executor", "runner"]):
            return "TensorRT-LLM"

        # LMDeploy
        if "lmdeploy" in q or q.endswith(".pipeline") or "turbomindengineconfig" in q or "pytorchengineconfig" in q:
            return "LMDeploy"
        if import_flags["lmdeploy"] and _qualname(call) in {"pipeline", "TurbomindEngineConfig", "PytorchEngineConfig"}:
            return "LMDeploy"

        # FlashInfer
        if "flashinfer" in q:
            return "FlashInfer"

        return None

    # ---------------------------
    # Collect model candidates from objects with context
    # ---------------------------
    def _collect_models(obj: Any, *, context_key: Optional[str]) -> List[str]:
        """
        Only emit:
          - real model-id strings (paths/uris/HF IDs)
          - VAR:... placeholders when context_key is modelish
        Never emit:
          - numeric/bool/None
          - EXPR placeholders unless they clearly embed a path/uri/HF id
        """
        found: List[str] = []

        def rec(o: Any, ck: Optional[str]) -> None:
            if isinstance(o, (int, float, bool)) or o is None:
                return

            if isinstance(o, str):
                t = o.strip()
                if not t:
                    return

                # If it's a VAR placeholder: include only when key/context indicates model-ish usage
                if t.startswith("VAR:"):
                    if ck and _is_modelish_key(ck):
                        found.append(t)
                    return

                # If it's EXPR placeholder: include only if it likely embeds a model identifier string.
                if t.startswith("EXPR:"):
                    if _is_bad_placeholder(t):
                        return
                    # allow EXPR if it contains pathish or HF-like org/name substring
                    expr = t[len("EXPR:") :]
                    if any(p in expr for p in PATHISH) or ("/" in expr and " " not in expr):
                        if ck and _is_modelish_key(ck):
                            found.append(t)
                    return

                # Real string literal: must look like model id/path/uri, OR be modelish-key + pathish/repo-ish
                if _looks_like_model_id(t):
                    found.append(t)
                else:
                    # If key says model but string doesn't look like model, ignore (fixes messages, sentences)
                    return
                return

            if isinstance(o, dict):
                for kk, vv in o.items():
                    kks = kk if isinstance(kk, str) else str(kk)
                    rec(vv, kks)
                return

            if isinstance(o, list):
                for vv in o:
                    rec(vv, ck)
                return

            # anything else ignored
            return

        rec(obj, context_key)
        return found

    # ---------------------------
    # Extraction pass over calls
    # ---------------------------
    seen_framework_calls = {k: False for k in result_keys}
    seen_framework_calls["Unknown"] = True  # always "available"

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        fw = _framework_from_call(node)
        if fw:
            seen_framework_calls[fw] = True

        qn = _qualname(node)
        qnl = qn.lower()

        kwdict = _merge_kwargs(node)

        candidates: List[str] = []

        # 1) Framework-specific positional model argument patterns:
        #    - vLLM: LLM(<model>, ...) sometimes positional
        #    - LMDeploy: pipeline(<model>, ...) commonly positional
        #    - SGLang: Engine(<model>, ...) sometimes positional
        def _pos0_as_model_if_stringy() -> None:
            if node.args:
                v0 = _resolve(node.args[0])
                # Only interpret positional-0 as model if it is string / VAR / EXPR with pathish
                if isinstance(v0, str):
                    for s in _collect_models(v0, context_key="model"):
                        candidates.append(s)

        if fw == "vLLM":
            if qnl.endswith(".llm") or qn in {"LLM", "vllm.LLM"}:
                _pos0_as_model_if_stringy()
        if fw == "LMDeploy":
            if qnl.endswith(".pipeline") or qn == "pipeline" or qn.endswith(".pipeline"):
                _pos0_as_model_if_stringy()
        if fw == "SGLang":
            if qn in {"Engine", "Runtime"} or qnl.endswith(".engine") or qnl.endswith(".runtime"):
                _pos0_as_model_if_stringy()

        # 2) Keyword-based: any modelish key (semantic tokens)
        for k, v in kwdict.items():
            if k == "**kwargs":
                continue
            if _is_modelish_key(k):
                candidates.extend(_collect_models(v, context_key=k))

        # 3) HF download helpers (even if not in modelish key)
        #    These often contain the true model id for GGUF or HF snapshots.
        if qnl.endswith("hf_hub_download") or qnl.endswith("snapshot_download") or qnl.endswith("safe_snapshot_download"):
            # repo_id usually first arg or a keyword
            if node.args:
                v0 = _resolve(node.args[0])
                candidates.extend(_collect_models(v0, context_key="repo_id"))
            for k, v in kwdict.items():
                if k and any(tok in k.lower() for tok in ["repo", "model", "id", "path", "dir"]):
                    candidates.extend(_collect_models(v, context_key=k))

        # 4) Transformers from_pretrained is a strong model indicator but not a "framework" by itself here.
        #    It should be attributed to the runtime framework present in the same script if plausible.
        #    We'll store it temporarily as Unknown and re-attribute later (esp. FlashInfer case).
        transformers_fp = False
        if qnl.endswith("from_pretrained"):
            transformers_fp = True
            if node.args:
                v0 = _resolve(node.args[0])
                candidates.extend(_collect_models(v0, context_key="pretrained_model_name_or_path"))
            for k, v in kwdict.items():
                if k and _is_modelish_key(k):
                    candidates.extend(_collect_models(v, context_key=k))

        # Record candidates
        if candidates:
            target_fw = fw or "Unknown"
            for s in candidates:
                _add(target_fw, s)

    # ---------------------------
    # Subprocess CLI extraction (best-effort)
    # ---------------------------
    MODEL_FLAGS = {
        "--model", "--model-name", "--model_name", "--model-path", "--model_path", "--model-dir", "--model_dir",
        "--checkpoint", "--ckpt", "--repo-id", "--repo_id", "--hf-model", "--hf_model", "--pretrained",
        "--engine-dir", "--engine_dir", "--hf_model_dir", "--hf_model_path"
    }

    def _extract_from_cli_list(tokens: List[Any]) -> List[str]:
        out: List[str] = []
        last_model_flag = False
        for tok in tokens:
            if not isinstance(tok, str):
                continue
            t = tok.strip().strip("\"'")
            if not t:
                continue
            tl = t.lower()
            if tl in MODEL_FLAGS:
                last_model_flag = True
                continue
            if last_model_flag:
                if _looks_like_model_id(t):
                    out.append(t)
                else:
                    # If it doesn't look like model id, ignore (fixes noise)
                    pass
                last_model_flag = False
                continue
            # positional: sometimes commands pass model directly (lmdeploy serve ... <model>)
            if _looks_like_model_id(t):
                out.append(t)
        return out

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        qnl = _qualname(node).lower()
        if qnl in {"subprocess.run", "subprocess.popen", "subprocess.call", "run", "popen", "call"} and node.args:
            argv0 = _resolve(node.args[0])
            if isinstance(argv0, list):
                for s in _extract_from_cli_list(argv0):
                    _add("Unknown", s)
            elif isinstance(argv0, str):
                # split naive
                toks = [x.strip() for x in argv0.split()]
                for s in _extract_from_cli_list(toks):
                    _add("Unknown", s)

    # ---------------------------
    # Attribution improvements:
    # - If only FlashInfer is present (imported/called) and models were found only in Unknown
    #   via transformers/hf downloads, attribute those to FlashInfer.
    # - If TensorRT-LLM is imported/called and we only saw models in Unknown from modelish keys,
    #   promote obvious model IDs/paths to TensorRT-LLM.
    # ---------------------------
    has_any_runtime = any(seen_framework_calls[k] for k in ["vLLM", "SGLang", "TensorRT-LLM", "LMDeploy"])
    has_flashinfer = import_flags["flashinfer"] or seen_framework_calls["FlashInfer"]

    # Promote Unknown -> FlashInfer only when FlashInfer is the only serving stack present
    if has_flashinfer and not has_any_runtime:
        unknown_models = list(out_store["Unknown"].keys())
        for s in unknown_models:
            if _looks_like_model_id(s):
                _add("FlashInfer", s)
                # keep in Unknown as well? requirement doesn't forbid duplicates across frameworks,
                # but generally you want a single attribution. We'll remove from Unknown.
                del out_store["Unknown"][s]

    # Promote Unknown -> TensorRT-LLM when TRT is present and Unknown strings look like model ids/paths
    has_trt = import_flags["tensorrt_llm"] or seen_framework_calls["TensorRT-LLM"]
    if has_trt:
        unknown_models = list(out_store["Unknown"].keys())
        for s in unknown_models:
            if _looks_like_model_id(s):
                _add("TensorRT-LLM", s)
                del out_store["Unknown"][s]

    # ---------------------------
    # Finalize output lists
    # ---------------------------
    final = {k: list(out_store[k].keys()) for k in result_keys}
    return final



import ast
import re
from collections import OrderedDict
from typing import Any, List, Optional

import ast
import re
from collections import OrderedDict
from typing import Any, Optional

import ast
import re
from collections import OrderedDict
from typing import Any, Optional


def extract_all_model_identifiers(code: str) -> list[str]:
    """
    STRICT mode: return ONLY concrete model identifiers (paths / HF IDs / URIs / checkpoint files).
    - No VAR:/ENV:/EXPR: placeholders
    - No config file paths (yaml/json/toml)
    - Stronger filtering to avoid grabbing large docstrings or unrelated URLs
    - Deduplicated, first-seen order
    """
    out = OrderedDict()

    def add(s: str) -> None:
        s = s.strip()
        if not s:
            return
        if s not in out:
            out[s] = None

    # --- patterns / heuristics ---
    URI_PREFIXES = ("s3://", "gs://", "hf://", "file://")
    PATH_PREFIXES = ("/", "./", "../")
    WIN_PATH_RE = re.compile(r"^[A-Za-z]:\\")
    ARTIFACT_EXTS = (
        ".engine", ".plan", ".safetensors", ".bin", ".pt", ".pth", ".ckpt",
        ".gguf", ".onnx", ".pb"
    )
    HF_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._/-]*$")

    def looks_like_model_identifier(s: str) -> bool:
        if not isinstance(s, str):
            return False
        t = s.strip()

        # reject multi-line / huge strings (docstrings, code blocks, etc.)
        if "\n" in t or "\r" in t:
            return False
        if len(t) > 300:
            return False

        low = t.lower()

        # checkpoint / engine artifacts
        if any(low.endswith(ext) for ext in ARTIFACT_EXTS):
            return True

        # URIs (non-http)
        if low.startswith(URI_PREFIXES):
            return True

        # local paths
        if t.startswith(PATH_PREFIXES) or WIN_PATH_RE.match(t):
            return True

        # HF model id (org/name[/...])
        if " " not in t and HF_ID_RE.match(t) and "://" not in t:
            return True

        return False

    # --- AST helpers ---
    def extract_constant_str(node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    def flatten_literals(obj: Any) -> None:
        """Collect any string literals inside python literal structures."""
        if isinstance(obj, str):
            if looks_like_model_identifier(obj):
                add(obj)
            return
        if isinstance(obj, (list, tuple, set)):
            for x in obj:
                flatten_literals(x)
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                flatten_literals(k)
                flatten_literals(v)
            return

    def make_hashable(x: Any) -> Any:
        """Convert common unhashables into hashables for dict keys."""
        if isinstance(x, list):
            return tuple(make_hashable(i) for i in x)
        if isinstance(x, set):
            return tuple(sorted(make_hashable(i) for i in x))
        if isinstance(x, dict):
            # dict keys in a key-position is odd; make a stable tuple
            return tuple((make_hashable(k), make_hashable(v)) for k, v in x.items())
        return x  # may still be unhashable (e.g., custom object), we'll guard

    def safe_eval_literal(node: ast.AST, *, for_key: bool = False) -> Any:
        """
        Only evaluate *literal* structures: constants, lists/tuples/sets/dicts.
        If for_key=True, coerce result to something hashable (or return None).
        """
        if isinstance(node, ast.Constant):
            val = node.value
            return make_hashable(val) if for_key else val

        if isinstance(node, ast.List):
            val = [safe_eval_literal(e, for_key=for_key) for e in node.elts]
            return tuple(val) if for_key else val

        if isinstance(node, ast.Tuple):
            val = tuple(safe_eval_literal(e, for_key=for_key) for e in node.elts)
            return val

        if isinstance(node, ast.Set):
            val = [safe_eval_literal(e, for_key=for_key) for e in node.elts]
            return tuple(sorted(val, key=lambda z: repr(z))) if for_key else val

        if isinstance(node, ast.Dict):
            d = {}
            for k_node, v_node in zip(node.keys, node.values):
                if k_node is None:
                    continue
                k = safe_eval_literal(k_node, for_key=True)  # IMPORTANT: key mode
                v = safe_eval_literal(v_node, for_key=False)

                if k is None:
                    continue
                # Skip still-unhashable keys safely
                try:
                    hash(k)
                except TypeError:
                    continue

                d[k] = v
            return d

        # Not a pure literal -> stop
        return None

    # --- parse ---
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # fallback: string-literal scan
        for s in re.findall(r"(?s)(['\"])(.*?)(?<!\\)\1", code):
            txt = s[1]
            if looks_like_model_identifier(txt):
                add(txt)
        return list(out.keys())

    # 1) Collect from literal assignments, dicts, lists, etc.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value_node = node.value
            if value_node is None:
                continue
            lit = safe_eval_literal(value_node)
            if lit is not None:
                flatten_literals(lit)

    # 2) Collect from call arguments (keyword and positional) but only if literal strings/lists/dicts
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        # keyword args
        for kw in node.keywords:
            if kw.arg is None:
                # **kwargs: only if it's a literal dict
                lit = safe_eval_literal(kw.value)
                if isinstance(lit, dict):
                    flatten_literals(lit)
                continue

            lit = safe_eval_literal(kw.value)
            if lit is not None:
                flatten_literals(lit)
            else:
                s = extract_constant_str(kw.value)
                if s and looks_like_model_identifier(s):
                    add(s)

        # positional args
        for arg in node.args:
            lit = safe_eval_literal(arg)
            if lit is not None:
                flatten_literals(lit)
            else:
                s = extract_constant_str(arg)
                if s and looks_like_model_identifier(s):
                    add(s)

    # 3) CLI strings: only extract *tokens* that look like model identifiers
    def scan_cli_string(cmd: str) -> None:
        if "\n" in cmd or len(cmd) > 5000:
            return
        for q in re.findall(r"['\"]([^'\"]+)['\"]", cmd):
            if looks_like_model_identifier(q):
                add(q)
        for tok in cmd.split():
            t = tok.strip("\"' ,")
            if looks_like_model_identifier(t):
                add(t)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            try:
                func_name = ast.unparse(node.func).lower()
            except Exception:
                func_name = ""

            if func_name in {"os.system", "system"} and node.args:
                s = extract_constant_str(node.args[0])
                if s:
                    scan_cli_string(s)

            if func_name in {"subprocess.run", "subprocess.call", "subprocess.popen",
                             "run", "call", "popen"} and node.args:
                arg0 = node.args[0]
                s0 = extract_constant_str(arg0)
                if s0:
                    scan_cli_string(s0)
                else:
                    lit = safe_eval_literal(arg0)
                    if isinstance(lit, list):
                        for item in lit:
                            if isinstance(item, str) and looks_like_model_identifier(item):
                                add(item)

    # 4) Final raw scan fallback for missed literals (still strict)
    for s in re.findall(r"['\"]([^'\"]{1,300})['\"]", code):
        if looks_like_model_identifier(s):
            add(s)

    return list(out.keys())

import re
from collections import OrderedDict
from typing import List


# ---- Regex to capture HF-style model IDs anywhere in a string ----
HF_MODEL_IN_TEXT_RE = re.compile(
    r"""
    (?P<org>[A-Za-z0-9][A-Za-z0-9._-]+)/
    (?P<model>
        [A-Za-z0-9][A-Za-z0-9._-]*      # base
        (?:                            # must contain a model-like token
            .*?
            (?:\d+(?:\.\d+)?[BM]|chat|instruct|llama|vicuna|qwen|deepseek|mistral|falcon|gemma)
            .*?
        )
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# File extensions that are NOT models
BAD_EXTENSIONS = (
    ".json", ".jsonl", ".yaml", ".yml", ".txt",
    ".engine", ".bin", ".pt", ".ckpt", ".safetensors"
)

# Folder keywords that usually mean "not a model"
BAD_PATH_KEYWORDS = {
    "dataset", "datasets", "output", "outputs", "example", "examples",
    "cache", "caches", "tmp", "temp", "logs", "log"
}


def clean_model_names(raw_models: List[str]) -> List[str]:
    """
    Cleans extracted model identifiers and returns only valid model names.
    """

    results = OrderedDict()

    for item in raw_models:
        if not isinstance(item, str):
            continue

        s = item.strip()

        if not s:
            continue

        lower = s.lower()

        # ❌ Drop obvious trash paths
        if lower.endswith(BAD_EXTENSIONS):
            continue

        if lower.startswith(("datasets/", "./", "../")):
            continue

        if any(k in lower.split("/") for k in BAD_PATH_KEYWORDS):
            continue

        # 🔍 Extract HF model ID anywhere in the string
        match = HF_MODEL_IN_TEXT_RE.search(s)
        if not match:
            continue

        model_name = match.group("model")

        # Final sanity check: must contain size or LLM keyword
        if not re.search(r"\d|llama|vicuna|qwen|deepseek|mistral|chat|instruct", model_name, re.IGNORECASE):
            continue

        results[model_name] = None

    return list(results.keys())


def clean_model_dict(model_dict: dict) -> dict:
    """
    Apply clean_model_names to each framework list inside the model dictionary.
    Preserves the same keys.
    """
    if not isinstance(model_dict, dict):
        return model_dict

    cleaned = {}

    for framework, models in model_dict.items():
        if isinstance(models, list):
            cleaned[framework] = clean_model_names(models)
        else:
            cleaned[framework] = models

    return cleaned



import re
import html
import unicodedata
from typing import Optional, List, Tuple
import pandas as pd

def preprocess_github_text(text: Optional[str]) -> str:
    """
    Aggressive README/About cleaner tuned for GitHub repos (unsupervised clustering).
    Removes:
      - HTML blocks (img/svg/table/details), HTML tags, comments
      - markdown images/badges, link-bars, navigation/ToC blocks
      - fenced + indented code blocks, inline code, path trees, env/config blocks
      - markdown + HTML tables, dataset catalogs
      - common non-discriminative sections (install/usage/setup/training/etc.)
      - citations/bibtex/reference footnotes/license boilerplate
      - non-English content (doc-level + segment-level heuristics)
      - duplicated repeated lines/paragraphs

    Returns: cleaned lowercase English text (may be empty).
    """

    # ---------------- Robust input handling ----------------
    if text is None:
        return ""
    if isinstance(text, float) and pd.isna(text):
        return ""
    if pd.isna(text):
        return ""
    if not isinstance(text, str):
        text = str(text)
    if text.strip().lower() in {"nan", "none", "null"}:
        return ""

    # ---------------- Helpers ----------------
    def _normalize(s: str) -> str:
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        s = html.unescape(s)
        s = unicodedata.normalize("NFKC", s)
        # remove control chars (keep \n)
        s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", s)
        return s

    def _drop_html_comments(s: str) -> str:
        return re.sub(r"(?is)<!--.*?-->", " ", s)

    def _drop_specific_html_blocks(s: str) -> str:
        # Drop blocks that are mostly layout/graphics and can be huge
        s = re.sub(r"(?is)<details.*?>.*?</details>", " ", s)
        s = re.sub(r"(?is)<table.*?>.*?</table>", " ", s)
        s = re.sub(r"(?is)<svg.*?>.*?</svg>", " ", s)
        s = re.sub(r"(?is)<img\b[^>]*>", " ", s)
        s = re.sub(r"(?is)<script.*?>.*?</script>", " ", s)
        s = re.sub(r"(?is)<style.*?>.*?</style>", " ", s)
        return s

    def _strip_html_tags_keep_text(s: str) -> str:
        return re.sub(r"(?is)<[^>]+>", " ", s)

    def _remove_fenced_code_blocks(s: str) -> str:
        s = re.sub(r"(?is)```.*?```", " ", s)
        s = re.sub(r"(?is)~~~.*?~~~", " ", s)
        return s

    def _remove_indented_code_blocks(s: str) -> str:
        lines = s.split("\n")
        out = []
        for line in lines:
            if re.match(r"^(?:\t| {4,})", line):
                continue
            out.append(line)
        return "\n".join(out)

    def _remove_inline_code(s: str) -> str:
        return re.sub(r"`[^`\n]+`", " ", s)

    def _remove_markdown_images(s: str) -> str:
        s = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", s)
        s = re.sub(r"!\[[^\]]*\]\[[^\]]+\]", " ", s)
        return s

    def _replace_md_links_keep_text(s: str) -> str:
        s = re.sub(r"\[([^\]]+)\]\((?:[^)]+)\)", r"\1", s)
        s = re.sub(r"\[([^\]]+)\]\[[^\]]+\]", r"\1", s)
        return s

    def _remove_urls_emails(s: str) -> str:
        s = re.sub(r"(?i)\b(?:https?://|www\.)\S+\b", " ", s)
        s = re.sub(r"(?i)\b[\w.\-+]+@[\w.\-]+\.\w+\b", " ", s)
        return s

    def _remove_badges_and_shields(s: str) -> str:
        badge_re = re.compile(
            r"(?i)(shields\.io|badge\.fury\.io|travis-ci|circleci|codecov|coveralls|appveyor|"
            r"github\.com/.*/workflows/|actions/workflows/|colab-badge|pepy\.tech|"
            r"pypi/pyversions|downloads|open\s+source\s+badge)"
        )
        lines = s.split("\n")
        out = []
        for line in lines:
            if badge_re.search(line):
                continue
            # common badge markdown pattern
            if re.search(r"\[\!\[[^\]]*\]\([^)]+\)\]\([^)]+\)", line) and len(line.strip()) < 300:
                continue
            out.append(line)
        return "\n".join(out)

    def _remove_link_bar_lines(s: str) -> str:
        # Lines with many links or emoji separators: "🤗 ... | 🤖 ... | ..."
        lines = s.split("\n")
        out = []
        for line in lines:
            if line.count("http") >= 2 or line.count("|") >= 3:
                # keep if it's a normal sentence, drop if it's basically a resource bar
                if len(re.findall(r"[A-Za-z]{3,}", line)) <= 6:
                    continue
            out.append(line)
        return "\n".join(out)

    def _remove_markdown_tables_and_pipe_catalogs(s: str) -> str:
        # Remove strict markdown tables + loose pipe-heavy catalogs
        lines = s.split("\n")
        out = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # strict table header separator
            if "|" in line and i + 1 < len(lines):
                sep = lines[i + 1]
                if re.match(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$", sep):
                    i += 2
                    while i < len(lines) and "|" in lines[i]:
                        i += 1
                    continue
            # loose: very pipe-heavy (dataset catalogs)
            if line.count("|") >= 5:
                i += 1
                continue
            out.append(line)
            i += 1
        return "\n".join(out)

    def _remove_reference_footnotes(s: str) -> str:
        return re.sub(r"(?im)^\s*\[[^\]]+\]\s*:\s*\S+.*$", " ", s)

    def _remove_bibtex_like_blocks(s: str) -> str:
        # already removes fenced code; this is for inline @inproceedings blobs not fenced
        return re.sub(r"(?is)@\w+\s*\{.*?\}\s*", " ", s)

    def _remove_path_tree_blocks(s: str) -> str:
        # remove directory trees and path-only listings
        lines = s.split("\n")
        out = []
        tree_markers = ("├", "└", "│", "──", "…")
        for line in lines:
            if any(m in line for m in tree_markers) and len(line.strip()) < 200:
                continue
            if re.fullmatch(r"\s*[/\w.\-]+/?\s*", line) and "/" in line and len(line.strip()) < 80:
                continue
            out.append(line)
        return "\n".join(out)

    def _remove_env_and_kv_blocks(s: str) -> str:
        # drop clusters of KEY=VALUE lines (often .env docs)
        lines = s.split("\n")
        out = []
        buf: List[str] = []
        def flush_buf():
            nonlocal buf
            if not buf:
                return []
            kv = sum(1 for x in buf if re.match(r"^\s*[A-Z0-9_]{2,}\s*=\s*.+\s*$", x))
            if kv >= 3:  # likely env/config
                buf = []
                return []
            keep = buf
            buf = []
            return keep

        for line in lines:
            if line.strip() == "":
                out.extend(flush_buf())
                out.append("")
                continue
            if re.match(r"^\s*[A-Z0-9_]{2,}\s*=\s*.+\s*$", line):
                buf.append(line)
                continue
            out.extend(flush_buf())
            out.append(line)
        out.extend(flush_buf())
        return "\n".join(out)

    def _remove_markdown_toc_blocks(s: str) -> str:
        # Remove "Table of Contents" sections and pure anchor nav lists
        lines = s.split("\n")
        out = []
        i = 0
        while i < len(lines):
            line = lines[i].strip().lower()
            if re.match(r"^(#+\s*)?table of contents\b", line):
                i += 1
                # drop following bullet/number lists of links
                while i < len(lines):
                    ln = lines[i].strip()
                    if ln.startswith(("*", "-", "+")) and re.search(r"\[[^\]]+\]\(#", ln):
                        i += 1
                        continue
                    if re.match(r"^\d+\.\s+\[[^\]]+\]\(#", ln):
                        i += 1
                        continue
                    # stop when list ends
                    break
                continue

            # Also drop standalone nav blocks near top: many "[X](#y)" lines
            if i < 80:
                # detect a run of >=4 anchor list lines
                j = i
                cnt = 0
                while j < len(lines):
                    ln = lines[j].strip()
                    if (ln.startswith(("*", "-", "+")) and re.search(r"\[[^\]]+\]\(#", ln)) or re.match(r"^\d+\.\s+\[[^\]]+\]\(#", ln):
                        cnt += 1
                        j += 1
                        continue
                    break
                if cnt >= 4:
                    i = j
                    continue

            out.append(lines[i])
            i += 1
        return "\n".join(out)

    def _remove_sections_by_header(s: str) -> str:
        # Normalize underline headers
        s = re.sub(r"(?m)^(.+)\n=+\s*$", r"# \1", s)
        s = re.sub(r"(?m)^(.+)\n-+\s*$", r"## \1", s)

        # lots of github readmes prefix emojis like ":beers:" or "✅"
        def norm_title(t: str) -> str:
            t = t.lower()
            t = re.sub(r":[a-z0-9_+\-]+:", " ", t)  # :beers:
            t = re.sub(r"[^\w\s\-+/]", " ", t)
            t = re.sub(r"\s+", " ", t).strip()
            return t

        REMOVE_HEADERS = [
            # instructions
            "installation", "install", "setup", "quick start", "getting started", "requirements",
            "usage", "how to use", "running", "run", "build", "compile", "configuration", "config",
            "training", "fine tuning", "finetuning", "sft", "grpo", "evaluation", "inference",
            "dataset preparation", "data", "examples", "tutorial", "tutorials", "benchmark", "benchmarks",
            "results", "update", "roadmap", "todo", "todo list", "changelog", "release", "releases",
            "faq", "troubleshooting", "support", "help", "contributing", "development",
            # legal/meta
            "license", "licence", "citation", "cite", "references", "acknowledgement", "acknowledgement s",
            "acknowledgements", "authors", "contributors", "collaboration", "disclaimer", "contact",
        ]

        header_re = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")
        matches = list(header_re.finditer(s))
        if not matches:
            return s

        remove_spans: List[Tuple[int, int]] = []
        for idx, m in enumerate(matches):
            level = len(m.group(1))
            title = norm_title(m.group(2))

            if any(h == title or h in title for h in REMOVE_HEADERS):
                start = m.start()
                end = len(s)
                for j in range(idx + 1, len(matches)):
                    next_level = len(matches[j].group(1))
                    if next_level <= level:
                        end = matches[j].start()
                        break
                remove_spans.append((start, end))

        if not remove_spans:
            return s

        remove_spans.sort()
        merged = []
        cur_s, cur_e = remove_spans[0]
        for a, b in remove_spans[1:]:
            if a <= cur_e:
                cur_e = max(cur_e, b)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = a, b
        merged.append((cur_s, cur_e))

        out = s
        for a, b in reversed(merged):
            out = out[:a] + "\n" + out[b:]
        return out

    def _remove_license_boilerplate_lines(s: str) -> str:
        lic = re.compile(
            r"(?i)\b(the\s+mit\s+license|apache\s+license|gnu\s+general\s+public\s+license|bsd\s+license|"
            r"copyright\s+\(c\)|permission\s+is\s+hereby\s+granted|"
            r"without\s+warranty|as\s+is\s+provided|limitations\s+under\s+the\s+license)\b"
        )
        return "\n".join([ln for ln in s.split("\n") if not lic.search(ln)])

    def _dedup_repeated_lines(s: str) -> str:
        # remove immediate duplicates and very common repeated headings
        lines = [ln.strip() for ln in s.split("\n")]
        out = []
        prev = None
        for ln in lines:
            if not ln:
                if out and out[-1] != "":
                    out.append("")
                prev = ""
                continue
            if ln == prev:
                continue
            out.append(ln)
            prev = ln
        return "\n".join(out).strip()

    # ---------------- English filtering (doc-level + segment-level) ----------------
    EN_STOPWORDS = {
        "the","and","or","to","of","in","for","on","with","as","is","are","be","this","that","it",
        "from","by","an","a","at","we","you","your","our","can","will","not","use","using",
        "library","framework","tool","project","package","module","repo","repository"
    }

    def _ratio_non_latin(s: str) -> float:
        letters = [ch for ch in s if ch.isalpha()]
        if not letters:
            return 0.0
        non_latin = sum(1 for ch in letters if ord(ch) > 0x024F)
        return non_latin / len(letters)

    def _english_likeness(s: str) -> float:
        s = s.strip()
        if len(s) < 12:
            return 0.0
        non_latin = _ratio_non_latin(s)
        if non_latin >= 0.35:
            return 0.0
        words = re.findall(r"[A-Za-z]+", s.lower())
        if len(words) < 3:
            return 0.0
        stop_hits = sum(1 for w in words if w in EN_STOPWORDS)
        stop_ratio = stop_hits / max(len(words), 1)
        visible = [ch for ch in s if not ch.isspace()]
        ascii_ratio = sum(1 for ch in visible if ord(ch) < 128) / max(len(visible), 1)
        score = 0.55 * ascii_ratio + 0.45 * min(1.0, stop_ratio * 2.0)
        score *= (1.0 - min(non_latin, 0.30))
        return max(0.0, min(1.0, score))

    def _doc_is_mostly_english(s: str) -> bool:
        # If doc is predominantly non-Latin, drop it entirely.
        # (Fixes Korean README: returns empty instead of keeping tiny English scraps)
        non_latin = _ratio_non_latin(s)
        if non_latin >= 0.45:
            return False
        # need some English words overall
        words = re.findall(r"[A-Za-z]+", s)
        if len(words) < 8:
            return False
        return True

    def _keep_english_segments(s: str, threshold: float = 0.45) -> str:
        # split by lines; further split long lines into sentences
        parts: List[str] = []
        for line in s.split("\n"):
            line = line.strip()
            if not line:
                continue
            if len(line) > 220:
                parts.extend([x.strip() for x in re.split(r"(?<=[.!?])\s+", line) if x.strip()])
            else:
                parts.append(line)
        kept = [p for p in parts if _english_likeness(p) >= threshold]
        return "\n".join(kept).strip()

    def _final_cleanup(s: str) -> str:
        s = re.sub(r"(?m)^\s{0,3}>\s+", "", s)         # blockquotes markers
        s = re.sub(r"(?m)^\s*[-*+]\s+", "", s)         # bullets
        s = re.sub(r"(?m)^\s*\d+\.\s+", "", s)         # numbered lists
        s = re.sub(r"[_*~]{1,3}", " ", s)              # emphasis
        s = re.sub(r"(?m)^\s*#{1,6}\s*", "", s)        # header hashes
        s = re.sub(r"[ \t]{2,}", " ", s)
        # remove leftover markdown anchor refs like "(#section)" alone
        s = re.sub(r"\(#[-\w]+\)", " ", s)
        lines = [ln.strip() for ln in s.split("\n")]
        lines = [ln for ln in lines if ln and not re.fullmatch(r"[\W_]+", ln)]
        s = "\n".join(lines)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip()

    # ---------------- Pipeline ----------------
    t = _normalize(text)

    t = _drop_html_comments(t)
    t = _drop_specific_html_blocks(t)
    t = _strip_html_tags_keep_text(t)

    t = _remove_badges_and_shields(t)
    t = _remove_link_bar_lines(t)

    t = _remove_fenced_code_blocks(t)
    t = _remove_indented_code_blocks(t)
    t = _remove_inline_code(t)

    t = _remove_markdown_images(t)
    t = _replace_md_links_keep_text(t)
    t = _remove_urls_emails(t)

    t = _remove_markdown_toc_blocks(t)
    t = _remove_markdown_tables_and_pipe_catalogs(t)
    t = _remove_reference_footnotes(t)
    t = _remove_bibtex_like_blocks(t)
    t = _remove_path_tree_blocks(t)
    t = _remove_env_and_kv_blocks(t)

    t = _remove_sections_by_header(t)
    t = _remove_license_boilerplate_lines(t)

    t = _dedup_repeated_lines(t)
    t = _final_cleanup(t)

    # doc-level language gate, then segment-level filter
    if not _doc_is_mostly_english(t):
        return ""
    t = _keep_english_segments(t, threshold=0.45)

    return t.lower().strip()



import re
import html
import unicodedata
from typing import Optional, List, Tuple, Dict

import pandas as pd


def preprocess_github_text_mode(text: Optional[str], mode: str = "cluster") -> str:
    """
    Two-mode GitHub README/About cleaner.

    Modes
    -----
    mode="cluster"  : aggressive; keeps mostly "what is it" / high-level description; removes most how-to sections.
    mode="keywords" : less aggressive; keeps more topical content (features, tasks, methods); still removes code, tables,
                      badges, huge navigation, etc.

    Returns
    -------
    cleaned lowercase English text (may be empty).

    Notes
    -----
    - Handles NaN/None/non-string values safely.
    - Removes non-English documents (doc-level) and also filters non-English segments (segment-level).
    - Designed to be deterministic and reproducible.
    """

    # ---------------- Robust input handling ----------------
    if text is None:
        return ""
    if isinstance(text, float) and pd.isna(text):
        return ""
    try:
        if pd.isna(text):
            return ""
    except Exception:
        pass
    if not isinstance(text, str):
        text = str(text)
    if text.strip().lower() in {"nan", "none", "null"}:
        return ""

    mode = (mode or "cluster").strip().lower()
    if mode not in {"cluster", "keywords"}:
        raise ValueError('mode must be either "cluster" or "keywords"')

    # ---------------- Mode knobs ----------------
    # "cluster" removes more sections and filters English segments more strictly.
    CFG: Dict[str, object] = {
        "cluster": {
            "english_seg_threshold": 0.48,
            "drop_toc": True,
            "drop_sections": True,          # drop common how-to/meta sections by header
            "drop_install_usage": True,     # extra strict list
            "drop_benchmark_tables": True,
            "keep_details_html": False,     # drop <details>...</details>
            "keep_tables_html": False,      # drop <table>...</table>
            "keep_long_lists": False,       # drop long bullet runs
            "max_line_len_keep": 240,       # aggressive
        },
        "keywords": {
            "english_seg_threshold": 0.40,
            "drop_toc": True,
            "drop_sections": True,
            "drop_install_usage": False,    # keep more (features/tasks/methods) even if under common headers
            "drop_benchmark_tables": True,
            "keep_details_html": True,      # keep text inside details (but code still removed)
            "keep_tables_html": False,      # still drop tables (they are noisy)
            "keep_long_lists": True,
            "max_line_len_keep": 420,
        },
    }[mode]

    # ---------------- Helpers ----------------
    def _normalize(s: str) -> str:
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        s = html.unescape(s)
        s = unicodedata.normalize("NFKC", s)
        s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", s)  # control chars, keep \n
        return s

    def _drop_html_comments(s: str) -> str:
        return re.sub(r"(?is)<!--.*?-->", " ", s)

    def _drop_specific_html_blocks(s: str) -> str:
        # Drop heavy layout blocks
        if not CFG["keep_details_html"]:
            s = re.sub(r"(?is)<details.*?>.*?</details>", " ", s)
        if not CFG["keep_tables_html"]:
            s = re.sub(r"(?is)<table.*?>.*?</table>", " ", s)
        s = re.sub(r"(?is)<svg.*?>.*?</svg>", " ", s)
        s = re.sub(r"(?is)<img\b[^>]*>", " ", s)
        s = re.sub(r"(?is)<script.*?>.*?</script>", " ", s)
        s = re.sub(r"(?is)<style.*?>.*?</style>", " ", s)
        return s

    def _strip_html_tags_keep_text(s: str) -> str:
        return re.sub(r"(?is)<[^>]+>", " ", s)

    def _remove_fenced_code_blocks(s: str) -> str:
        s = re.sub(r"(?is)```.*?```", " ", s)
        s = re.sub(r"(?is)~~~.*?~~~", " ", s)
        return s

    def _remove_indented_code_blocks(s: str) -> str:
        lines = s.split("\n")
        out = []
        for line in lines:
            if re.match(r"^(?:\t| {4,})", line):
                continue
            out.append(line)
        return "\n".join(out)

    def _remove_inline_code(s: str) -> str:
        return re.sub(r"`[^`\n]+`", " ", s)

    def _remove_markdown_images(s: str) -> str:
        s = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", s)
        s = re.sub(r"!\[[^\]]*\]\[[^\]]+\]", " ", s)
        return s

    def _replace_md_links_keep_text(s: str) -> str:
        s = re.sub(r"\[([^\]]+)\]\((?:[^)]+)\)", r"\1", s)
        s = re.sub(r"\[([^\]]+)\]\[[^\]]+\]", r"\1", s)
        return s

    def _remove_urls_emails(s: str) -> str:
        s = re.sub(r"(?i)\b(?:https?://|www\.)\S+\b", " ", s)
        s = re.sub(r"(?i)\b[\w.\-+]+@[\w.\-]+\.\w+\b", " ", s)
        return s

    def _remove_badges_and_shields(s: str) -> str:
        badge_re = re.compile(
            r"(?i)(shields\.io|badge\.fury\.io|travis-ci|circleci|codecov|coveralls|appveyor|"
            r"github\.com/.*/workflows/|actions/workflows/|colab-badge|pepy\.tech|"
            r"pypi/pyversions|downloads|open\s+source\s+badge)"
        )
        lines = s.split("\n")
        out = []
        for line in lines:
            if badge_re.search(line):
                continue
            if re.search(r"\[\!\[[^\]]*\]\([^)]+\)\]\([^)]+\)", line) and len(line.strip()) < 350:
                continue
            out.append(line)
        return "\n".join(out)

    def _remove_link_bar_lines(s: str) -> str:
        # Lines with many links or separators: "🤗 ... | 🤖 ... | ..."
        lines = s.split("\n")
        out = []
        for line in lines:
            if line.count("http") >= 2 or line.count("|") >= 3:
                # drop if mostly a resource bar
                if len(re.findall(r"[A-Za-z]{3,}", line)) <= 8:
                    continue
            out.append(line)
        return "\n".join(out)

    def _remove_markdown_tables_and_pipe_catalogs(s: str) -> str:
        lines = s.split("\n")
        out = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # strict markdown table
            if "|" in line and i + 1 < len(lines):
                sep = lines[i + 1]
                if re.match(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$", sep):
                    i += 2
                    while i < len(lines) and "|" in lines[i]:
                        i += 1
                    continue
            # loose pipe-heavy catalogs
            if "|" in line and line.count("|") >= 5:
                i += 1
                continue
            out.append(line)
            i += 1
        return "\n".join(out)

    def _remove_reference_footnotes(s: str) -> str:
        return re.sub(r"(?im)^\s*\[[^\]]+\]\s*:\s*\S+.*$", " ", s)

    def _remove_bibtex_like_blocks(s: str) -> str:
        return re.sub(r"(?is)@\w+\s*\{.*?\}\s*", " ", s)

    def _remove_path_tree_blocks(s: str) -> str:
        lines = s.split("\n")
        out = []
        tree_markers = ("├", "└", "│", "──")
        for line in lines:
            if any(m in line for m in tree_markers) and len(line.strip()) < 260:
                continue
            if re.fullmatch(r"\s*[/\w.\-]+/?\s*", line) and "/" in line and len(line.strip()) < 90:
                continue
            out.append(line)
        return "\n".join(out)

    def _remove_env_and_kv_blocks(s: str) -> str:
        # drop clusters of KEY=VALUE lines
        lines = s.split("\n")
        out = []
        buf: List[str] = []

        def flush_buf() -> List[str]:
            nonlocal buf
            if not buf:
                return []
            kv = sum(1 for x in buf if re.match(r"^\s*[A-Z0-9_]{2,}\s*=\s*.+\s*$", x))
            # in keywords mode we still drop big env dumps
            if kv >= (3 if mode == "cluster" else 4):
                buf = []
                return []
            keep = buf
            buf = []
            return keep

        for line in lines:
            if line.strip() == "":
                out.extend(flush_buf())
                out.append("")
                continue
            if re.match(r"^\s*[A-Z0-9_]{2,}\s*=\s*.+\s*$", line):
                buf.append(line)
                continue
            out.extend(flush_buf())
            out.append(line)
        out.extend(flush_buf())
        return "\n".join(out)

    def _remove_markdown_toc_blocks(s: str) -> str:
        if not CFG["drop_toc"]:
            return s

        lines = s.split("\n")
        out = []
        i = 0
        while i < len(lines):
            line_l = lines[i].strip().lower()
            if re.match(r"^(#+\s*)?table of contents\b", line_l):
                i += 1
                while i < len(lines):
                    ln = lines[i].strip()
                    if (ln.startswith(("*", "-", "+")) and re.search(r"\[[^\]]+\]\(#", ln)) or re.match(
                        r"^\d+\.\s+\[[^\]]+\]\(#", ln
                    ):
                        i += 1
                        continue
                    break
                continue

            # drop big anchor-nav blocks near top
            if i < 120:
                j = i
                cnt = 0
                while j < len(lines):
                    ln = lines[j].strip()
                    if (ln.startswith(("*", "-", "+")) and re.search(r"\[[^\]]+\]\(#", ln)) or re.match(
                        r"^\d+\.\s+\[[^\]]+\]\(#", ln
                    ):
                        cnt += 1
                        j += 1
                        continue
                    break
                if cnt >= 4:
                    i = j
                    continue

            out.append(lines[i])
            i += 1
        return "\n".join(out)

    def _remove_long_bullet_runs(s: str) -> str:
        # In cluster mode, long bullet runs (features/todo lists) can dominate embeddings.
        # In keywords mode, keep them.
        if CFG["keep_long_lists"]:
            return s

        lines = s.split("\n")
        out = []
        i = 0
        while i < len(lines):
            # detect run of bullets
            j = i
            cnt = 0
            while j < len(lines) and re.match(r"^\s*[-*+]\s+", lines[j]):
                cnt += 1
                j += 1
            if cnt >= 8:
                i = j
                continue
            out.append(lines[i])
            i += 1
        return "\n".join(out)

    def _remove_sections_by_header(s: str) -> str:
        if not CFG["drop_sections"]:
            return s

        # normalize underline headers
        s = re.sub(r"(?m)^(.+)\n=+\s*$", r"# \1", s)
        s = re.sub(r"(?m)^(.+)\n-+\s*$", r"## \1", s)

        def norm_title(t: str) -> str:
            t = t.lower()
            t = re.sub(r":[a-z0-9_+\-]+:", " ", t)   # :beers:
            t = re.sub(r"[^\w\s\-+/]", " ", t)
            t = re.sub(r"\s+", " ", t).strip()
            return t

        # Base removal list (both modes)
        REMOVE_HEADERS_BASE = {
            # meta/legal
            "license", "licence", "citation", "cite", "references", "acknowledgement",
            "acknowledgements", "authors", "contributors", "collaboration", "disclaimer",
            "contact", "contributing", "development", "changelog", "release", "releases", "faq",
            "troubleshooting", "support",
        }

        # Additional for cluster mode (remove how-to sections)
        REMOVE_HEADERS_CLUSTER_EXTRA = {
            "installation", "install", "setup", "quick start", "getting started", "requirements",
            "usage", "how to use", "running", "run", "build", "compile", "configuration", "config",
            "examples", "tutorial", "tutorials",
            "data", "dataset", "datasets", "benchmark", "benchmarks",
            "training", "fine tuning", "finetuning", "evaluation", "inference",
            "results", "update", "roadmap", "todo", "todo list",
            "testing",
        }

        header_re = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")
        matches = list(header_re.finditer(s))
        if not matches:
            return s

        remove_spans: List[Tuple[int, int]] = []

        for idx, m in enumerate(matches):
            level = len(m.group(1))
            title = norm_title(m.group(2))

            remove_headers = set(REMOVE_HEADERS_BASE)
            if mode == "cluster":
                remove_headers |= REMOVE_HEADERS_CLUSTER_EXTRA

            # In keywords mode, do NOT auto-remove many technical sections.
            # We still remove meta/legal + truly generic headings.
            # But we keep sections like "training", "evaluation", etc. because they carry keywords.
            if mode == "keywords" and not CFG["drop_install_usage"]:
                # Keep many potentially keyword-rich sections by not including them
                pass

            should_remove = False

            # Exact match or contains patterns (handles "## :beers: Installation")
            for h in remove_headers:
                if title == h or f" {h} " in f" {title} " or title.startswith(h + " ") or title.endswith(" " + h):
                    should_remove = True
                    break

            # extra: "install via pip", "pip install", etc. (cluster only)
            if mode == "cluster" and re.search(r"\b(pip install|conda|apt-get|brew install|docker)\b", title):
                should_remove = True

            if not should_remove:
                continue

            start = m.start()
            end = len(s)
            for j in range(idx + 1, len(matches)):
                next_level = len(matches[j].group(1))
                if next_level <= level:
                    end = matches[j].start()
                    break
            remove_spans.append((start, end))

        if not remove_spans:
            return s

        remove_spans.sort()
        merged = []
        cur_s, cur_e = remove_spans[0]
        for a, b in remove_spans[1:]:
            if a <= cur_e:
                cur_e = max(cur_e, b)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = a, b
        merged.append((cur_s, cur_e))

        out = s
        for a, b in reversed(merged):
            out = out[:a] + "\n" + out[b:]
        return out

    def _remove_license_boilerplate_lines(s: str) -> str:
        lic = re.compile(
            r"(?i)\b(the\s+mit\s+license|apache\s+license|gnu\s+general\s+public\s+license|bsd\s+license|"
            r"copyright\s+\(c\)|permission\s+is\s+hereby\s+granted|"
            r"without\s+warranty|as\s+is\s+provided|limitations\s+under\s+the\s+license)\b"
        )
        return "\n".join([ln for ln in s.split("\n") if not lic.search(ln)])

    def _dedup_repeated_lines(s: str) -> str:
        lines = [ln.strip() for ln in s.split("\n")]
        out = []
        prev = None
        for ln in lines:
            if not ln:
                if out and out[-1] != "":
                    out.append("")
                prev = ""
                continue
            if ln == prev:
                continue
            out.append(ln)
            prev = ln
        return "\n".join(out).strip()

    # ---------------- English filtering ----------------
    EN_STOPWORDS = {
        "the", "and", "or", "to", "of", "in", "for", "on", "with", "as", "is", "are", "be", "this", "that", "it",
        "from", "by", "an", "a", "at", "we", "you", "your", "our", "can", "will", "not", "use", "using",
        "library", "framework", "tool", "project", "package", "module", "repo", "repository",
    }

    def _ratio_non_latin(s: str) -> float:
        letters = [ch for ch in s if ch.isalpha()]
        if not letters:
            return 0.0
        non_latin = sum(1 for ch in letters if ord(ch) > 0x024F)
        return non_latin / len(letters)

    def _english_likeness(s: str) -> float:
        s = s.strip()
        if len(s) < 12:
            return 0.0
        non_latin = _ratio_non_latin(s)
        if non_latin >= 0.35:
            return 0.0
        words = re.findall(r"[A-Za-z]+", s.lower())
        if len(words) < 3:
            return 0.0
        stop_hits = sum(1 for w in words if w in EN_STOPWORDS)
        stop_ratio = stop_hits / max(len(words), 1)
        visible = [ch for ch in s if not ch.isspace()]
        ascii_ratio = sum(1 for ch in visible if ord(ch) < 128) / max(len(visible), 1)
        score = 0.55 * ascii_ratio + 0.45 * min(1.0, stop_ratio * 2.0)
        score *= (1.0 - min(non_latin, 0.30))
        return max(0.0, min(1.0, score))

    def _doc_is_mostly_english(s: str) -> bool:
        non_latin = _ratio_non_latin(s)
        if non_latin >= 0.45:
            return False
        words = re.findall(r"[A-Za-z]+", s)
        if len(words) < 8:
            return False
        return True

    def _keep_english_segments(s: str, threshold: float) -> str:
        max_len = int(CFG["max_line_len_keep"])
        parts: List[str] = []
        for line in s.split("\n"):
            line = line.strip()
            if not line:
                continue
            if len(line) > max_len:
                parts.extend([x.strip() for x in re.split(r"(?<=[.!?])\s+", line) if x.strip()])
            else:
                parts.append(line)
        kept = [p for p in parts if _english_likeness(p) >= threshold]
        return "\n".join(kept).strip()

    def _final_cleanup(s: str) -> str:
        # remove markdown clutter
        s = re.sub(r"(?m)^\s{0,3}>\s+", "", s)          # blockquotes marker
        s = re.sub(r"(?m)^\s*[-*+]\s+", "", s)          # bullets marker
        s = re.sub(r"(?m)^\s*\d+\.\s+", "", s)          # numbered lists marker
        s = re.sub(r"[_*~]{1,3}", " ", s)               # emphasis
        s = re.sub(r"(?m)^\s*#{1,6}\s*", "", s)         # header hashes
        s = re.sub(r"\(#[-\w]+\)", " ", s)              # leftover anchor refs
        # collapse spaces
        s = re.sub(r"[ \t]{2,}", " ", s)
        lines = [ln.strip() for ln in s.split("\n")]
        # remove empty and pure-symbol lines
        lines = [ln for ln in lines if ln and not re.fullmatch(r"[\W_]+", ln)]
        s = "\n".join(lines)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip()

    # ---------------- Pipeline ----------------
    t = _normalize(text)

    t = _drop_html_comments(t)
    t = _drop_specific_html_blocks(t)
    t = _strip_html_tags_keep_text(t)

    t = _remove_badges_and_shields(t)
    t = _remove_link_bar_lines(t)

    t = _remove_fenced_code_blocks(t)
    t = _remove_indented_code_blocks(t)
    t = _remove_inline_code(t)

    t = _remove_markdown_images(t)
    t = _replace_md_links_keep_text(t)
    t = _remove_urls_emails(t)

    t = _remove_markdown_toc_blocks(t)

    # Tables are noise for both modes
    if CFG["drop_benchmark_tables"]:
        t = _remove_markdown_tables_and_pipe_catalogs(t)

    t = _remove_reference_footnotes(t)
    t = _remove_bibtex_like_blocks(t)

    t = _remove_path_tree_blocks(t)
    t = _remove_env_and_kv_blocks(t)

    # Optional: in cluster mode, drop long bullet runs to avoid list-dominance
    t = _remove_long_bullet_runs(t)

    # Drop sections by header (aggressiveness depends on mode)
    t = _remove_sections_by_header(t)

    # Drop obvious license boilerplate lines
    t = _remove_license_boilerplate_lines(t)

    # Dedup and cleanup
    t = _dedup_repeated_lines(t)
    t = _final_cleanup(t)

    # English filters
    if not _doc_is_mostly_english(t):
        return ""
    t = _keep_english_segments(t, threshold=float(CFG["english_seg_threshold"]))

    return t.lower().strip()



import json
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT


# --------- small deterministic lexicons (extend freely) ----------
MODALITY_LEXICON: Dict[str, List[str]] = {
    "text": ["nlp", "text", "document", "qa", "question answering", "chat", "token", "language model", "llm"],
    "image": ["image", "vision", "cv", "segmentation", "detection", "object detection", "classification", "ocr"],
    "audio": ["audio", "speech", "asr", "tts", "voice", "speaker", "sound"],
    "video": ["video", "frame", "tracking", "action recognition"],
    "tabular": ["tabular", "csv", "dataframe", "spreadsheet", "sql", "relational", "table"],
    "time_series": ["time series", "timeseries", "forecast", "forecasting", "temporal"],
    "graph": ["graph", "gnn", "node", "edge", "network analysis", "knowledge graph", "kg"],
    "code": ["code", "compiler", "program", "programming", "static analysis", "benchmark", "cuda", "kernel"],
    "multimodal": ["multimodal", "vision-language", "vlm", "text-to-image", "text-to-video"],
}

TECHNIQUE_LEXICON: Dict[str, List[str]] = {
    "LLM": ["llm", "large language model", "gpt", "llama", "mistral", "qwen", "deepseek", "codellama"],
    "RAG": ["rag", "retrieval augmented", "retrieval-augmented", "vector database", "vectordb", "faiss", "chroma", "milvus", "weaviate"],
    "retrieval": ["bm25", "dense retrieval", "sparse retrieval", "rerank", "reranking", "ranker", "information retrieval"],
    "embeddings": ["embedding", "sentence-transformers", "sbert", "e5", "bge", "instructor"],
    "fine_tuning": ["finetune", "fine-tune", "lora", "qlora", "peft", "adapter", "sft"],
    "prompting": ["prompt", "prompting", "few-shot", "chain of thought", "cot"],
    "rl": ["reinforcement learning", "ppo", "dpo", "rlhf"],
    "evaluation_benchmark": ["benchmark", "leaderboard", "evaluation", "metrics", "dataset suite"],
    "deployment_serving": ["inference", "serving", "vllm", "tensorrt", "triton", "onnx", "quantization", "latency"],
}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
WS_RE = re.compile(r"\s+")


def make_repo_tag_extractor(
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    seed: int = 42,
):
    """
    Factory that returns a single function you can apply to a pandas column.

    It caches models in the closure so you don't reload on every .apply().
    Output is ONE json string (store in one df column).
    """
    rng = np.random.default_rng(seed)
    embedder = SentenceTransformer(embed_model_name)
    kw_model = KeyBERT(model=embedder)

    # A tiny list to reduce junk phrases in keywords
    STOP_PHRASES = {
        "github", "repository", "installation", "getting started", "quick start",
        "license", "contributing", "readme", "documentation", "example", "examples",
        "dataset", "datasets", "paper", "citation", "references", "contact",
    }

    def _normalize(s: str) -> str:
        if s is None:
            return ""
        s = str(s)
        s = URL_RE.sub(" ", s)
        s = WS_RE.sub(" ", s).strip()
        return s

    def _lexicon_tags(text: str, lex: Dict[str, List[str]], min_hits: int) -> List[str]:
        t = " " + _normalize(text).lower() + " "
        labels = []
        for label, terms in lex.items():
            hits = 0
            for term in terms:
                term_l = term.lower()
                # word-boundary-ish matching, tolerant of spaces/hyphens
                pattern = r"(?<!\w)" + re.escape(term_l).replace(r"\ ", r"[\s\-]+") + r"(?!\w)"
                if re.search(pattern, t):
                    hits += 1
                if hits >= min_hits:
                    labels.append(label)
                    break
        return labels

    def _pick_keywords(text: str, mode: str) -> List[str]:
        """
        mode:
          - 'cluster'  => fewer, higher precision tags
          - 'keywords' => more, richer tags
        """
        text = _normalize(text)
        if not text:
            return []

        if mode == "cluster":
            top_n = 8
            ngram = (1, 2)
            min_score = 0.28
            use_mmr = True
            diversity = 0.7
        else:  # "keywords"
            top_n = 15
            ngram = (1, 3)
            min_score = 0.18
            use_mmr = True
            diversity = 0.55

        kws = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=ngram,
            stop_words="english",
            use_mmr=use_mmr,
            diversity=diversity,
            top_n=top_n,
        )

        out = []
        for kw, score in kws:
            k = kw.strip().lower()
            if score < min_score:
                continue
            if len(k) < 3:
                continue
            if k in STOP_PHRASES:
                continue
            out.append(k)

        # unique while preserving order
        seen = set()
        uniq = []
        for k in out:
            if k not in seen:
                uniq.append(k)
                seen.add(k)
        return uniq

    def tag_repo_readme(
        readme_text: str,
        mode: str = "cluster",
        *,
        head_sentences: int = 3,
        # if you already pre-weighted the first lines in preprocessing, keep this low
        max_chars: int = 4500,
    ) -> str:
        """
        Apply directly on your df column:
            df["tags_json"] = df["preprocessed_readme"].apply(tag_repo_readme)

        Returns a single JSON string with:
          - modalities
          - techniques
          - keywords (mode-dependent)
          - purpose_hint (first ~N sentences heuristic)
        """
        mode = (mode or "cluster").strip().lower()
        if mode not in {"cluster", "keywords"}:
            mode = "cluster"

        txt = _normalize(readme_text)
        if not txt:
            return json.dumps(
                {"modalities": [], "techniques": [], "keywords": [], "purpose_hint": ""},
                ensure_ascii=False,
            )

        # Keep it fast (1k repos): cap chars
        txt = txt[:max_chars]

        # Purpose hint: first few sentences / lines (works well with preprocessed text too)
        # We avoid nltk; use a simple split that is stable on your preprocessed format.
        # If your preprocessing has newline-separated sentences, this is perfect.
        parts = [p.strip() for p in re.split(r"[.\n]+", txt) if p.strip()]
        purpose_hint = " ".join(parts[:head_sentences])[:600]

        # Aggressiveness: require more hits in "cluster" mode
        min_hits = 2 if mode == "cluster" else 1

        modalities = _lexicon_tags(txt, MODALITY_LEXICON, min_hits=min_hits)
        techniques = _lexicon_tags(txt, TECHNIQUE_LEXICON, min_hits=min_hits)

        # Keywords via local embeddings
        keywords = _pick_keywords(txt, mode=mode)

        # Small cleanup: if 'multimodal' appears, keep it; optionally drop overly specific duplicates
        if "multimodal" in modalities and mode == "cluster":
            # keep multimodal + at most 2 other modalities
            others = [m for m in modalities if m != "multimodal"]
            modalities = ["multimodal"] + others[:2]

        result = {
            "mode": mode,
            "modalities": modalities,
            "techniques": techniques,
            "keywords": keywords,
            "purpose_hint": purpose_hint,
        }
        return json.dumps(result, ensure_ascii=False)

    return tag_repo_readme



def merge_dict_and_list_unique_case_insensitive(col1: dict, col2: list) -> list:
    """
    Merge dict[str, list] and list into a single 1D list.
    - Deduplicated
    - Order-preserving
    - Case-insensitive comparison
    - Preserves original casing of first occurrence
    """
    seen = set()
    result = []

    def add(item):
        if not isinstance(item, str):
            return
        key = item.lower()   # 👈 case-insensitive key
        if key not in seen:
            seen.add(key)
            result.append(item)

    # Column 1: dict[str, list]
    if isinstance(col1, dict):
        for v in col1.values():
            if isinstance(v, list):
                for item in v:
                    add(item)

    # Column 2: list
    if isinstance(col2, list):
        for item in col2:
            add(item)

    return result




import pandas as pd
def check_model_identifier_match(df,
                                 dict_col="cleaned_extract_models_by_framework",
                                 list_col="clean_extract_all_model_identifiers"):
    """
    For each row:
    - If `list_col` is empty or None → return True
    - Otherwise, check whether any identifier in `list_col`
      exists in any of the lists inside the dictionary in `dict_col`.

    Returns a pandas Series of booleans.
    """

    def row_match(row):
        identifiers = row[list_col]

        # Edge case: no identifiers to check
        if not identifiers:
            return True

        model_dict = row[dict_col] or {}

        # Flatten all lists inside the dictionary
        all_models = {
            model
            for model_list in model_dict.values()
            for model in model_list
        }

        return any(identifier in all_models for identifier in identifiers)

    return df.apply(row_match, axis=1)

import pandas as pd



def find_cross_dataframe_matches(dfs, col_name, output_col="appears_in"):
    """
    Parameters
    ----------
    dfs : dict
        Dictionary of dataframes, e.g. {"A": A, "B": B, ...}
    col_name : str
        Column name to check across dataframes
    output_col : str
        Name of the output column storing matching dataframe names

    Returns
    -------
    dict
        Dictionary of dataframes with a new column added
    """

    # Precompute sets for fast lookup
    value_sets = {
        name: set(df[col_name].dropna())
        for name, df in dfs.items()
    }

    result = {}

    for name, df in dfs.items():
        df_copy = df.copy()

        def check_value(val):
            matches = [
                other_name
                for other_name, values in value_sets.items()
                if val in values
            ]
            return matches  # includes own dataframe name automatically

        df_copy[output_col] = df_copy[col_name].apply(check_value)
        result[name] = df_copy

    return result


def collect_parent_child_keys(df, params_col="params"):
    """
    Build a dict: {parent_key: [unique child keys across all rows]}
    where df[params_col] contains 2D dictionaries (possibly stored as strings).
    """
    import ast
    import pandas as pd

    result = {}

    for x in df[params_col].tolist():
        # Skip null-like values
        if x is None:
            continue
        if isinstance(x, float) and pd.isna(x):
            continue

        # If stored as string, parse safely to python object
        if isinstance(x, str):
            s = x.strip()
            if not s:
                continue
            try:
                x = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                # Not a valid python-literal dict string
                continue

        # Now x should be a dict
        if not isinstance(x, dict):
            continue

        for parent_key, child in x.items():
            # Only consider parent -> child_dict structure
            if not isinstance(child, dict):
                continue

            if parent_key not in result:
                result[parent_key] = set()

            result[parent_key].update(child.keys())

    # Convert sets to lists (optionally sorted for stable output)
    return {k: sorted(v) for k, v in result.items()}


def get_unique_merged_models(df, column="merged_models"):
    """
    Returns a list of unique elements appearing in a column of lists.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe
    column : str
        Column name containing 1D lists
    
    Returns
    -------
    list
        Unique elements across all rows
    """
    return sorted(
        {
            item
            for sublist in df[column]
            if isinstance(sublist, list)
            for item in sublist
        }
    )


def assign_model_categories(
    df,
    model_dict,
    models_col="merged_models",
    output_col="model_categories",
    case_insensitive=True
):
    """
    Assign model categories based on membership in dictionary value lists.

    Parameters
    ----------
    df : pandas.DataFrame
    model_dict : dict[str, list[str]]
        Mapping of category -> model names
    models_col : str
        Column containing lists of model names
    output_col : str
        Name of output column
    case_insensitive : bool
        Whether to match model names case-insensitively

    Returns
    -------
    pandas.DataFrame
        DataFrame with new column added
    """

    # ---- build reverse lookup: model -> category
    reverse_map = {}

    for category, models in model_dict.items():
        for m in models:
            key = m.lower() if case_insensitive else m
            reverse_map.setdefault(key, set()).add(category)

    # ---- row-wise mapping
    def find_categories(model_list):
        if not isinstance(model_list, list):
            return []

        categories = set()
        for m in model_list:
            key = m.lower() if case_insensitive else m
            if key in reverse_map:
                categories.update(reverse_map[key])

        return sorted(categories)

    df[output_col] = df[models_col].apply(find_categories)
    return df



def clean_repositories_df(df):
    df = df.copy()

    # Ensure numeric columns are numeric
    df["star"] = pd.to_numeric(df["star"], errors="coerce")
    df["fork"] = pd.to_numeric(df["fork"], errors="coerce")
    df["commit_number"] = pd.to_numeric(df["commit_number"], errors="coerce")

    # Parse last update date
    df["last_update"] = pd.to_datetime(
        df["last_update"], errors="coerce", utc=True
    )

    # Build filter mask (rows to KEEP)
    keep_mask = (
        (df["star"] >= 10) &
        (df["fork"] >= 10) &
        (df["commit_number"] >= 100) &
        (df["last_update"].dt.year >= 2025)
    )

    # Repo names that are removed
    removed_repos = df.loc[~keep_mask, "repo_fullname"].tolist()

    # Cleaned DataFrame
    cleaned_df = df.loc[keep_mask].reset_index(drop=True)

    return removed_repos, cleaned_df



# | Aspect             | Answers the Question          |
# | ------------------ | ----------------------------- |
# | Model Family       | Who built it?                 |
# | Alignment Type     | How was it trained to behave? |
# | Modality           | What inputs does it accept?   |
# | Scale              | How big is it?                |
# | Architecture       | How is it structured?         |
# | Quantization       | How compressed is it?         |
# | Context Length     | How much can it remember?     |
# | Specialization     | What is it best at?           |
# | Deployment Context | Where is it used?             |
# | Versioning         | How mature is it?             |
###########################################################################

Baichuan_family = ["Baichuan-13B-Chat","Baichuan-7B","Baichuan2-13B-Chat","Baichuan2-7B-Chat","baichuan2-7b-chat"]
InternVL_family = ["InternVL-Chat","InternVL2-2B","InternVL2-8B","InternVL2_5-26B","InternVL2_5-4B","StructTable-InternVL2-1B"]
LLaMA_family = ["Llama-2-7b-chat","Llama-3.1-8B-Instruct","Llama-4-Scout-17B-16E-Instruct","Meta-Llama-3-1-8B-Instruct","Meta-Llama-3-8B-Instruct","llama-3.1-nemotron-70b-instruct","llama-3.2-nv-embedqa-1b","llama2-chat-70b-4b"]
Qwen_family = ["Qwen-14B-Chat","Qwen-7B-Chat","Qwen-VL-Chat","Qwen1.5-4B-Chat","Qwen1.5-7B-Chat","Qwen2-7B-Instruct","Qwen2-VL-2B-Instruct","Qwen2.5-VL-3B-Instruct","Long-Grounding-qwen","share_data/base_llms/Qwen"]
Yi_family = ["Yi-1.5-9B"]
ChatGLM_family = ["chatglm2-6b","chatglm3-6b","work/models_all/ZhipuAI/glm-4v-9b"]
DeepSeek_family = ["deepseek-coder-6.7b-instruct","deepseek-llm-7b-chat","deepseek-vl-1.3b-chat","deepseek-vl-7b-chat"]
InternLM_family = ["internlm-chat-20b","internlm-chat-20b-4b","internlm-chat-7b","internlm-xcomposer2-4khd-7b","internlm-xcomposer2-7b","internlm-xcomposer2-7b-4b","internlm-xcomposer2-vl-7b","internlm-xcomposer2-vl-7b-4b","internlm-xcomposer2d5-7b","internlm-xcomposer2d5-7b-4b","internlm2-1_8b","internlm2-20b","internlm2-7b","internlm2-base-20b","internlm2-base-7b","internlm2-chat-1_8b","internlm2-chat-20b","internlm2-chat-20b-4b","internlm2-chat-20b-inner-4b","internlm2-chat-7b","internlm2-math-7b","internlm2-math-base-7b","internlm2-math-plus-1_8b","internlm2-math-plus-7b","internlm2_5-1_8b-chat","internlm2_5-20b","internlm2_5-20b-chat","internlm2_5-20b-chat-4b","internlm2_5-7b","internlm2_5-7b-chat","internlm2_5-7b-chat-1m"]
LLaVA_family = ["llava-v1.6-34b","llava-v1.6-mistral-7b","llava-v1.6-vicuna-13b"]
Vicuna_family = ["vicuna-13b","vicuna-7b"]
MiniCPM_family = ["MiniCPM-Llama"]
SigLIP_family = ["siglip-so400m"]
Streamer_family = ["streamer-sales-lelemiao-7b","streamer-sales-lelemiao-7b-4b"]

model_family_dict = {"Baichuan_family":Baichuan_family,"InternVL_family":InternVL_family,"LLaMA_family":LLaMA_family,"Qwen_family":Qwen_family,"Yi_family":Yi_family,"ChatGLM_family":ChatGLM_family,"DeepSeek_family":DeepSeek_family,"InternLM_family":InternLM_family,"LLaVA_family":LLaVA_family,"Vicuna_family":Vicuna_family,"MiniCPM_family":MiniCPM_family,"SigLIP_family":SigLIP_family,"Streamer_family":Streamer_family}

###########################################################################
Chat_aligned = ["Baichuan-13B-Chat","Baichuan2-13B-Chat","Baichuan2-7B-Chat","baichuan2-7b-chat","InternVL-Chat","Qwen-14B-Chat","Qwen-7B-Chat","Qwen-VL-Chat","Qwen1.5-4B-Chat","Qwen1.5-7B-Chat","Yi-1.5-9B","chatglm2-6b","chatglm3-6b","deepseek-llm-7b-chat","deepseek-vl-1.3b-chat","deepseek-vl-7b-chat","internlm-chat-20b","internlm-chat-20b-4b","internlm-chat-7b","internlm2-chat-1_8b","internlm2-chat-20b","internlm2-chat-20b-4b","internlm2-chat-20b-inner-4b","internlm2-chat-7b","internlm2_5-1_8b-chat","internlm2_5-20b-chat","internlm2_5-20b-chat-4b","internlm2_5-7b-chat","internlm2_5-7b-chat-1m","Llama-2-7b-chat","llama2-chat-70b-4b","vicuna-13b","vicuna-7b"]
Instruct_tuned = ["Llama-3.1-8B-Instruct","Llama-4-Scout-17B-16E-Instruct","Meta-Llama-3-1-8B-Instruct","Meta-Llama-3-8B-Instruct","llama-3.1-nemotron-70b-instruct","Qwen2-7B-Instruct","Qwen2-VL-2B-Instruct","Qwen2.5-VL-3B-Instruct","llava-v1.6-34b","llava-v1.6-mistral-7b","llava-v1.6-vicuna-13b"]
Code_instruct = ["deepseek-coder-6.7b-instruct"]
Math_tuned = ["internlm2-math-7b","internlm2-math-base-7b","internlm2-math-plus-1_8b","internlm2-math-plus-7b"]
Embedding_retrieval = ["llama-3.2-nv-embedqa-1b"]
Base_or_unspecified = ["Baichuan-7B","InternVL2-2B","InternVL2-8B","InternVL2_5-26B","InternVL2_5-4B","StructTable-InternVL2-1B","MiniCPM-Llama","internlm2-1_8b","internlm2-20b","internlm2-7b","internlm2-base-20b","internlm2-base-7b","internlm2_5-20b","internlm2_5-7b","internlm-xcomposer2-4khd-7b","internlm-xcomposer2-7b","internlm-xcomposer2-7b-4b","internlm-xcomposer2-vl-7b","internlm-xcomposer2-vl-7b-4b","internlm-xcomposer2d5-7b","internlm-xcomposer2d5-7b-4b","share_data/base_llms/Qwen","siglip-so400m","streamer-sales-lelemiao-7b","streamer-sales-lelemiao-7b-4b","work/models_all/ZhipuAI/glm-4v-9b"]
Specialized_grounding = ["Long-Grounding-qwen"]

alignment_type_dict = {"Chat_aligned":Chat_aligned,"Instruct_tuned":Instruct_tuned,"Code_instruct":Code_instruct,"Math_tuned":Math_tuned,"Embedding_retrieval":Embedding_retrieval,"Base_or_unspecified":Base_or_unspecified,"Specialized_grounding":Specialized_grounding}
###########################################################################
Text_only = ["Baichuan-13B-Chat","Baichuan-7B","Baichuan2-13B-Chat","Baichuan2-7B-Chat","baichuan2-7b-chat","Llama-2-7b-chat","Llama-3.1-8B-Instruct","Llama-4-Scout-17B-16E-Instruct","Meta-Llama-3-1-8B-Instruct","Meta-Llama-3-8B-Instruct","llama-3.1-nemotron-70b-instruct","llama2-chat-70b-4b","Qwen-14B-Chat","Qwen-7B-Chat","Qwen1.5-4B-Chat","Qwen1.5-7B-Chat","Qwen2-7B-Instruct","Yi-1.5-9B","chatglm2-6b","chatglm3-6b","deepseek-coder-6.7b-instruct","deepseek-llm-7b-chat","internlm-chat-20b","internlm-chat-20b-4b","internlm-chat-7b","internlm2-1_8b","internlm2-20b","internlm2-7b","internlm2-base-20b","internlm2-base-7b","internlm2-chat-1_8b","internlm2-chat-20b","internlm2-chat-20b-4b","internlm2-chat-20b-inner-4b","internlm2-chat-7b","internlm2-math-7b","internlm2-math-base-7b","internlm2-math-plus-1_8b","internlm2-math-plus-7b","internlm2_5-1_8b-chat","internlm2_5-20b","internlm2_5-20b-chat","internlm2_5-20b-chat-4b","internlm2_5-7b","internlm2_5-7b-chat","internlm2_5-7b-chat-1m","MiniCPM-Llama","vicuna-13b","vicuna-7b","Long-Grounding-qwen","share_data/base_llms/Qwen","streamer-sales-lelemiao-7b","streamer-sales-lelemiao-7b-4b"]
Vision_Language = ["InternVL-Chat","InternVL2-2B","InternVL2-8B","InternVL2_5-26B","InternVL2_5-4B","StructTable-InternVL2-1B","Qwen-VL-Chat","Qwen2-VL-2B-Instruct","Qwen2.5-VL-3B-Instruct","deepseek-vl-1.3b-chat","deepseek-vl-7b-chat","internlm-xcomposer2-4khd-7b","internlm-xcomposer2-7b","internlm-xcomposer2-7b-4b","internlm-xcomposer2-vl-7b","internlm-xcomposer2-vl-7b-4b","internlm-xcomposer2d5-7b","internlm-xcomposer2d5-7b-4b","llava-v1.6-34b","llava-v1.6-mistral-7b","llava-v1.6-vicuna-13b","work/models_all/ZhipuAI/glm-4v-9b"]
Embedding_or_Encoder = ["llama-3.2-nv-embedqa-1b","siglip-so400m"]

modality_dict = {"Text_only":Text_only,"Vision_Language":Vision_Language,"Embedding_or_Encoder":Embedding_or_Encoder}
###########################################################################
Small_models_upto_2B = ["InternVL2-2B","Qwen2-VL-2B-Instruct","Qwen2.5-VL-3B-Instruct","internlm2-1_8b","internlm2-chat-1_8b","internlm2-math-plus-1_8b","internlm2_5-1_8b-chat","StructTable-InternVL2-1B","llama-3.2-nv-embedqa-1b","siglip-so400m"]

Medium_models_3B_to_9B = ["Baichuan-7B","Baichuan2-7B-Chat","baichuan2-7b-chat","InternVL2-8B","InternVL2_5-4B","Llama-2-7b-chat","Llama-3.1-8B-Instruct","Meta-Llama-3-1-8B-Instruct","Meta-Llama-3-8B-Instruct","Qwen-7B-Chat","Qwen1.5-4B-Chat","Qwen1.5-7B-Chat","Qwen2-7B-Instruct","Yi-1.5-9B","chatglm2-6b","chatglm3-6b","deepseek-coder-6.7b-instruct","deepseek-llm-7b-chat","deepseek-vl-1.3b-chat","deepseek-vl-7b-chat","internlm-chat-7b","internlm2-7b","internlm2-base-7b","internlm2-chat-7b","internlm2-math-7b","internlm2-math-base-7b","internlm2-math-plus-7b","internlm2_5-7b","internlm2_5-7b-chat","internlm2_5-7b-chat-1m","internlm-xcomposer2-4khd-7b","internlm-xcomposer2-7b","internlm-xcomposer2-7b-4b","internlm-xcomposer2-vl-7b","internlm-xcomposer2-vl-7b-4b","internlm-xcomposer2d5-7b","internlm-xcomposer2d5-7b-4b","MiniCPM-Llama","vicuna-7b","streamer-sales-lelemiao-7b","streamer-sales-lelemiao-7b-4b"]

Large_models_10B_to_34B = ["Baichuan-13B-Chat","Baichuan2-13B-Chat","InternVL2_5-26B","Qwen-14B-Chat","internlm-chat-20b","internlm-chat-20b-4b","internlm2-20b","internlm2-base-20b","internlm2-chat-20b","internlm2-chat-20b-4b","internlm2-chat-20b-inner-4b","internlm2_5-20b","internlm2_5-20b-chat","internlm2_5-20b-chat-4b","llava-v1.6-34b","llava-v1.6-vicuna-13b","vicuna-13b","work/models_all/ZhipuAI/glm-4v-9b"]

Very_large_models_70B_plus = ["llama2-chat-70b-4b","llama-3.1-nemotron-70b-instruct"]

MoE_models = ["Llama-4-Scout-17B-16E-Instruct"]

scale_dict = {"Small_models_upto_2B":Small_models_upto_2B,"Medium_models_3B_to_9B":Medium_models_3B_to_9B,"Large_models_10B_to_34B":Large_models_10B_to_34B,"Very_large_models_70B_plus":Very_large_models_70B_plus,"MoE_models":MoE_models}
###########################################################################

Dense_Transformer = ["Baichuan-13B-Chat","Baichuan-7B","Baichuan2-13B-Chat","Baichuan2-7B-Chat","baichuan2-7b-chat","Llama-2-7b-chat","Llama-3.1-8B-Instruct","Meta-Llama-3-1-8B-Instruct","Meta-Llama-3-8B-Instruct","Qwen-14B-Chat","Qwen-7B-Chat","Qwen1.5-4B-Chat","Qwen1.5-7B-Chat","Qwen2-7B-Instruct","Yi-1.5-9B","chatglm2-6b","chatglm3-6b","deepseek-llm-7b-chat","deepseek-coder-6.7b-instruct","internlm-chat-20b","internlm-chat-20b-4b","internlm-chat-7b","internlm2-1_8b","internlm2-20b","internlm2-7b","internlm2-base-20b","internlm2-base-7b","internlm2-chat-1_8b","internlm2-chat-20b","internlm2-chat-20b-4b","internlm2-chat-20b-inner-4b","internlm2-chat-7b","internlm2-math-7b","internlm2-math-base-7b","internlm2-math-plus-1_8b","internlm2-math-plus-7b","internlm2_5-1_8b-chat","internlm2_5-20b","internlm2_5-20b-chat","internlm2_5-20b-chat-4b","internlm2_5-7b","internlm2_5-7b-chat","internlm2_5-7b-chat-1m","MiniCPM-Llama","vicuna-13b","vicuna-7b","streamer-sales-lelemiao-7b","streamer-sales-lelemiao-7b-4b","share_data/base_llms/Qwen"]

Mixture_of_Experts = ["Llama-4-Scout-17B-16E-Instruct","llama-3.1-nemotron-70b-instruct"]

Vision_Encoder_LLM_Hybrid = ["InternVL-Chat","InternVL2-2B","InternVL2-8B","InternVL2_5-26B","InternVL2_5-4B","StructTable-InternVL2-1B","Qwen-VL-Chat","Qwen2-VL-2B-Instruct","Qwen2.5-VL-3B-Instruct","deepseek-vl-1.3b-chat","deepseek-vl-7b-chat","internlm-xcomposer2-4khd-7b","internlm-xcomposer2-7b","internlm-xcomposer2-7b-4b","internlm-xcomposer2-vl-7b","internlm-xcomposer2-vl-7b-4b","internlm-xcomposer2d5-7b","internlm-xcomposer2d5-7b-4b","llava-v1.6-34b","llava-v1.6-mistral-7b","llava-v1.6-vicuna-13b","work/models_all/ZhipuAI/glm-4v-9b"]

Encoder_only_or_Embedding = ["llama-3.2-nv-embedqa-1b","siglip-so400m"]

Specialized_Grounding_or_Retrieval_Architecture = ["Long-Grounding-qwen"]

architecture_dict = {"Dense_Transformer":Dense_Transformer,"Mixture_of_Experts":Mixture_of_Experts,"Vision_Encoder_LLM_Hybrid":Vision_Encoder_LLM_Hybrid,"Encoder_only_or_Embedding":Encoder_only_or_Embedding,"Specialized_Grounding_or_Retrieval_Architecture":Specialized_Grounding_or_Retrieval_Architecture}

###########################################################################
Quantized_4bit = ["internlm-chat-20b-4b","internlm-xcomposer2-7b-4b","internlm-xcomposer2-vl-7b-4b","internlm-xcomposer2d5-7b-4b","internlm2-chat-20b-4b","internlm2-chat-20b-inner-4b","internlm2_5-20b-chat-4b","llama2-chat-70b-4b","streamer-sales-lelemiao-7b-4b"]

Full_precision_or_unspecified = ["Baichuan-13B-Chat","Baichuan-7B","Baichuan2-13B-Chat","Baichuan2-7B-Chat","baichuan2-7b-chat","InternVL-Chat","InternVL2-2B","InternVL2-8B","InternVL2_5-26B","InternVL2_5-4B","Llama-2-7b-chat","Llama-3.1-8B-Instruct","Llama-4-Scout-17B-16E-Instruct","Long-Grounding-qwen","Meta-Llama-3-1-8B-Instruct","Meta-Llama-3-8B-Instruct","MiniCPM-Llama","Qwen-14B-Chat","Qwen-7B-Chat","Qwen-VL-Chat","Qwen1.5-4B-Chat","Qwen1.5-7B-Chat","Qwen2-7B-Instruct","Qwen2-VL-2B-Instruct","Qwen2.5-VL-3B-Instruct","StructTable-InternVL2-1B","Yi-1.5-9B","chatglm2-6b","chatglm3-6b","deepseek-coder-6.7b-instruct","deepseek-llm-7b-chat","deepseek-vl-1.3b-chat","deepseek-vl-7b-chat","internlm-chat-20b","internlm-chat-7b","internlm-xcomposer2-4khd-7b","internlm-xcomposer2-7b","internlm-xcomposer2-vl-7b","internlm-xcomposer2d5-7b","internlm2-1_8b","internlm2-20b","internlm2-7b","internlm2-base-20b","internlm2-base-7b","internlm2-chat-1_8b","internlm2-chat-20b","internlm2-chat-7b","internlm2-math-7b","internlm2-math-base-7b","internlm2-math-plus-1_8b","internlm2-math-plus-7b","internlm2_5-1_8b-chat","internlm2_5-20b","internlm2_5-20b-chat","internlm2_5-7b","internlm2_5-7b-chat","internlm2_5-7b-chat-1m","llama-3.1-nemotron-70b-instruct","llama-3.2-nv-embedqa-1b","llava-v1.6-34b","llava-v1.6-mistral-7b","llava-v1.6-vicuna-13b","share_data/base_llms/Qwen","siglip-so400m","streamer-sales-lelemiao-7b","vicuna-13b","vicuna-7b","work/models_all/ZhipuAI/glm-4v-9b"]

quantization_dict = {"Quantized_4bit":Quantized_4bit,"Full_precision_or_unspecified":Full_precision_or_unspecified}
###########################################################################
Standard_context_upto_8k = ["Baichuan-13B-Chat","Baichuan-7B","Baichuan2-13B-Chat","Baichuan2-7B-Chat","baichuan2-7b-chat","InternVL-Chat","InternVL2-2B","InternVL2-8B","InternVL2_5-4B","Llama-2-7b-chat","Llama-3.1-8B-Instruct","Meta-Llama-3-1-8B-Instruct","Meta-Llama-3-8B-Instruct","MiniCPM-Llama","Qwen-14B-Chat","Qwen-7B-Chat","Qwen-VL-Chat","Qwen1.5-4B-Chat","Qwen1.5-7B-Chat","Qwen2-7B-Instruct","Qwen2-VL-2B-Instruct","Qwen2.5-VL-3B-Instruct","StructTable-InternVL2-1B","Yi-1.5-9B","chatglm2-6b","chatglm3-6b","deepseek-coder-6.7b-instruct","deepseek-llm-7b-chat","deepseek-vl-1.3b-chat","deepseek-vl-7b-chat","internlm-chat-7b","internlm2-1_8b","internlm2-7b","internlm2-base-7b","internlm2-chat-1_8b","internlm2-chat-7b","internlm2-math-7b","internlm2-math-base-7b","internlm2-math-plus-1_8b","internlm2-math-plus-7b","internlm2_5-1_8b-chat","internlm2_5-7b","internlm2_5-7b-chat","llava-v1.6-mistral-7b","llava-v1.6-vicuna-13b","vicuna-7b","vicuna-13b","streamer-sales-lelemiao-7b","streamer-sales-lelemiao-7b-4b","share_data/base_llms/Qwen","siglip-so400m"]

Extended_context_16k_to_32k = ["InternVL2_5-26B","internlm-chat-20b","internlm-chat-20b-4b","internlm2-20b","internlm2-base-20b","internlm2-chat-20b","internlm2-chat-20b-4b","internlm2-chat-20b-inner-4b","internlm2_5-20b","internlm2_5-20b-chat","internlm2_5-20b-chat-4b","internlm-xcomposer2-7b","internlm-xcomposer2-7b-4b","internlm-xcomposer2-vl-7b","internlm-xcomposer2-vl-7b-4b","internlm-xcomposer2d5-7b","internlm-xcomposer2d5-7b-4b","llava-v1.6-34b","work/models_all/ZhipuAI/glm-4v-9b"]

Long_context_64k_plus = ["Llama-4-Scout-17B-16E-Instruct","llama-3.1-nemotron-70b-instruct","llama2-chat-70b-4b"]

Ultra_long_context_1M = ["internlm2_5-7b-chat-1m"]

Specialized_long_grounding = ["Long-Grounding-qwen","llama-3.2-nv-embedqa-1b","internlm-xcomposer2-4khd-7b"]

context_length_dict = {"Standard_context_upto_8k":Standard_context_upto_8k,"Extended_context_16k_to_32k":Extended_context_16k_to_32k,"Long_context_64k_plus":Long_context_64k_plus,"Ultra_long_context_1M":Ultra_long_context_1M,"Specialized_long_grounding":Specialized_long_grounding}
###########################################################################
General_chat_assistants = ["Baichuan-13B-Chat","Baichuan2-13B-Chat","Baichuan2-7B-Chat","baichuan2-7b-chat","InternVL-Chat","Llama-2-7b-chat","Qwen-14B-Chat","Qwen-7B-Chat","Qwen1.5-4B-Chat","Qwen1.5-7B-Chat","Yi-1.5-9B","chatglm2-6b","chatglm3-6b","deepseek-llm-7b-chat","internlm-chat-20b","internlm-chat-20b-4b","internlm-chat-7b","internlm2-chat-1_8b","internlm2-chat-20b","internlm2-chat-20b-4b","internlm2-chat-20b-inner-4b","internlm2-chat-7b","internlm2_5-1_8b-chat","internlm2_5-20b-chat","internlm2_5-20b-chat-4b","internlm2_5-7b-chat","internlm2_5-7b-chat-1m","vicuna-13b","vicuna-7b","llama2-chat-70b-4b"]

Instruction_following = ["Llama-3.1-8B-Instruct","Meta-Llama-3-1-8B-Instruct","Meta-Llama-3-8B-Instruct","Llama-4-Scout-17B-16E-Instruct","llama-3.1-nemotron-70b-instruct","Qwen2-7B-Instruct"]

Vision_language_models = ["Qwen-VL-Chat","Qwen2-VL-2B-Instruct","Qwen2.5-VL-3B-Instruct","InternVL2-2B","InternVL2-8B","InternVL2_5-26B","InternVL2_5-4B","StructTable-InternVL2-1B","deepseek-vl-1.3b-chat","deepseek-vl-7b-chat","internlm-xcomposer2-4khd-7b","internlm-xcomposer2-7b","internlm-xcomposer2-7b-4b","internlm-xcomposer2-vl-7b","internlm-xcomposer2-vl-7b-4b","internlm-xcomposer2d5-7b","internlm-xcomposer2d5-7b-4b","llava-v1.6-34b","llava-v1.6-mistral-7b","llava-v1.6-vicuna-13b","work/models_all/ZhipuAI/glm-4v-9b"]

Code_generation = ["deepseek-coder-6.7b-instruct"]

Mathematics_reasoning = ["internlm2-math-7b","internlm2-math-base-7b","internlm2-math-plus-1_8b","internlm2-math-plus-7b"]

Embedding_and_retrieval = ["llama-3.2-nv-embedqa-1b","siglip-so400m"]

Grounding_and_long_context = ["Long-Grounding-qwen"]

General_base_models = ["Baichuan-7B","InternVL2-2B","MiniCPM-Llama","internlm2-1_8b","internlm2-20b","internlm2-7b","internlm2-base-20b","internlm2-base-7b","internlm2_5-20b","internlm2_5-7b","share_data/base_llms/Qwen","streamer-sales-lelemiao-7b","streamer-sales-lelemiao-7b-4b"]

specialization_dict = {"General_chat_assistants":General_chat_assistants,"Instruction_following":Instruction_following,"Vision_language_models":Vision_language_models,"Code_generation":Code_generation,"Mathematics_reasoning":Mathematics_reasoning,"Embedding_and_retrieval":Embedding_and_retrieval,"Grounding_and_long_context":Grounding_and_long_context,"General_base_models":General_base_models}
###########################################################################
Production_ready_chat_models = ["Baichuan-13B-Chat","Baichuan2-13B-Chat","Baichuan2-7B-Chat","baichuan2-7b-chat","InternVL-Chat","Llama-2-7b-chat","Qwen-14B-Chat","Qwen-7B-Chat","Qwen-VL-Chat","Qwen1.5-4B-Chat","Qwen1.5-7B-Chat","Yi-1.5-9B","chatglm2-6b","chatglm3-6b","deepseek-llm-7b-chat","internlm-chat-20b","internlm-chat-20b-4b","internlm-chat-7b","internlm2-chat-1_8b","internlm2-chat-20b","internlm2-chat-20b-4b","internlm2-chat-20b-inner-4b","internlm2-chat-7b","internlm2_5-1_8b-chat","internlm2_5-20b-chat","internlm2_5-20b-chat-4b","internlm2_5-7b-chat","internlm2_5-7b-chat-1m","vicuna-13b","vicuna-7b","llama2-chat-70b-4b"]

Instruction_serving_and_agents = ["Llama-3.1-8B-Instruct","Meta-Llama-3-1-8B-Instruct","Meta-Llama-3-8B-Instruct","Llama-4-Scout-17B-16E-Instruct","llama-3.1-nemotron-70b-instruct","Qwen2-7B-Instruct"]

Multimodal_application_models = ["InternVL2-2B","InternVL2-8B","InternVL2_5-26B","InternVL2_5-4B","StructTable-InternVL2-1B","Qwen2-VL-2B-Instruct","Qwen2.5-VL-3B-Instruct","deepseek-vl-1.3b-chat","deepseek-vl-7b-chat","internlm-xcomposer2-4khd-7b","internlm-xcomposer2-7b","internlm-xcomposer2-7b-4b","internlm-xcomposer2-vl-7b","internlm-xcomposer2-vl-7b-4b","internlm-xcomposer2d5-7b","internlm-xcomposer2d5-7b-4b","llava-v1.6-34b","llava-v1.6-mistral-7b","llava-v1.6-vicuna-13b","work/models_all/ZhipuAI/glm-4v-9b"]

Research_and_base_models = ["Baichuan-7B","InternVL2-2B","MiniCPM-Llama","internlm2-1_8b","internlm2-20b","internlm2-7b","internlm2-base-20b","internlm2-base-7b","internlm2_5-20b","internlm2_5-7b","share_data/base_llms/Qwen"]

Edge_and_resource_constrained = ["internlm-chat-20b-4b","internlm-xcomposer2-7b-4b","internlm-xcomposer2-vl-7b-4b","internlm-xcomposer2d5-7b-4b","internlm2-chat-20b-4b","internlm2-chat-20b-inner-4b","internlm2_5-20b-chat-4b","streamer-sales-lelemiao-7b","streamer-sales-lelemiao-7b-4b"]

Domain_specific_pipelines = ["deepseek-coder-6.7b-instruct","internlm2-math-7b","internlm2-math-base-7b","internlm2-math-plus-1_8b","internlm2-math-plus-7b","llama-3.2-nv-embedqa-1b","siglip-so400m","Long-Grounding-qwen"]

deployment_context_dict = {"Production_ready_chat_models":Production_ready_chat_models,"Instruction_serving_and_agents":Instruction_serving_and_agents,"Multimodal_application_models":Multimodal_application_models,"Research_and_base_models":Research_and_base_models,"Edge_and_resource_constrained":Edge_and_resource_constrained,"Domain_specific_pipelines":Domain_specific_pipelines}
###########################################################################
First_generation = ["Baichuan-13B-Chat","Baichuan-7B","InternVL-Chat","Llama-2-7b-chat","Qwen-14B-Chat","Qwen-7B-Chat","Qwen-VL-Chat","MiniCPM-Llama","Yi-1.5-9B","chatglm2-6b","internlm-chat-20b","internlm-chat-20b-4b","internlm-chat-7b","vicuna-13b","vicuna-7b","share_data/base_llms/Qwen","siglip-so400m"]
Second_generation = ["Baichuan2-13B-Chat","Baichuan2-7B-Chat","baichuan2-7b-chat","InternVL2-2B","InternVL2-8B","Qwen1.5-4B-Chat","Qwen1.5-7B-Chat","chatglm3-6b","internlm2-1_8b","internlm2-20b","internlm2-7b","internlm2-base-20b","internlm2-base-7b","internlm2-chat-1_8b","internlm2-chat-20b","internlm2-chat-20b-4b","internlm2-chat-20b-inner-4b","internlm2-chat-7b","internlm2-math-7b","internlm2-math-base-7b","internlm2-math-plus-1_8b","internlm2-math-plus-7b","llava-v1.6-34b","llava-v1.6-mistral-7b","llava-v1.6-vicuna-13b","deepseek-llm-7b-chat","deepseek-coder-6.7b-instruct"]
Incremental_1_5_updates = ["Qwen2-7B-Instruct","Qwen2-VL-2B-Instruct","Qwen2.5-VL-3B-Instruct","InternVL2_5-26B","InternVL2_5-4B","internlm2_5-1_8b-chat","internlm2_5-20b","internlm2_5-20b-chat","internlm2_5-20b-chat-4b","internlm2_5-7b","internlm2_5-7b-chat","internlm2_5-7b-chat-1m","deepseek-vl-1.3b-chat","deepseek-vl-7b-chat"]
Third_generation = ["Llama-3.1-8B-Instruct","Meta-Llama-3-1-8B-Instruct","Meta-Llama-3-8B-Instruct","llama-3.1-nemotron-70b-instruct","llama-3.2-nv-embedqa-1b","llama2-chat-70b-4b"]
Next_generation_or_experimental = ["Llama-4-Scout-17B-16E-Instruct","Long-Grounding-qwen","StructTable-InternVL2-1B","internlm-xcomposer2-4khd-7b","internlm-xcomposer2-7b","internlm-xcomposer2-7b-4b","internlm-xcomposer2-vl-7b","internlm-xcomposer2-vl-7b-4b","internlm-xcomposer2d5-7b","internlm-xcomposer2d5-7b-4b","streamer-sales-lelemiao-7b","streamer-sales-lelemiao-7b-4b","work/models_all/ZhipuAI/glm-4v-9b"]

versioning_dict = {"First_generation":First_generation,"Second_generation":Second_generation,"Incremental_1_5_updates":Incremental_1_5_updates,"Third_generation":Third_generation,"Next_generation_or_experimental":Next_generation_or_experimental}
###########################################################################
category_dicts = {
    "model_family": model_family_dict,
    "alignment_type": alignment_type_dict,
    "modality": modality_dict,
    "scale": scale_dict,
    "architecture": architecture_dict,
    "quantization": quantization_dict,
    "context_length": context_length_dict,
    "specialization": specialization_dict,
    "deployment_context": deployment_context_dict,
    "versioning": versioning_dict,
}


# category_columns = list(category_dicts.keys())




def add_llm_category_columns(df: pd.DataFrame, merged_col: str, category_dicts: dict) -> pd.DataFrame:

    out = df.copy()

    # Ensure merged_models is always a list (handle NaN / None / non-list)
    def _as_list(x):
        return x if isinstance(x, list) else []

    out[merged_col] = out[merged_col].apply(_as_list)

    # Build per-category: model_name -> set({subcategory_name, ...})
    model_to_subcats_by_category = {}
    for category_name, subcat_map in category_dicts.items():
        model_to_subcats = {}
        for subcat_name, models in subcat_map.items():
            for m in models:
                model_to_subcats.setdefault(m, set()).add(subcat_name)
        model_to_subcats_by_category[category_name] = model_to_subcats

    # One pass per category (10 times if you have 10 categories)
    for category_name, model_to_subcats in model_to_subcats_by_category.items():

        def _detect_subcats(model_list):
            found = set()
            for m in model_list:
                if m in model_to_subcats:
                    found.update(model_to_subcats[m])
            return sorted(found)

        out[category_name] = out[merged_col].apply(_detect_subcats)

    return out



from collections import Counter
import pandas as pd

def count_and_sort_llm_categories(
    df: pd.DataFrame,
    category_columns: list[str]
) -> dict:

    results = {}

    for col in category_columns:
        counter = Counter()

        for values in df[col]:
            if isinstance(values, list):
                counter.update(values)

        # sort by descending count
        sorted_counter = dict(
            sorted(counter.items(), key=lambda x: x[1], reverse=True)
        )

        results[col] = sorted_counter

    return results



def get_inputs(tool_name):
    vllm_inputs = ["LLM","SamplingParams","EngineArgs","VllmConfig","config","CUDAGraphMode","vllm.LLM","vllm.SamplingParams","vllm.EngineArgs","vllm.VllmConfig","vllm.config","vllm.config.CUDAGraphMode","lora","generate","awq","llmcompressor","vllm.lora","llm.generate","awq.AutoAWQForCausalLM","AutoAWQForCausalLM","gptqmodel.GPTQModel","gptqmodel.QuantizeConfig","QuantizeConfig", "config","ModelConfig","config.ModelConfig","config","CacheConfig","config.CacheConfig","config","LoadConfig","config.LoadConfig","config","ParallelConfig","config.ParallelConfig","config","SchedulerConfig","config.SchedulerConfig","config","DeviceConfig","config.DeviceConfig","config","SpeculativeConfig","config.SpeculativeConfig","config","LoRAConfig","config.LoRAConfig","config","MultiModalConfig","config.MultiModalConfig","config","PoolerConfig","config.PoolerConfig","config","StructuredOutputsConfig","config.StructuredOutputsConfig","config","ProfilerConfig","config.ProfilerConfig","config","ObservabilityConfig","config.ObservabilityConfig","config","KVTransferConfig","config.KVTransferConfig","config","CompilationConfig","config.CompilationConfig","config","VllmConfig","config.VllmConfig","PoolingParams","LLMEngine","AsyncLLMEngine","SamplingParams","PoolingParams","multimodal","multimodal","MULTIMODAL_REGISTRY","multimodal.MULTIMODAL_REGISTRY","multimodal","inputs","MultiModalDataDict","multimodal.inputs.MultiModalDataDict","multimodal","inputs","PlaceholderRange","multimodal.inputs.PlaceholderRange","multimodal","inputs","NestedTensors","multimodal.inputs.NestedTensors","multimodal","inputs","MultiModalFieldElem","multimodal.inputs.MultiModalFieldElem","multimodal","inputs","MultiModalFieldConfig","multimodal.inputs.MultiModalFieldConfig","multimodal","inputs","MultiModalKwargsItem","multimodal.inputs.MultiModalKwargsItem","multimodal","inputs","MultiModalKwargsItems","multimodal.inputs.MultiModalKwargsItems","multimodal","inputs","MultiModalInputs","multimodal.inputs.MultiModalInputs","AttentionConfig","CUDAGraphMode","KernelConfig","ModelConfig","MultiModalConfig","OffloadConfig","StructuredOutputsConfig","VllmConfig","vllm.config","vllm.ModelConfig","vllm.config.ModelConfig","vllm.config","vllm.CacheConfig","vllm.config.CacheConfig","vllm.config","vllm.LoadConfig","vllm.config.LoadConfig","vllm.config","vllm.ParallelConfig","vllm.config.ParallelConfig","vllm.config","vllm.SchedulerConfig","vllm.config.SchedulerConfig","vllm.config","vllm.DeviceConfig","vllm.config.DeviceConfig","vllm.config","vllm.SpeculativeConfig","vllm.config.SpeculativeConfig","vllm.config","vllm.LoRAConfig","vllm.config.LoRAConfig","vllm.config","vllm.MultiModalConfig","vllm.config.MultiModalConfig","vllm.config","vllm.PoolerConfig","vllm.config.PoolerConfig","vllm.config","vllm.StructuredOutputsConfig","vllm.config.StructuredOutputsConfig","vllm.config","vllm.ProfilerConfig","vllm.config.ProfilerConfig","vllm.config","vllm.ObservabilityConfig","vllm.config.ObservabilityConfig","vllm.config","vllm.KVTransferConfig","vllm.config.KVTransferConfig","vllm.config","vllm.CompilationConfig","vllm.config.CompilationConfig","vllm.config","vllm.VllmConfig","vllm.config.VllmConfig","vllm.PoolingParams","vllm.LLMEngine","vllm.AsyncLLMEngine","vllm.SamplingParams","vllm.PoolingParams","vllm.multimodal","vllm.multimodal","vllm.MULTIMODAL_REGISTRY","vllm.multimodal.MULTIMODAL_REGISTRY","vllm.multimodal","vllm.inputs","vllm.MultiModalDataDict","vllm.multimodal.inputs.MultiModalDataDict","vllm.multimodal","vllm.inputs","vllm.PlaceholderRange","vllm.multimodal.inputs.PlaceholderRange","vllm.multimodal","vllm.inputs","vllm.NestedTensors","vllm.multimodal.inputs.NestedTensors","vllm.multimodal","vllm.inputs","vllm.MultiModalFieldElem","vllm.multimodal.inputs.MultiModalFieldElem","vllm.multimodal","vllm.inputs","vllm.MultiModalFieldConfig","vllm.multimodal.inputs.MultiModalFieldConfig","vllm.multimodal","vllm.inputs","vllm.MultiModalKwargsItem","vllm.multimodal.inputs.MultiModalKwargsItem","vllm.multimodal","vllm.inputs","vllm.MultiModalKwargsItems","vllm.multimodal.inputs.MultiModalKwargsItems","vllm.multimodal","vllm.inputs","vllm.MultiModalInputs","vllm.multimodal.inputs.MultiModalInputs","vllm.AttentionConfig","vllm.CUDAGraphMode","vllm.KernelConfig","vllm.ModelConfig","vllm.MultiModalConfig","vllm.OffloadConfig","vllm.StructuredOutputsConfig","vllm.VllmConfig"] # VLLM
    sglang_inputs = ["sgl.Engine","Engine","sglang.Engine"] #SGLANG
    tensorrt_inputs = ["TrtLlmArgs","LLM","BuildConfig","tensorrt_llm.LLM","tensorrt_llm.llmapi.BuildConfig","tensorrt_llm.LLM","llm.generate","llm.get_kv_cache_events","tensorrt_llm.TrtLlmArgs","tensorrt_llm.BuildConfig","tensorrt_llm.llmapi.LLM","LLM.generate","LLM.generate_async","LLM","TorchLlmArg","TrtLlmArgs","llm.generate","LLM.generate_async","tensorrt_llm.llmapi.LLM","tensorrt_llm.llmapi.TorchLlmArg","tensorrt_llm.llmapi.TrtLlmArgs","tensorrt_llm.llmapi.llm.generate","tensorrt_llm.llmapi.llm.LLM.generate_async"] # tensorrt_llm
    lmdeploy_inputs = ["TurbomindEngineConfig","PytorchEngineConfig","pipeline", "lmdeploy.TurbomindEngineConfig", "lmdeploy.PytorchEngineConfig","lmdeploy.pipeline"] # lmdeploy
    flashinfer_inputs = ["single_decode_with_kv_cache","cudnn_batch_decode_with_kv_cache","trtllm_batch_decode_with_kv_cache","BatchDecodeWithPagedKVCacheWrapper","CUDAGraphBatchDecodeWithPagedKVCacheWrapper","xqa","xqa_mla","single_prefill_with_kv_cache","single_prefill_with_kv_cache_return_lse","cudnn_batch_prefill_with_kv_cache","trtllm_batch_context_with_kv_cache","BatchPrefillWithPagedKVCacheWrapper","BatchPrefillWithRaggedKVCacheWrapper","BatchMLAPagedAttentionWrapper","trtllm_batch_decode_with_kv_cache_mla","mm_fp4","bmm_fp8","gemm_fp8_nt_groupwise","group_gemm_fp8_nt_groupwise","group_deepgemm_fp8_nt_groupwise","batch_deepgemm_fp8_nt_groupwise","group_gemm_mxfp4_nt_groupwise","SegmentGEMMWrapper","RoutingMethodType","WeightLayout","convert_to_block_layout","reorder_rows_for_gated_act_gemm","cutlass_fused_moe","trtllm_fp4_block_scale_moe","trtllm_fp8_block_scale_moe","trtllm_fp8_per_tensor_scale_moe","merge_state","merge_state_in_place","merge_states","MultiLevelCascadeAttentionWrapper","BatchDecodeWithSharedPrefixPagedKVCacheWrapper","BatchPrefillWithSharedPrefixPagedKVCacheWrapper","CudaRTLibrary","create_shared_buffer","free_shared_buffer","pack_strided_memory","Mapping","AllReduceFusionOp","AllReduceFusionPattern","AllReduceStrategyConfig","AllReduceStrategyType","trtllm_allreduce_fusion","trtllm_custom_all_reduce","trtllm_moe_allreduce_fusion","trtllm_moe_finalize_allreduce_fusion","trtllm_create_ipc_workspace_for_all_reduce","trtllm_create_ipc_workspace_for_all_reduce_fusion","trtllm_destroy_ipc_workspace_for_all_reduce","trtllm_destroy_ipc_workspace_for_all_reduce_fusion","trtllm_lamport_initialize","trtllm_lamport_initialize_all","compute_fp4_swizzled_layout_sf_size","vllm_all_reduce","vllm_dispose","vllm_init_custom_ar","vllm_register_buffer","vllm_register_graph_buffers","vllm_get_graph_buffer_ipc_meta","vllm_meta_size","MnnvlMemory","McastGPUBuffer","create_tensor_from_cuda_memory","alloc_and_copy_to_cuda","trtllm_mnnvl_all_reduce","trtllm_mnnvl_fused_allreduce_rmsnorm","mpi_barrier","MoeAlltoAll","moe_a2a_initialize","moe_a2a_dispatch","moe_a2a_combine","moe_a2a_sanitize_expert_ids","moe_a2a_get_workspace_size_per_rank","moe_a2a_wrap_payload_tensor_in_workspace","BlockSparseAttentionWrapper","VariableBlockSparseAttentionWrapper","append_paged_kv_cache","append_paged_mla_kv_cache","get_batch_indices_positions","sampling_from_probs","top_p_sampling_from_probs","top_k_sampling_from_probs","min_p_sampling_from_probs","top_k_top_p_sampling_from_logits","top_k_top_p_sampling_from_probs","top_p_renorm_probs","top_k_renorm_probs","top_k_mask_logits","chain_speculative_sampling","top_k","top_k_page_table_transform","top_k_ragged_transform","topk.can_implement_filtered_topk","LogitsPipe","LogitsProcessor","Temperature","Softmax","TopK","TopP","MinP","Sample","TensorType","TaggedTensor","rmsnorm","fused_add_rmsnorm","gemma_rmsnorm","gemma_fused_add_rmsnorm","layernorm","apply_rope_inplace","apply_llama31_rope_inplace","apply_rope","apply_llama31_rope","apply_rope_pos_ids","apply_rope_pos_ids_inplace","apply_llama31_rope_pos_ids","apply_llama31_rope_pos_ids_inplace","apply_rope_with_cos_sin_cache","apply_rope_with_cos_sin_cache_inplace","silu_and_mul","gelu_tanh_and_mul","gelu_and_mul","silu_and_mul_scaled_nvfp4_experts_quantize","packbits","segment_packbits","split_device_green_ctx","split_device_green_ctx_by_sm_count","fp4_quantize","nvfp4_quantize","nvfp4_batched_quantize","nvfp4_block_scale_interleave","e2m1_and_ufp8sf_scale_to_float","scaled_fp4_grouped_quantize","shuffle_matrix_a","shuffle_matrix_sf_a","SfLayout","set_seed","sleep_after_kernel_run","attention_flops","attention_flops_with_actual_seq_lens","attention_tflops_per_sec","attention_tflops_per_sec_with_actual_seq_lens","attention_tb_per_sec","attention_tb_per_sec_with_actual_seq_lens","bench_gpu_time","bench_gpu_time_with_cuda_event","bench_gpu_time_with_cudagraph","bench_gpu_time_with_cupti","flashinfer.decode.single_decode_with_kv_cache", "flashinfer.decode.cudnn_batch_decode_with_kv_cache", "flashinfer.decode.trtllm_batch_decode_with_kv_cache", "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper", "flashinfer.decode.CUDAGraphBatchDecodeWithPagedKVCacheWrapper", "flashinfer.xqa.xqa", "flashinfer.xqa.xqa_mla", "flashinfer.prefill.single_prefill_with_kv_cache", "flashinfer.prefill.single_prefill_with_kv_cache_return_lse", "flashinfer.prefill.cudnn_batch_prefill_with_kv_cache", "flashinfer.prefill.trtllm_batch_context_with_kv_cache", "flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper", "flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper", "flashinfer.mla.BatchMLAPagedAttentionWrapper", "flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla", "flashinfer.gemm.mm_fp4", "flashinfer.gemm.bmm_fp8", "flashinfer.gemm.gemm_fp8_nt_groupwise", "flashinfer.gemm.group_gemm_fp8_nt_groupwise", "flashinfer.gemm.group_deepgemm_fp8_nt_groupwise", "flashinfer.gemm.batch_deepgemm_fp8_nt_groupwise", "flashinfer.gemm.group_gemm_mxfp4_nt_groupwise", "flashinfer.gemm.SegmentGEMMWrapper", "flashinfer.fused_moe.RoutingMethodType", "flashinfer.fused_moe.WeightLayout", "flashinfer.fused_moe.convert_to_block_layout", "flashinfer.fused_moe.reorder_rows_for_gated_act_gemm", "flashinfer.fused_moe.cutlass_fused_moe", "flashinfer.fused_moe.trtllm_fp4_block_scale_moe", "flashinfer.fused_moe.trtllm_fp8_block_scale_moe", "flashinfer.fused_moe.trtllm_fp8_per_tensor_scale_moe", "flashinfer.cascade.merge_state", "flashinfer.cascade.merge_state_in_place", "flashinfer.cascade.merge_states", "flashinfer.cascade.MultiLevelCascadeAttentionWrapper", "flashinfer.cascade.BatchDecodeWithSharedPrefixPagedKVCacheWrapper", "flashinfer.cascade.BatchPrefillWithSharedPrefixPagedKVCacheWrapper", "flashinfer.cascade.MultiLevelCascadeAttentionWrapper", "flashinfer.cascade.BatchDecodeWithSharedPrefixPagedKVCacheWrapper", "flashinfer.cascade.BatchPrefillWithSharedPrefixPagedKVCacheWrapper", "flashinfer.comm.CudaRTLibrary", "flashinfer.comm.create_shared_buffer", "flashinfer.comm.free_shared_buffer", "flashinfer.comm.pack_strided_memory", "flashinfer.comm.Mapping", "flashinfer.comm.AllReduceFusionOp", "flashinfer.comm.AllReduceFusionPattern", "flashinfer.comm.AllReduceStrategyConfig", "flashinfer.comm.AllReduceStrategyType", "flashinfer.comm.trtllm_allreduce_fusion", "flashinfer.comm.trtllm_custom_all_reduce", "flashinfer.comm.trtllm_moe_allreduce_fusion", "flashinfer.comm.trtllm_moe_finalize_allreduce_fusion", "flashinfer.comm.trtllm_create_ipc_workspace_for_all_reduce", "flashinfer.comm.trtllm_create_ipc_workspace_for_all_reduce_fusion", "flashinfer.comm.trtllm_destroy_ipc_workspace_for_all_reduce", "flashinfer.comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion", "flashinfer.comm.trtllm_lamport_initialize", "flashinfer.comm.trtllm_lamport_initialize_all", "flashinfer.comm.compute_fp4_swizzled_layout_sf_size", "flashinfer.comm.vllm_all_reduce", "flashinfer.comm.vllm_dispose", "flashinfer.comm.vllm_init_custom_ar", "flashinfer.comm.vllm_register_buffer", "flashinfer.comm.vllm_register_graph_buffers", "flashinfer.comm.vllm_get_graph_buffer_ipc_meta", "flashinfer.comm.vllm_meta_size", "flashinfer.comm", "flashinfer.comm.mnnvl.MnnvlMemory", "flashinfer.comm", "flashinfer.comm.mnnvl.McastGPUBuffer", "flashinfer.comm", "flashinfer.comm.mnnvl.create_tensor_from_cuda_memory", "flashinfer.comm", "flashinfer.comm.mnnvl.alloc_and_copy_to_cuda", "flashinfer.comm", "flashinfer.comm.trtllm_mnnvl_ar.trtllm_mnnvl_all_reduce", "flashinfer.comm", "flashinfer.comm.trtllm_mnnvl_ar.trtllm_mnnvl_fused_allreduce_rmsnorm", "flashinfer.comm", "flashinfer.comm.trtllm_mnnvl_ar.mpi_barrier", "flashinfer.comm.MoeAlltoAll", "flashinfer.comm.moe_a2a_initialize", "flashinfer.comm.moe_a2a_dispatch", "flashinfer.comm.moe_a2a_combine", "flashinfer.comm.moe_a2a_sanitize_expert_ids", "flashinfer.comm.moe_a2a_get_workspace_size_per_rank", "flashinfer.comm.moe_a2a_wrap_payload_tensor_in_workspace", "flashinfer.sparse.BlockSparseAttentionWrapper", "flashinfer.sparse.VariableBlockSparseAttentionWrapper", "flashinfer.page.append_paged_kv_cache", "flashinfer.page.append_paged_mla_kv_cache", "flashinfer.page.get_batch_indices_positions", "flashinfer.sampling.sampling_from_probs", "flashinfer.sampling.", "flashinfer.sampling.top_p_sampling_from_probs", "flashinfer.sampling.", "flashinfer.sampling.top_k_sampling_from_probs", "flashinfer.sampling.", "flashinfer.sampling.min_p_sampling_from_probs", "flashinfer.sampling.", "flashinfer.sampling.top_k_top_p_sampling_from_logits", "flashinfer.sampling.", "flashinfer.sampling.top_k_top_p_sampling_from_probs", "flashinfer.sampling.", "flashinfer.sampling.top_p_renorm_probs", "flashinfer.sampling.", "flashinfer.sampling.top_k_renorm_probs", "flashinfer.sampling.", "flashinfer.sampling.top_k_mask_logits", "flashinfer.sampling.", "flashinfer.sampling.chain_speculative_sampling", "flashinfer.topk.top_k", "flashinfer.topk.top_k_page_table_transform", "flashinfer.topk.top_k_ragged_transform", "nan.topk.can_implement_filtered_topk", "flashinfer.logits_processor.LogitsPipe", "flashinfer.logits_processor.LogitsPipe", "flashinfer.logits_processor.LogitsProcessor", "flashinfer.logits_processor.Temperature", "flashinfer.logits_processor.Softmax", "flashinfer.logits_processor.TopK", "flashinfer.logits_processor.TopP", "flashinfer.logits_processor.MinP", "flashinfer.logits_processor.Sample", "flashinfer.logits_processor.TensorType", "flashinfer.logits_processor.TaggedTensor", "flashinfer.logits_processor.nan", "flashinfer.norm.rmsnorm", "flashinfer.norm.fused_add_rmsnorm", "flashinfer.norm.gemma_rmsnorm", "flashinfer.norm.gemma_fused_add_rmsnorm", "flashinfer.norm.layernorm", "flashinfer.rope.apply_rope_inplace", "flashinfer.rope.apply_llama31_rope_inplace", "flashinfer.rope.apply_rope", "flashinfer.rope.apply_llama31_rope", "flashinfer.rope.apply_rope_pos_ids", "flashinfer.rope.apply_rope_pos_ids_inplace", "flashinfer.rope.apply_llama31_rope_pos_ids", "flashinfer.rope.apply_llama31_rope_pos_ids_inplace", "flashinfer.rope.apply_rope_with_cos_sin_cache", "flashinfer.rope.apply_rope_with_cos_sin_cache_inplace", "flashinfer.activation.silu_and_mul", "flashinfer.activation.gelu_tanh_and_mul", "flashinfer.activation.gelu_and_mul", "flashinfer.activation.silu_and_mul_scaled_nvfp4_experts_quantize", "flashinfer.quantization.packbits", "flashinfer.quantization.segment_packbits", "flashinfer.green_ctx.split_device_green_ctx", "flashinfer.green_ctx.split_device_green_ctx_by_sm_count", "flashinfer.fp4_quantization.fp4_quantize", "flashinfer.fp4_quantization.nvfp4_quantize", "flashinfer.fp4_quantization.nvfp4_batched_quantize", "flashinfer.fp4_quantization.nvfp4_block_scale_interleave", "flashinfer.fp4_quantization.e2m1_and_ufp8sf_scale_to_float", "flashinfer.fp4_quantization.scaled_fp4_grouped_quantize", "flashinfer.fp4_quantization.shuffle_matrix_a", "flashinfer.fp4_quantization.shuffle_matrix_sf_a", "flashinfer.fp4_quantization.SfLayout", "flashinfer.testing.set_seed", "flashinfer.testing.sleep_after_kernel_run", "flashinfer.testing.attention_flops", "flashinfer.testing.attention_flops_with_actual_seq_lens", "flashinfer.testing.attention_tflops_per_sec", "flashinfer.testing.attention_tflops_per_sec_with_actual_seq_lens", "flashinfer.testing.attention_tb_per_sec", "flashinfer.testing.attention_tb_per_sec_with_actual_seq_lens", "flashinfer.testing.bench_gpu_time", "flashinfer.testing.bench_gpu_time_with_cuda_event", "flashinfer.testing.bench_gpu_time_with_cudagraph", "flashinfer.testing.bench_gpu_time_with_cupti","flashinfer.single_decode_with_kv_cache", "flashinfer.cudnn_batch_decode_with_kv_cache", "flashinfer.trtllm_batch_decode_with_kv_cache", "flashinfer.BatchDecodeWithPagedKVCacheWrapper", "flashinfer.CUDAGraphBatchDecodeWithPagedKVCacheWrapper", "flashinfer.xqa", "flashinfer.xqa_mla", "flashinfer.single_prefill_with_kv_cache", "flashinfer.single_prefill_with_kv_cache_return_lse", "flashinfer.cudnn_batch_prefill_with_kv_cache", "flashinfer.trtllm_batch_context_with_kv_cache", "flashinfer.BatchPrefillWithPagedKVCacheWrapper", "flashinfer.BatchPrefillWithRaggedKVCacheWrapper", "flashinfer.BatchMLAPagedAttentionWrapper", "flashinfer.trtllm_batch_decode_with_kv_cache_mla", "flashinfer.mm_fp4", "flashinfer.bmm_fp8", "flashinfer.gemm_fp8_nt_groupwise", "flashinfer.group_gemm_fp8_nt_groupwise", "flashinfer.group_deepgemm_fp8_nt_groupwise", "flashinfer.batch_deepgemm_fp8_nt_groupwise", "flashinfer.group_gemm_mxfp4_nt_groupwise", "flashinfer.SegmentGEMMWrapper", "flashinfer.RoutingMethodType", "flashinfer.WeightLayout", "flashinfer.convert_to_block_layout", "flashinfer.reorder_rows_for_gated_act_gemm", "flashinfer.cutlass_fused_moe", "flashinfer.trtllm_fp4_block_scale_moe", "flashinfer.trtllm_fp8_block_scale_moe", "flashinfer.trtllm_fp8_per_tensor_scale_moe", "flashinfer.merge_state", "flashinfer.merge_state_in_place", "flashinfer.merge_states", "flashinfer.MultiLevelCascadeAttentionWrapper", "flashinfer.BatchDecodeWithSharedPrefixPagedKVCacheWrapper", "flashinfer.BatchPrefillWithSharedPrefixPagedKVCacheWrapper", "flashinfer.MultiLevelCascadeAttentionWrapper", "flashinfer.BatchDecodeWithSharedPrefixPagedKVCacheWrapper", "flashinfer.BatchPrefillWithSharedPrefixPagedKVCacheWrapper", "flashinfer.CudaRTLibrary", "flashinfer.create_shared_buffer", "flashinfer.free_shared_buffer", "flashinfer.pack_strided_memory", "flashinfer.Mapping", "flashinfer.AllReduceFusionOp", "flashinfer.AllReduceFusionPattern", "flashinfer.AllReduceStrategyConfig", "flashinfer.AllReduceStrategyType", "flashinfer.trtllm_allreduce_fusion", "flashinfer.trtllm_custom_all_reduce", "flashinfer.trtllm_moe_allreduce_fusion", "flashinfer.trtllm_moe_finalize_allreduce_fusion", "flashinfer.trtllm_create_ipc_workspace_for_all_reduce", "flashinfer.trtllm_create_ipc_workspace_for_all_reduce_fusion", "flashinfer.trtllm_destroy_ipc_workspace_for_all_reduce", "flashinfer.trtllm_destroy_ipc_workspace_for_all_reduce_fusion", "flashinfer.trtllm_lamport_initialize", "flashinfer.trtllm_lamport_initialize_all", "flashinfer.compute_fp4_swizzled_layout_sf_size", "flashinfer.vllm_all_reduce", "flashinfer.vllm_dispose", "flashinfer.vllm_init_custom_ar", "flashinfer.vllm_register_buffer", "flashinfer.vllm_register_graph_buffers", "flashinfer.vllm_get_graph_buffer_ipc_meta", "flashinfer.vllm_meta_size", "flashinfer.MnnvlMemory", "flashinfer.McastGPUBuffer", "flashinfer.create_tensor_from_cuda_memory", "flashinfer.alloc_and_copy_to_cuda", "flashinfer.trtllm_mnnvl_all_reduce", "flashinfer.trtllm_mnnvl_fused_allreduce_rmsnorm", "flashinfer.mpi_barrier", "flashinfer.MoeAlltoAll", "flashinfer.moe_a2a_initialize", "flashinfer.moe_a2a_dispatch", "flashinfer.moe_a2a_combine", "flashinfer.moe_a2a_sanitize_expert_ids", "flashinfer.moe_a2a_get_workspace_size_per_rank", "flashinfer.moe_a2a_wrap_payload_tensor_in_workspace", "flashinfer.BlockSparseAttentionWrapper", "flashinfer.VariableBlockSparseAttentionWrapper", "flashinfer.append_paged_kv_cache", "flashinfer.append_paged_mla_kv_cache", "flashinfer.get_batch_indices_positions", "flashinfer.sampling_from_probs", "flashinfer.", "flashinfer.top_p_sampling_from_probs", "flashinfer.", "flashinfer.top_k_sampling_from_probs", "flashinfer.", "flashinfer.min_p_sampling_from_probs", "flashinfer.", "flashinfer.top_k_top_p_sampling_from_logits", "flashinfer.", "flashinfer.top_k_top_p_sampling_from_probs", "flashinfer.", "flashinfer.top_p_renorm_probs", "flashinfer.", "flashinfer.top_k_renorm_probs", "flashinfer.", "flashinfer.top_k_mask_logits", "flashinfer.", "flashinfer.chain_speculative_sampling", "flashinfer.top_k", "flashinfer.top_k_page_table_transform", "flashinfer.top_k_ragged_transform", "flashinfer.topk.can_implement_filtered_topk", "flashinfer.LogitsPipe", "flashinfer.LogitsPipe", "flashinfer.LogitsProcessor", "flashinfer.Temperature", "flashinfer.Softmax", "flashinfer.TopK", "flashinfer.TopP", "flashinfer.MinP", "flashinfer.Sample", "flashinfer.TensorType", "flashinfer.TaggedTensor", "flashinfer.nan", "flashinfer.rmsnorm", "flashinfer.fused_add_rmsnorm", "flashinfer.gemma_rmsnorm", "flashinfer.gemma_fused_add_rmsnorm", "flashinfer.layernorm", "flashinfer.apply_rope_inplace", "flashinfer.apply_llama31_rope_inplace", "flashinfer.apply_rope", "flashinfer.apply_llama31_rope", "flashinfer.apply_rope_pos_ids", "flashinfer.apply_rope_pos_ids_inplace", "flashinfer.apply_llama31_rope_pos_ids", "flashinfer.apply_llama31_rope_pos_ids_inplace", "flashinfer.apply_rope_with_cos_sin_cache", "flashinfer.apply_rope_with_cos_sin_cache_inplace", "flashinfer.silu_and_mul", "flashinfer.gelu_tanh_and_mul", "flashinfer.gelu_and_mul", "flashinfer.silu_and_mul_scaled_nvfp4_experts_quantize", "flashinfer.packbits", "flashinfer.segment_packbits", "flashinfer.split_device_green_ctx", "flashinfer.split_device_green_ctx_by_sm_count", "flashinfer.fp4_quantize", "flashinfer.nvfp4_quantize", "flashinfer.nvfp4_batched_quantize", "flashinfer.nvfp4_block_scale_interleave", "flashinfer.e2m1_and_ufp8sf_scale_to_float", "flashinfer.scaled_fp4_grouped_quantize", "flashinfer.shuffle_matrix_a", "flashinfer.shuffle_matrix_sf_a", "flashinfer.SfLayout", "flashinfer.set_seed", "flashinfer.sleep_after_kernel_run", "flashinfer.attention_flops", "flashinfer.attention_flops_with_actual_seq_lens", "flashinfer.attention_tflops_per_sec", "flashinfer.attention_tflops_per_sec_with_actual_seq_lens", "flashinfer.attention_tb_per_sec", "flashinfer.attention_tb_per_sec_with_actual_seq_lens", "flashinfer.bench_gpu_time", "flashinfer.bench_gpu_time_with_cuda_event", "flashinfer.bench_gpu_time_with_cudagraph", "flashinfer.bench_gpu_time_with_cupti"] # flashinfer
    if tool_name == "vllm":
        return vllm_inputs
    elif tool_name == "sglang":
        return sglang_inputs
    elif tool_name == "tensorrt_llm":
        return tensorrt_inputs
    elif tool_name == "lmdeploy":
        return lmdeploy_inputs
    elif tool_name == "flashinfer":
        return flashinfer_inputs
        

def get_tools_parameters(tool_name):

    # Sglang variable and parameters name:
    #  for example, for speculative decoding we have : sgl.Engine(speculative_algorithm="eagle"), sgl.Engine(speculative_draft_model_path="TinyLlama/TinyLlama-1.1B"), sgl.Engine(speculative_num_steps=4)
    # Thus, in the following dictionary we have a key and value like: "speculative decoding": ["speculative_algorithm", "speculative_draft_model_path", "speculative_num_steps"]

    vllm_parameters = {"Speculative decoding": ["speculative_config"],
                    "Early existing":["early_stopping"],
                    "Cascade inference":["disable_cascade_attn"],
                    "Paged Attention":["attention_kernels","paged_attention_kernel"],
                    "Continuous batching":["max_num_batched_tokens","batch_size"],
                    "Quantization":["quantization","QuantizeConfig","GPTQModifier","SmoothQuantModifier","QuantizationModifier"],
                    "Chunked prefill":["enable_chunked_prefill"],
                    "parallelism":["PARTITION_SIZE","tensor_parallel_size","pipeline-parallel-size"],
                    "Prefix caching":["enable_prefix_caching"],
                    "Multi-LoRA":["enable_lora","lora_request"]}

    sglan_parameters = {"RadixAttention":["disable_radix_cache","radix_eviction_policy"],
            "zero-overhead CPU scheduler": ["schedule_policy", "schedule_conservativeness", "enable_priority_scheduling", "schedule_low_priority_values_first", "priority_scheduling_preemption_threshold", "num_continuous_decode_steps", "scheduler_recv_interval", "enable_two_batch_overlap", "enable_single_batch_overlap", "tbo_token_distribution_threshold"],
            "prefill-decode disaggregation": ["disaggregation_mode", "disaggregation_transfer_backend", "disaggregation_bootstrap_port", "disaggregation_decode_tp", "disaggregation_decode_dp", "disaggregation_prefill_pp", "disaggregation_ib_device", "disaggregation_decode_enable_offload_kvcache", "disaggregation_decode_polling_interval"],
            "speculative decoding": ["speculative_algorithm", "speculative_draft_model_path", "speculative_num_steps", "speculative_eagle_topk", "speculative_eagle_topk_scale", "speculative_accept_threshold_single", "speculative_accept_threshold_acc", "speculative_token_map", "speculative_num_draft_tokens"],
            "continuous batching": ["max_running_requests", "max_queued_requests", "max_total_tokens", "num_continuous_decode_steps", "schedule_conservativeness", "disable_overlap_schedule", "enable_two_batch_overlap", "enable_single_batch_overlap", "tbo_token_distribution_threshold", "max_prefill_tokens", "chunked_prefill_size", "schedule_policy", "scheduler_recv_interval", "schedule_low_priority_values_first", "enable_priority_scheduling", "priority_scheduling_preemption_threshold"],
            "paged attention": ["page_size"],
            "tensor parallelism": ["tensor_parallel_size"],
            "pipeline parallelism": ["pipeline_parallel_size"],
            "expert parallelism": ["ep_size", "moe_mode", "moe_group_by", "moe_num_experts", "moe_top_k"],
            "data parallelism": ["dp_size"],
            "chunked prefill": ["chunked_prefill_size", "max_prefill_tokens"],
            "quantization": ["quantization"],
            "multi-LoRA batching": ["enable_lora", "lora_model_dir", "max_lora_models", "max_lora_rank", "lora_dtype"]}

    openllm_parameters = {"quantization": ["quantize"],
                        "tensor parallelism": ["tensor_parallel_size"],
                        "pipeline parallelism": ["pipeline_parallel_size"]}
            
    tensorrt_llm_parameters = {"In-Flight Batching": ["dynamic_batch", "batching_type", "max_batch_size"],
                            "Paged Attention": ["kv_cache_config"],
                            "Multi-GPU": ["tensor_parallel_size", "moe_expert_parallel_size", "moe_tensor_parallel_size"],
                            "quantization": ["quant_config"],
                            "Speculative Decoding": ["speculative_decoding_mode", "speculative_config"],
                            "KV Cache Management": ["kv_cache_config", "cache_transceiver_config", "kv_cache_retention_config"],
                            "Chunked Prefill": ["enable_chunked_prefill"],
                            "LoRA Support": ["enable_lora", "lora_config", "max_lora_rank"],
                            "Guided Decoding": ["guided_decoding_backend"],
                            "Disaggregated Serving": ["disaggregated_params"]}

    lmdeploy_parameters = {"persistent batch": ["max_batch_size", "max_prefill_token_num", "max_prefill_iters", "session_len", "prefill_interval"],
                        "KV-cache": ["cache_max_entry_count", "enable_prefix_caching", "quant_policy", "eviction_type", "cache_chunk_size", "cache_block_seq_len"],
                        "dynamic split": ["max_prefill_token_num", "num_tokens_per_iter", "max_prefill_iters", "prefill_interval"],
                        "tensor parallelism": ["tp", "attn_tp_size", "mlp_tp_size", "moe_tp_size"],
                        "high-performance CUDA kernels": ["dtype", "use_logn_attn", "communicator", "devices", "device_type"],
                        "Quantization": ["model_format", "quant_policy"],
                        "Automatic Prefix Caching": ["enable_prefix_caching"]}


    if tool_name == "vllm":        return vllm_parameters
    elif tool_name == "sglang":     return sglan_parameters
    elif tool_name == "openllm":    return openllm_parameters
    elif tool_name == "tensorrt-llm": return tensorrt_llm_parameters
    elif tool_name == "lmdeploy":   return lmdeploy_parameters
    else: raise ValueError(f"Unknown tool name: {tool_name}")   



def get_imports(tool_name):
    # initial version of imports

    # vllm_imports = {"CUDA":["config.CUDAGraphMode"],
    #                 "Quantization":["awq.AutoAWQForCausalLM","gptqmodel.GPTQModel","gptqmodel.QuantizeConfig","llmcompressor.transformers.oneshot","llmcompressor.modifiers.quantization.GPTQModifier","llmcompressor.modifiers.smoothquant.SmoothQuantModifier","llmcompressor.modifiers.quantization.QuantizationModifier"],
    #                 "Multi-LoRA":["lora.request.LoRARequest"]}

    # sglang_imports = {"RadixAttention": ["srt.mem_cache.radix", "srt.mem_cache.radix_cache", "srt.mem_cache.hiradix", "srt.mem_cache.mem_cache", "srt.entrypoints.engine.Engine"],
    #                 "zero overhead": ["srt.managers.schedule_batch", "srt.managers.schedule_policy", "srt.managers.scheduler", "srt.managers.zero_overhead_scheduler", "srt.managers.scheduling_utils", "srt.server.scheduler_manager"],
    #                 "prefill-decode disaggregation": ["srt.server.prefill_decode", "srt.managers.prefill_manager", "srt.managers.decode_manager", "srt.server.prefill_decode.prefill_decode_manager"],
    #                 "speculative decoding": ["srt.speculative.speculative_engine", "srt.speculative.speculative_draft_model", "srt.speculative.speculative_manager", "srt.speculative.speculative_worker", "srt.speculative"],
    #                 "continuous batching": ["srt.managers.continuous_batch_manager", "srt.managers.batch_scheduler", "srt.managers.batch_manager", "srt.server.batch_worker"],
    #                 "paged attention": ["srt.attention.paged_attention", "srt.attention.attention_backend", "srt.kv_cache.kv_manager", "srt.kv_cache.paged_kv_cache"],
    #                 "tensor parallelism": ["srt.parallel.tensor_parallel", "srt.distributed.process_group", "srt.parallel", "srt.distributed"],
    #                 "pipeline parallelism": ["srt.parallel.pipeline_parallel", "srt.distributed.distributed_backend", "srt.parallel", "srt.distributed"],
    #                 "expert parallelism": ["srt.moe.expert_parallel", "srt.moe.moe_manager", "srt.moe"],
    #                 "data parallelism": ["srt.parallel.data_parallel", "srt.distributed.process_group", "srt.parallel", "srt.distributed"],
    #                 "all parallelism": ["srt.parallel.tensor_parallel", "srt.parallel.pipeline_parallel", "srt.parallel.data_parallel", "srt.distributed.distributed_backend", "srt.distributed.process_group", "srt.moe.expert_parallel", "srt.moe.moe_manager", "srt.parallel", "srt.distributed", "srt.moe"],
    #                 "structured outputs": ["api.structured_output", "api.json_schema", "api.grammar_parser", "srt.struct.constrained_decoding", "srt.struct.schema_validator", "srt.struct.structure_manager", "srt.struct"],
    #                 "chunked prefill": ["srt.server.prefill_decode", "srt.server.prefill_worker", "srt.server.prefill_decode_manager", "srt.managers.prefill_manager", "srt.managers.decode_manager", "srt.managers.scheduling_utils", "srt.scheduler.chunked_prefill_scheduler", "srt.scheduler.chunk_manager"],
    #                 "quantization": ["srt.layers.quantization.fp8_utils", "srt.layers.quantization.quant_config", "srt.layers.quantization.quant_manager", "srt.layers.quantization.quant_utils", "srt.layers.quantization.awq_utils", "srt.layers.quantization.gptq_utils", "srt.layers.quantization.smoothquant_utils", "srt.backend.torchao_quant", "srt.backend.quant_backend", "srt.runtime.quant_loader", "srt.layers.quantization"],
    #                 "multi-LoRA batching": ["srt.lora", "srt.lora.lora_manager", "srt.lora.lora", "srt.lora.mem_pool", "srt.lora.utils", "srt.lora.lora_registry", "srt.lora.backend.triton_backend", "srt.lora.backend.chunked_backend", "srt.lora.lora_manager.LoRAManager", "srt.lora.lora.LoRAAdapter", "srt.lora.backend.chunked_backend.ChunkedSgmvLoRABackend"]
    #                 }

    # tensorrt_llm_imports = {"In-Flight Batching": ["llmapi.DynamicBatchConfig", "llmapi.BatchingType", "llmapi.llm_args.BatchingType", "llmapi.llm_args.SchedulerConfig", "llmapi.llm_args.DynamicBatchConfig"],
    #                         "Paged Attention": ["llmapi.KvCacheConfig", "llmapi.llm_args.KvCacheConfig"],
    #                         "Multi-GPU Multi-Node": ["bindings.executor"],
    #                         "Quantization": ["llmapi.QuantConfig", "llmapi.QuantAlgo", "quantization", "models.modeling_utils.QuantConfig", "models.LLaMAForCausalLM"],
    #                         "Speculative Decoding": ["llmapi.llm_args.MTPDecodingConfig", "llmapi.llm_args.LookaheadDecodingConfig", "llmapi.llm_args.DraftTargetDecodingConfig", "llmapi.llm_args.MedusaDecodingConfig", "llmapi.llm_args.EagleDecodingConfig", "llmapi.llm_args.NGramDecodingConfig", "llmapi.LookaheadDecodingConfig", "llmapi.MedusaDecodingConfig", "llmapi.EagleDecodingConfig", "llmapi.MTPDecodingConfig", "llmapi.NGramDecodingConfig", "llmapi.DraftTargetDecodingConfig", "llmapi.BuildConfig", "models.modeling_utils.SpeculativeDecodingMode"],
    #                         "KV Cache Management": ["llmapi.KvCacheConfig", "llmapi.llm_args.KvCacheConfig", "llmapi.llm_args.CacheTransceiverConfig", "llmapi.KvCacheRetentionConfig", "bindings.executor.KvCacheTransferMode"],
    #                         "Chunked Prefill": ["llmapi.ContextChunkingPolicy"],
    #                         "LoRA Support": ["executor.LoRARequest", "lora_helper.LoraConfig", "lora_manager.LoraConfig", "lora_manager.LoraManager"],
    #                         "Guided Decoding": ["llmapi.GuidedDecodingParams", "sampling_params.GuidedDecodingParams"],
    #                         "Disaggregated Serving": ["disaggregated_params.DisaggregatedParams"]
    #                         }

    # lmdeploy_imports = {"Quantization": ["lite.apis.auto_awq"]}

    # flashinfer_imports = {"FlashInfer Attention Kernels":["decode","xqa","prefill","mla"],
    #                     "flashinfer.gemm":["gemm"],
    #                     "fused_moe":["fused_moe"],
    #                     "cascade":["cascade"],
    #                     "comm":["comm","comm.mnnvl","comm.trtllm_mnnvl_ar"],
    #                     "sparse":["sparse"],
    #                     "page":["page"],
    #                     "sampling":["sampling"],
    #                     "topk":["topk"],
    #                     "logits_processor":["logits_processor"],
    #                     "norm":["norm"],
    #                     "rope":["rope"],
    #                     "activation":["activation"],
    #                     "quantization":["quantization","fp4_quantization"],
    #                     "green_ctx":["green_ctx"],
    #                     "testing":["testing"]
    #                     } 


    # # simplified imports, removed repetitive main imports like "srt" 
    # vllm_imports = {"CUDA":["CUDAGraphMode"],
    #             "Quantization":["awq","gptqmodel.QuantizeConfig","QuantizeConfig","llmcompressor","llmcompressor.transformers.oneshot","llmcompressor.modifiers.quantization.GPTQModifier","llmcompressor.modifiers.smoothquant.SmoothQuantModifier","llmcompressor.modifiers.quantization.QuantizationModifier"],
    #             "Multi-LoRA":["lora.request.LoRARequest","LoRAConfig"],
    #             "Speculative decoding": ["SpeculativeConfig"],
    #             "Chunked prefill": ["SchedulerConfig"],
    #             "parallelism": ["ParallelConfig"],                
    #             "KVCaching": ["CacheConfig","KVTransferConfig","KVEventsConfig"]}
    # sglang_imports = {"RadixAttention": ["mem_cache.radix", "mem_cache.radix_cache", "mem_cache.hiradix", "mem_cache.mem_cache", "entrypoints.engine.Engine"],
    #             "zero overhead": ["schedule_batch", "schedule_policy", "scheduler", "zero_overhead_scheduler", "scheduling_utils", "scheduler_manager"],
    #             "prefill-decode disaggregation": ["prefill_decode", "prefill_manager", "decode_manager", "prefill_decode.prefill_decode_manager"],
    #             "speculative decoding": ["speculative.speculative_engine", "speculative.speculative_draft_model", "speculative.speculative_manager", "speculative.speculative_worker", "speculative"],
    #             "continuous batching": ["continuous_batch_manager", "batch_scheduler", "batch_manager", "batch_worker"],
    #             "paged attention": ["attention.paged_attention", "attention.attention_backend", "kv_cache.kv_manager", "kv_cache.paged_kv_cache"],
    #             "tensor parallelism": ["parallel.tensor_parallel", "distributed.process_group", "parallel", "distributed"],
    #             "pipeline parallelism": ["parallel.pipeline_parallel", "distributed.distributed_backend", "parallel", "distributed"],
    #             "expert parallelism": ["moe.expert_parallel", "moe.moe_manager", "moe"],
    #             "data parallelism": ["parallel.data_parallel", "distributed.process_group", "parallel", "distributed"],
    #             "all parallelism": ["parallel.tensor_parallel", "parallel.pipeline_parallel", "parallel.data_parallel", "distributed.distributed_backend", "distributed.process_group", "moe.expert_parallel", "moe.moe_manager", "parallel", "distributed", "moe"],
    #             "structured outputs": ["api.structured_output", "api.json_schema", "api.grammar_parser", "struct.constrained_decoding", "struct.schema_validator", "struct.structure_manager", "struct"],
    #             "chunked prefill": ["prefill_decode", "prefill_worker", "prefill_decode_manager", "prefill_manager", "decode_manager", "scheduling_utils", "scheduler.chunked_prefill_scheduler", "scheduler.chunk_manager"],
    #             "quantization": ["layers.quantization.fp8_utils", "layers.quantization.quant_config", "layers.quantization.quant_manager", "layers.quantization.quant_utils", "layers.quantization.awq_utils", "layers.quantization.gptq_utils", "layers.quantization.smoothquant_utils", "backend.torchao_quant", "backend.quant_backend", "runtime.quant_loader", "layers.quantization"],
    #             "multi-LoRA batching": ["lora", "lora.lora_manager", "lora.lora", "lora.mem_pool", "lora.utils", "lora.lora_registry", "lora.backend.triton_backend", "lora.backend.chunked_backend", "lora.lora_manager.LoRAManager", "lora.lora.LoRAAdapter", "lora.backend.chunked_backend.ChunkedSgmvLoRABackend"]
    #             }
    # tensorrt_llm_imports = {"In-Flight Batching": ["DynamicBatchConfig", "BatchingType", "BatchingType", "SchedulerConfig", "DynamicBatchConfig"],
    #                     "Paged Attention": ["KvCacheConfig", "KvCacheConfig"],
    #                     "Multi-GPU Multi-Node": ["bindings.executor"],
    #                     "Quantization": ["QuantConfig", "QuantAlgo", "quantization", "models.modeling_utils.QuantConfig", "models.LLaMAForCausalLM"],
    #                     "Speculative Decoding": ["MTPDecodingConfig", "LookaheadDecodingConfig", "DraftTargetDecodingConfig", "MedusaDecodingConfig", "EagleDecodingConfig", "NGramDecodingConfig", "LookaheadDecodingConfig", "MedusaDecodingConfig", "EagleDecodingConfig", "MTPDecodingConfig", "NGramDecodingConfig", "DraftTargetDecodingConfig", "BuildConfig", "models.modeling_utils.SpeculativeDecodingMode"],
    #                     "KV Cache Management": ["KvCacheConfig", "KvCacheConfig", "CacheTransceiverConfig", "KvCacheRetentionConfig", "bindings.executor.KvCacheTransferMode"],
    #                     "Chunked Prefill": ["ContextChunkingPolicy"],
    #                     "LoRA Support": ["LoRARequest", "lora_helper.LoraConfig", "lora_manager.LoraConfig", "lora_manager.LoraManager"],
    #                     "Guided Decoding": ["GuidedDecodingParams", "sampling_params.GuidedDecodingParams"],
    #                     "Disaggregated Serving": ["disaggregated_params.DisaggregatedParams"]
    #                     }
    # lmdeploy_imports = {"Quantization": ["lite.apis.auto_awq"]}
    # flashinfer_imports = {"FlashInfer Attention Kernels":["decode","xqa","prefill","mla"],
    #                     "flashinfer.gemm":["gemm"],
    #                     "fused_moe":["fused_moe"],
    #                     "cascade":["cascade"],
    #                     "comm":["comm","comm.mnnvl","comm.trtllm_mnnvl_ar"],
    #                     "sparse":["sparse"],
    #                     "page":["page"],
    #                     "sampling":["sampling"],
    #                     "topk":["topk"],
    #                     "logits_processor":["logits_processor"],
    #                     "norm":["norm"],
    #                     "rope":["rope"],
    #                     "activation":["activation"],
    #                     "quantization":["quantization","fp4_quantization"],
    #                     "green_ctx":["green_ctx"],
    #                     "testing":["testing"]
    #                     } 


    # second time simplified imports, removed repetitive main imports like "srt" 
    vllm_imports = {"CUDA":["CUDAGraphMode","CUDA"],
                "Quantization":["awq","Quantize","llmcompressor","quantization","smoothquant","SmoothQuantModifier","QuantizationModifier","Quant"],
                "Multi-LoRA":["lora","LoRARequest","LoRA"],
                "Speculative decoding": ["Speculative"],
                "Chunked prefill": ["Scheduler"],
                "parallelism": ["Parallel"],                
                "KVCaching": ["Cache","KVTransfer","KVEvents"]}
    
    sglang_imports = {"RadixAttention": ["mem_cache","radix", "radix_cache", "hiradix"],
                "zero overhead": ["zero_overhead"],
                "prefill-decode disaggregation": ["prefill_decode", "prefill", "decode"],
                "speculative decoding": ["speculative"],
                "continuous batching": ["scheduling", "scheduler","schedule","continuous_batch", "batch"],
                "paged attention": ["attention", "paged_attention", "attention_backend"],
                "tensor parallelism": ["parallel","tensor_parallel", "distributed"],
            
                "pipeline parallelism": ["pipeline_parallel", "distributed_backend", "parallel", "distributed"],

                "expert parallelism": ["moe","expert_parallel", "moe_manager"],

                "data parallelism": ["data_parallel", "distributed", "parallel", "distributed"],

                "all parallelism": ["tensor_parallel", "pipeline_parallel", "data_parallel", "distributed_backend", "process_group", "expert_parallel", "moe_manager", "parallel", "distributed", "moe"],

                "structured outputs": ["structured_output", "json_schema", "grammar_parser", "constrained_decoding", "schema_validator", "structure_manager"],

                "chunked prefill": ["prefill", "prefill_decode", "prefill_worker", "prefill_decode_manager", "prefill_manager", "decode_manager", "chunked_prefill", "chunk_manager"],

                "quantization": ["quantization","fp8_utils", "quantization","quant", "quant_manager", "quant_utils","awq", "smoothquant", "torchao_quant", "quant_backend", "quant_loader"],
                "multi-LoRA batching": ["lora", "lora_manager", "lora_registry", "lora_manager","LoRAManager", "LoRAAdapter", "ChunkedSgmvLoRABackend"],
                "kv_cache": ["kv_cache","kv_manager", "kv_cache"]}
                
    tensorrt_llm_imports = {"In-Flight Batching": ["BatchingType", "Scheduler", "DynamicBatch"],
                        "Paged Attention": ["KvCache"],
                        "Multi-GPU Multi-Node": ["bindings.executor"],
                        "Quantization": ["Quant", "QuantAlgo", "quantization"],
                        "Speculative Decoding": ["speculative", "Decoding","MTPDecoding", "LookaheadDecoding", "DraftTargetDecoding", "MedusaDecoding", "EagleDecoding", "NGramDecoding", "LookaheadDecoding", "MedusaDecoding", "EagleDecoding", "MTPDecoding", "NGramDecoding", "DraftTargetDecoding", "SpeculativeDecodingMode"],
                        "KV Cache Management": ["KvCache", "CacheTransceiver", "KvCacheRetention", "KvCacheTransferMode"],
                        "Chunked Prefill": ["Chunk","prefill","ContextChunkingPolicy"],
                        "LoRA Support": ["LoRARequest", "lora_helper","Lora", "lora_manager", "lora_manager","LoraManager"],
                        "Guided Decoding": ["GuidedDecodingParams"],
                        "Disaggregated Serving": ["disaggregated_params", "DisaggregatedParams"]
                        }

    lmdeploy_imports = {"Quantization": ["lite.apis.auto_awq"]}

    flashinfer_imports = {"FlashInfer Attention Kernels":["decode","xqa","prefill","mla"],
                        "flashinfer.gemm":["gemm"],
                        "fused_moe":["fused_moe"],
                        "cascade":["cascade"],
                        "comm":["comm","comm.mnnvl","comm.trtllm_mnnvl_ar"],
                        "sparse":["sparse"],
                        "page":["page"],
                        "sampling":["sampling"],
                        "topk":["topk"],
                        "logits_processor":["logits_processor"],
                        "norm":["norm"],
                        "rope":["rope"],
                        "activation":["activation"],
                        "quantization":["quantization","fp4_quantization"],
                        "green_ctx":["green_ctx"],
                        "testing":["testing"]
                        } 



    if tool_name == "vllm":        return vllm_imports
    elif tool_name == "sglang":     return sglang_imports
    elif tool_name == "tensorrt_llm": return tensorrt_llm_imports
    elif tool_name == "lmdeploy":   return lmdeploy_imports
    elif tool_name == "flashinfer": return flashinfer_imports   
    else: raise ValueError(f"Unknown tool name: {tool_name}")



def filter_imports_and_features_data(df):
    df = df.copy()

    base_keys_to_keep = {'sglang', 'vllm', 'tensorrt_llm', 'lmdeploy', 'flashinfer'}

    def filter_dict(d, keys):
        if isinstance(d, dict):
            return {k: v for k, v in d.items() if k in keys}
        return {}

    def filter_string(s, keys):
        if isinstance(s, str):
            items = [item.strip() for item in s.split(',')]
            filtered = [item for item in items if item in keys]
            return ', '.join(filtered)
        return ""

    # ---- Step 1: Filter imports_as using base keys ----
    df['filtered_imports_as'] = df['imports_as'].apply(
        lambda d: filter_dict(d, base_keys_to_keep)
    )

    # ---- Step 2: Collect all values from filtered_imports_as ----
    additional_keys = set()

    for d in df['filtered_imports_as']:
        if isinstance(d, dict):
            for v in d.values():
                if isinstance(v, (list, set, tuple)):
                    additional_keys.update(v)
                else:
                    additional_keys.add(v)

    # ---- Step 3: Expand keys_to_keep ----
    expanded_keys_to_keep = base_keys_to_keep.union(additional_keys)
    print("Expanded keys to keep:", expanded_keys_to_keep)
    # ---- Step 4: Apply filtering with expanded keys ----
    df['filtered_imports_features'] = df['imports_features'].apply(
        lambda d: filter_dict(d, expanded_keys_to_keep)
    )

    df['filtered_imports'] = df['imports'].apply(
        lambda s: filter_string(s, expanded_keys_to_keep)
    )

    return df



def to_list(x):
    """Convert one cell to a plain Python list."""
    if x is None:
        return []

    if isinstance(x, float) and pd.isna(x):
        return []

    if isinstance(x, list):
        return x

    if isinstance(x, (tuple, set, np.ndarray)):
        return list(x)

    if isinstance(x, str):
        x = x.strip()
        if x == "" or x == "[]":
            return []
        try:
            parsed = ast.literal_eval(x)
            if parsed is None:
                return []
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, (tuple, set, np.ndarray)):
                return list(parsed)
            return [parsed]
        except Exception:
            return [x]

    return [x]

def merge_lists(series):
    """Merge list-valued cells, keep unique values, preserve order."""
    merged = []
    seen = set()

    for cell in series:
        items = to_list(cell)
        for item in items:
            if item in ("", None):
                continue
            key = repr(item)
            if key not in seen:
                seen.add(key)
                merged.append(item)

    return merged




def add_reference_cols(df_all, reference_params, params_col):
    # token -> set of k3 columns it belongs to (fast lookup)
    token_to_k3s = {}
    for k3, tokens in reference_params.items():
        for tok in tokens:
            token_to_k3s.setdefault(tok, set()).add(k3)

    out = df_all.copy()

    # Ensure target columns exist
    for k3 in reference_params.keys():
        if k3 not in out.columns:
            out[k3] = None

    def extract_hits(params):
        hits = {k3: [] for k3 in reference_params.keys()}
        if not isinstance(params, dict):
            return hits

        for _, d2 in params.items():  # Dict1 values should be Dict2
            if not isinstance(d2, dict):
                continue

            for k2, v2 in d2.items():
                # match on key k2
                for k3 in token_to_k3s.get(k2, ()):
                    hits[k3].append(k2)

                # match on value v2 (only if hashable, typically str/int)
                try:
                    for k3 in token_to_k3s.get(v2, ()):
                        hits[k3].append(v2)
                except TypeError:
                    pass

        # unique-preserve-order; empty -> None
        for k3, lst in hits.items():
            if lst:
                hits[k3] = list(dict.fromkeys(lst))
            else:
                hits[k3] = None

        return hits

    # Expand dict-of-lists to columns and merge back
    hits_df = out[params_col].apply(extract_hits).apply(pd.Series)
    out.update(hits_df)

    return out



import pandas as pd
import ast
import re



# def add_reference_columns(
#     df_all,
#     reference_params,
#     params_col="params",
#     store="matched_substring",   # "matched_substring" or "matched_item"
#     case_sensitive=False):
#     def safe_isna(x):
#         try:
#             return pd.isna(x)
#         except Exception:
#             return False

#     def normalize_text(x):
#         if x is None or safe_isna(x):
#             return ""
#         s = str(x).strip()
#         if not case_sensitive:
#             s = s.lower()
#         s = s.replace("-", "_")
#         s = re.sub(r"[^a-zA-Z0-9_\.]+", " ", s)
#         s = re.sub(r"\s+", " ", s).strip()
#         return s

#     def split_terms(s):
#         if not s:
#             return []
#         parts = re.split(r"[.\s:/=,\(\)\[\]\{\}]+", s)
#         return [p for p in parts if p]

#     def parse_params(x):
#         if isinstance(x, dict):
#             return x
#         if x is None or safe_isna(x):
#             return {}
#         if isinstance(x, str):
#             s = x.strip()
#             if not s:
#                 return {}
#             try:
#                 return ast.literal_eval(s)
#             except Exception:
#                 return {"__raw__": s}
#         return {"__raw__": str(x)}

#     def walk_nested(obj):
#         found = []

#         if isinstance(obj, dict):
#             for k, v in obj.items():
#                 found.append(str(k))          # include keys
#                 found.extend(walk_nested(v))  # include nested values

#         elif isinstance(obj, (list, tuple, set)):
#             for item in obj:
#                 found.extend(walk_nested(item))

#         else:
#             if obj is not None and not safe_isna(obj):
#                 found.append(str(obj))

#         return found

#     def make_term_list(x):
#         if isinstance(x, (list, tuple, set)):
#             return list(x)
#         return [x]

#     # Normalize reference terms
#     ref_terms_by_col = {}
#     for ref_col, terms in reference_params.items():
#         norm_terms = []
#         for t in make_term_list(terms):
#             nt = normalize_text(t)
#             if nt:
#                 norm_terms.append(nt)
#         ref_terms_by_col[ref_col] = list(dict.fromkeys(norm_terms))

#     out = df_all.copy()

#     # create output columns
#     for ref_col in reference_params.keys():
#         out[ref_col] = ""

#     def extract_matches(params):
#         parsed = parse_params(params)
#         items = walk_nested(parsed)

#         hits = {ref_col: [] for ref_col in reference_params.keys()}

#         for raw_item in items:
#             norm_item = normalize_text(raw_item)
#             if not norm_item:
#                 continue

#             searchable_parts = [norm_item] + split_terms(norm_item)

#             for ref_col, ref_terms in ref_terms_by_col.items():
#                 for ref_term in ref_terms:
#                     matched = (
#                         ref_term == norm_item
#                         or ref_term in searchable_parts
#                         or any(ref_term in part for part in searchable_parts)
#                     )

#                     if matched:
#                         if store == "matched_item":
#                             hit_value = raw_item
#                         else:
#                             hit_value = ref_term
#                         hits[ref_col].append(hit_value)

#         # unique + preserve order, empty -> ""
#         for ref_col, vals in hits.items():
#             hits[ref_col] = list(dict.fromkeys(vals)) if vals else ""

#         return hits

#     hits_df = out[params_col].apply(extract_matches).apply(pd.Series)

#     for col in reference_params.keys():
#         out[col] = hits_df[col]

#     return out
    



import pandas as pd
import re
import ast

def add_reference_columns(
    df_all,
    reference_params,
    params_col="params",
    store="matched_substring",   # "matched_substring" or "matched_item"
    case_sensitive=False
):
    def safe_isna(x):
        try:
            return pd.isna(x)
        except Exception:
            return False

    def normalize_text(x):
        if x is None or safe_isna(x):
            return ""
        s = str(x).strip()
        if not case_sensitive:
            s = s.lower()
        s = s.replace("-", "_")
        s = re.sub(r"[^a-zA-Z0-9_\.]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def split_terms(s):
        if not s:
            return []
        parts = re.split(r"[.\s:/=,\(\)\[\]\{\}]+", s)
        return [p for p in parts if p]

    def parse_params(x):
        if isinstance(x, dict):
            return x
        if x is None or safe_isna(x):
            return {}
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return {}
            try:
                return ast.literal_eval(s)
            except Exception:
                return {"__raw__": s}
        return {"__raw__": str(x)}

    def walk_nested(obj):
        found = []

        if isinstance(obj, dict):
            for k, v in obj.items():
                found.append(str(k))
                found.extend(walk_nested(v))

        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                found.extend(walk_nested(item))

        else:
            if obj is not None and not safe_isna(obj):
                found.append(str(obj))

        return found

    def make_term_list(x):
        if isinstance(x, (list, tuple, set)):
            return list(x)
        return [x]

    # Normalize reference terms
    ref_terms_by_col = {}
    for ref_col, terms in reference_params.items():
        norm_terms = []
        for t in make_term_list(terms):
            nt = normalize_text(t)
            if nt:
                norm_terms.append(nt)
        ref_terms_by_col[ref_col] = list(dict.fromkeys(norm_terms))

    out = df_all.copy()

    def extract_matches(params):
        parsed = parse_params(params)
        items = walk_nested(parsed)

        hits = {ref_col: [] for ref_col in reference_params.keys()}

        for raw_item in items:
            norm_item = normalize_text(raw_item)
            if not norm_item:
                continue

            searchable_parts = [norm_item] + split_terms(norm_item)

            for ref_col, ref_terms in ref_terms_by_col.items():
                for ref_term in ref_terms:
                    matched = (
                        ref_term == norm_item
                        or ref_term in searchable_parts
                        or any(ref_term in part for part in searchable_parts)
                    )

                    if matched:
                        if store == "matched_item":
                            hit_value = raw_item
                        else:
                            hit_value = ref_term
                        hits[ref_col].append(hit_value)

        for ref_col, vals in hits.items():
            hits[ref_col] = list(dict.fromkeys(vals)) if vals else ""

        return hits

    hits_df = out[params_col].apply(extract_matches).apply(pd.Series)

    # final output: repo_full_name + new added columns only
    result = pd.concat(
        [out[["repo_full_name"]].reset_index(drop=True),
         hits_df[list(reference_params.keys())].reset_index(drop=True)],
        axis=1
    )

    return result



    
def summarize_repo(row, tool_name, client, model):
    about = str(row["about"])
    readme = str(row["readme"])
    keywords = str(row["keywords"])



    # prompt = f"""
    # You are analyzing a GitHub repository.

    # Repository about section:
    # {about}

    # README excerpt:
    # {readme}

    # keywords:
    # {keywords}

    # The repository uses {tool_name}, an LLM serving framework.

    # Your goal is to extract concise descriptions that capture the technical purpose,
    # design, and capabilities of the repository so that similar repositories can later
    # be grouped together using machine learning models.

    # Use ONLY the information contained in the README, about section, and keywords.

    # Return the output in EXACTLY the following format.
    # Each field must appear on ONE single line.

    # 1. repository_intent: <one concise sentence describing the main purpose of the repository>
    # 2. technical_focus: <one sentence describing the main technical components or ideas implemented>
    # 3. primary_use_cases: <one sentence describing the main tasks or workflows the repository supports>
    # 4. system_design: <one sentence describing the software architecture or system design>
    # 5. key_technical_terms: <5–10 comma-separated technical terms or phrases from the README, about section, or keywords>

    # Rules:
    # - Do not add titles, explanations, markdown, or extra text.
    # - Do not change field names or numbering.
    # - All the fields and all the sentences SHOULD BE WRITTEN IN ENGLISH even if the README, about, or keywords texts are in another language.
    # - Do not use bold or formatting.
    # - Each field must remain on a single line.
    # - Use only information explicitly present in README, about, or keywords.
    # - Do not infer or invent missing information.
    # - If a field cannot be determined from the text, leave it empty after the colon.
    # """


    prompt = f"""
    You are analyzing a GitHub repository.

    Repository about section:
    {about}

    README excerpt:
    {readme}

    keywords:
    {keywords}

    The repository uses {tool_name}, an LLM serving framework.

    Your goal is to extract concise technical keywords and phrases that capture the repository's purpose,
    design, and capabilities so that similar repositories can later be grouped together using machine learning models.

    Use ONLY the information contained in the README, about section, and keywords.

    Return the output in EXACTLY the following format.
    Each field must appear on ONE single line.

    1. repository_intent: <exactly 10 comma-separated technical keywords or short phrases describing the repository’s main purpose>
    2. technical_focus: <exactly 10 comma-separated technical keywords or short phrases describing the main technical components, methods, or ideas implemented>
    3. primary_use_cases: <exactly 10 comma-separated technical keywords or short phrases describing the main tasks, primary_use_cases, applications, or workflows the repository supports>
    4. system_design: <exactly 10 comma-separated technical keywords or short phrases describing the software architecture, serving design, execution model, or infrastructure>

    Rules:
    - Do not add titles, explanations, markdown, or extra text.
    - Do not change field names or numbering.
    - All fields must be written in ENGLISH even if the README, about, or keywords texts are in another language.
    - Do not use bold or formatting.
    - Each field must remain on a single line.
    - Use only information explicitly present in README, about, or keywords.
    - Do not infer or invent missing information.
    - Prefer technical nouns and noun phrases over full sentences.
    - Prefer repository-specific terms over generic words like "tool", "system", "framework", or "support".
    - Avoid repeated keywords within the same field.
    - Keep each keyword or phrase short, ideally 1 to 4 words.
    - Separate items with commas only.
    - If fewer than 10 valid items are explicitly available for a field, return only the available items after the colon.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You summarize Github repositories."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(e)
        return ""




# Get related used imports
def coerce_to_dict(x):
    """Convert stringified dict to dict if needed."""
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        x = x.strip()
        if x.startswith("{") and x.endswith("}"):
            try:
                return ast.literal_eval(x)
            except Exception:
                return None
    return None


def flatten_v1_strings(d1):
    """Collect all strings from dictionary values."""
    out = []
    if not isinstance(d1, dict):
        return out

    for val in d1.values():
        if isinstance(val, (list, tuple, set)):
            out.extend([s for s in val if isinstance(s, str)])
        elif isinstance(val, str):
            out.append(val)
    return out


def find_matches(d1, imports_dict, normalize=True):
    """
    Returns:
        dict {k2: [matched_v2,...]}

    A match occurs when a substring from v2 appears anywhere in v1.
    Matching is case-insensitive when normalize=True.
    """

    d1 = coerce_to_dict(d1)
    v1_strings = flatten_v1_strings(d1)

    # initialize output
    out = {k2: [] for k2 in imports_dict.keys()}

    if not v1_strings:
        return out

    # normalize v1 strings
    if normalize:
        v1_strings = [s.strip().lower() for s in v1_strings]

    for k2, v2_list in imports_dict.items():

        hits = set()

        for v2 in v2_list:

            if not isinstance(v2, str):
                continue

            css = v2.strip().lower() if normalize else v2.strip()

            if any(css in v1 for v1 in v1_strings):
                hits.add(css if normalize else v2)

        out[k2] = sorted(hits)

    return out



###################################################################################################
# topic modeling

def data_cleaning(df, target_column, fasttext_model_path=None):
    import re
    import html
    import pandas as pd
    import contractions
    import spacy
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    fasttext_model = None
    if fasttext_model_path is not None:
        import fasttext
        fasttext_model = fasttext.load_model(fasttext_model_path)

    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in dataframe.")

    out = df.copy()

    url_pattern = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
    html_tag_pattern = re.compile(r"<[^>]+>")
    multi_space_pattern = re.compile(r"\s+")
    token_pattern = re.compile(r"\b[a-zA-Z]+\b")

    custom_stopwords = {
        "vllm", "sglang", "tensorrt", "tensorrtllm", "lmdeploy", "flashinfer", "llm", "llms",
        "repository", "repositories",
        "supports", "support",
        "implements", "implement", "implemented",
        "utilizes", "utilize", "using",
        "includes", "include",
        "provides", "provide",
        "technical", "focus",
        "software", "architecture",
        "task", "tasks",
        "component", "components",
        "feature", "features",
        "model", "models",
        "framework", "frameworks",
        "various", "related", "designed", "based"
    }
    stop_words = set(ENGLISH_STOP_WORDS).union(custom_stopwords)

    def normalize_text(text):
        if pd.isna(text):
            return None
        text = str(text)
        text = html.unescape(text)
        text = url_pattern.sub(" ", text)
        text = html_tag_pattern.sub(" ", text)

        # remove repeated template phrases
        text = re.sub(r"\blarge language models?\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\brepository intent\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\btechnical focus\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\bprimary use cases?\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\bsystem design\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\bsoftware architecture\b", " ", text, flags=re.IGNORECASE)

        text = contractions.fix(text)
        text = multi_space_pattern.sub(" ", text).strip()
        text = text.lower()
        return text if text else None

    def is_deleted_or_removed(text):
        if text is None:
            return True
        stripped = text.strip().lower()
        return stripped in {"[deleted]", "[removed]", "deleted", "removed", ""}

    def is_english_fasttext(text):
        if fasttext_model is None:
            return True
        if text is None or not text.strip():
            return False
        label, prob = fasttext_model.predict(text.replace("\n", " "), k=1)
        return label[0] == "__label__en"

    def regex_tokenize(text):
        return token_pattern.findall(text.lower())

    out[target_column] = out[target_column].apply(normalize_text)

    mask_valid = out[target_column].notna()
    mask_valid &= ~out[target_column].apply(is_deleted_or_removed)
    mask_valid &= out[target_column].apply(is_english_fasttext)

    out = out.loc[mask_valid].copy()
    out = out.drop_duplicates(subset=[target_column]).reset_index(drop=True)

    def slightly_preprocess(text):
        tokens = regex_tokenize(text)
        tokens = [tok for tok in tokens if tok not in stop_words]
        return " ".join(tokens)

    def fully_preprocess(text):
        doc = nlp(text)
        lemmas = []
        for token in doc:
            if not token.is_alpha:
                continue
            tok = token.text.lower().strip()
            if tok in stop_words or token.is_stop:
                continue
            if token.pos_ in {"NOUN", "VERB"}:
                lemma = token.lemma_.strip().lower()
                if lemma and lemma not in stop_words and lemma.isalpha():
                    lemmas.append(lemma)
        return " ".join(lemmas)

    slight_col = f"{target_column}_slightly_preprocessed"
    full_col = f"{target_column}_fully_preprocessed"

    out[slight_col] = out[target_column].apply(slightly_preprocess)
    out[full_col] = out[target_column].apply(fully_preprocess)

    return out


def evaluate_topic_numbers_bertopic_hdbscan_with_k_reduction(
    df,
    target_column,
    k_values=range(5, 20, 3),
    embedding_model_name="all-MiniLM-L6-v2",
    umap_n_neighbors=15,
    umap_n_components=5,
    umap_min_dist=0.0,
    umap_metric="cosine",
    hdbscan_min_cluster_size=10,
    hdbscan_min_samples=None,
    hdbscan_metric="euclidean",
    top_n_words=10,
    vectorizer_param_grid=None,
    verbose=True
):
    """
    Evaluate BERTopic + UMAP + HDBSCAN on:
      1) original text
      2) slightly preprocessed text
      3) fully preprocessed text

    For each dataset:
      - fit BERTopic for each CountVectorizer hyperparameter combination
      - store the initial detected topic result before reduction
      - reduce topics to each k in k_values
      - compute CV and UMass coherence
      - store topic sizes and keywords for each initial/reduced run
      - store and print results for all runs

    Returns
    -------
    dict
        {
            "original_results": DataFrame,
            "slight_results": DataFrame,
            "full_results": DataFrame,
            "all_results": DataFrame,
            "original_models": list,
            "slight_models": list,
            "full_models": list,
            "original_docs_used": list,
            "slight_docs_used": list,
            "full_docs_used": list,
            "run_registry": dict
        }
    """
    import numpy as np
    import pandas as pd
    import copy
    from itertools import product

    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    import umap.umap_ as umap
    import hdbscan

    from sklearn.feature_extraction.text import CountVectorizer
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel

    slight_col = f"{target_column}_slightly_preprocessed"
    full_col = f"{target_column}_fully_preprocessed"

    required_cols = [target_column, slight_col, full_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Default wide hyperparameter search space
    if vectorizer_param_grid is None:
        vectorizer_param_grid = {
            "ngram_range": [(1, 1), (1, 2), (1, 3)],
            "min_df": [1, 2, 3],
            "max_df": [0.70, 0.80, 0.90]
        }
        # vectorizer_param_grid = {
        #     "ngram_range": [(1, 1), (1, 2)],
        #     "min_df": [1, 2],
        #     "max_df": [0.80, 0.90]
        # }

    # Build all hyperparameter combinations
    vectorizer_configs = list(product(
        vectorizer_param_grid["ngram_range"],
        vectorizer_param_grid["min_df"],
        vectorizer_param_grid["max_df"]
    ))

    embedding_model = SentenceTransformer(embedding_model_name)

    # Stores per-run topic details so the best run can be printed later
    run_registry = {}
    run_counter = 0

    def _tokenize_for_coherence(docs):
        return [str(doc).split() for doc in docs]

    def _extract_topic_words(topic_model, top_n=10):
        topic_words = []
        topics_dict = topic_model.get_topics()

        for topic_id, word_scores in topics_dict.items():
            if topic_id == -1:
                continue
            words = [word for word, _ in word_scores[:top_n] if word and word.strip()]
            if len(words) > 0:
                topic_words.append(words)

        return topic_words

    def _compute_coherence(topic_words, tokenized_docs):
        if len(topic_words) == 0:
            return np.nan, np.nan

        dictionary = Dictionary(tokenized_docs)
        corpus = [dictionary.doc2bow(text) for text in tokenized_docs]

        vocab = set(dictionary.token2id.keys())
        filtered_topic_words = []
        for topic in topic_words:
            filtered = [w for w in topic if w in vocab]
            if len(filtered) > 0:
                filtered_topic_words.append(filtered)

        if len(filtered_topic_words) == 0:
            return np.nan, np.nan

        cv_model = CoherenceModel(
            topics=filtered_topic_words,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence="c_v"
        )
        umass_model = CoherenceModel(
            topics=filtered_topic_words,
            corpus=corpus,
            dictionary=dictionary,
            coherence="u_mass"
        )

        cv_score = cv_model.get_coherence()
        umass_score = umass_model.get_coherence()
        return cv_score, umass_score

    def _summarize_topics(topic_model, top_n=10):
        """
        Returns a DataFrame with:
          - topic_id
          - n_samples
          - keywords
        """
        info = topic_model.get_topic_info().copy()

        if "Topic" not in info.columns:
            return pd.DataFrame(columns=["topic_id", "n_samples", "keywords"])

        info = info[info["Topic"] != -1].copy()   # remove noise topic

        if info.empty:
            return pd.DataFrame(columns=["topic_id", "n_samples", "keywords"])

        def _topic_keywords(topic_id):
            words_scores = topic_model.get_topic(topic_id)
            if words_scores is None:
                return []
            return [word for word, _ in words_scores[:top_n]]

        info["keywords"] = info["Topic"].apply(_topic_keywords)

        info = info.rename(columns={
            "Topic": "topic_id",
            "Count": "n_samples"
        })

        return info[["topic_id", "n_samples", "keywords"]].sort_values("topic_id").reset_index(drop=True)

    def _fit_and_evaluate_one(texts, dataset_name):
        nonlocal run_counter, run_registry

        texts = pd.Series(texts).fillna("").astype(str)
        texts = texts[texts.str.strip() != ""].reset_index(drop=True)
        docs = texts.tolist()

        tokenized_docs = _tokenize_for_coherence(docs)

        all_rows = []
        fitted_models = []

        for ngram_range, min_df, max_df in vectorizer_configs:
            if verbose:
                print("\n" + "=" * 90)
                print(f"Dataset: {dataset_name}")
                print(f"Vectorizer config -> ngram_range={ngram_range}, min_df={min_df}, max_df={max_df}")
                print("=" * 90)

            config_id = f"{dataset_name}__{ngram_range}__{min_df}__{max_df}"

            try:
                umap_model = umap.UMAP(
                    n_neighbors=umap_n_neighbors,
                    n_components=umap_n_components,
                    min_dist=umap_min_dist,
                    metric=umap_metric,
                    random_state=42
                )

                hdbscan_model = hdbscan.HDBSCAN(
                    min_cluster_size=hdbscan_min_cluster_size,
                    min_samples=hdbscan_min_samples,
                    metric=hdbscan_metric,
                    cluster_selection_method="eom",
                    prediction_data=True
                )

                vectorizer_model = CountVectorizer(
                    ngram_range=ngram_range,
                    min_df=min_df,
                    max_df=max_df
                )

                topic_model = BERTopic(
                    embedding_model=embedding_model,
                    umap_model=umap_model,
                    hdbscan_model=hdbscan_model,
                    vectorizer_model=vectorizer_model,
                    top_n_words=top_n_words,
                    verbose=verbose
                )

                topics, probs = topic_model.fit_transform(docs)

                fitted_models.append({
                    "dataset": dataset_name,
                    "ngram_range": ngram_range,
                    "min_df": min_df,
                    "max_df": max_df,
                    "config_id": config_id,
                    "model": topic_model
                })

                n_noise_initial = int((np.array(topics) == -1).sum())
                noise_ratio_initial = n_noise_initial / len(topics)

                # ---------------------------------------------------------
                # NEW: store the initial detected result before reduction
                # ---------------------------------------------------------
                initial_topic_ids = [
                    tid for tid in topic_model.get_topics().keys()
                    if tid != -1
                ]
                initial_n_topics = len(initial_topic_ids)

                initial_topic_words = _extract_topic_words(topic_model, top_n=top_n_words)
                initial_cv_score, initial_umass_score = _compute_coherence(
                    initial_topic_words, tokenized_docs
                )

                initial_topic_summary = _summarize_topics(topic_model, top_n=top_n_words)

                run_key = f"run_{run_counter}"
                run_registry[run_key] = {
                    "topic_summary": initial_topic_summary.copy(),
                    "model": topic_model,
                    "dataset": dataset_name,
                    "config_id": config_id,
                    "topic_stage": "initial_detected"
                }
                run_counter += 1

                initial_row = {
                    "dataset": dataset_name,
                    "config_id": config_id,
                    "topic_stage": "initial_detected",
                    "ngram_range": ngram_range,
                    "min_df": min_df,
                    "max_df": max_df,
                    "initial_k_detected": initial_n_topics,
                    "k_requested": initial_n_topics,
                    "k_actual": initial_n_topics,
                    "cv_coherence": initial_cv_score,
                    "umass_coherence": initial_umass_score,
                    "n_noise": n_noise_initial,
                    "noise_ratio": noise_ratio_initial,
                    "run_key": run_key,
                    "error": None
                }
                all_rows.append(initial_row)

                if verbose:
                    print("Initial detected result:")
                    print(initial_row)

                # ---------------------------------------------------------
                # Existing reduced-k evaluation loop, now storing more info
                # ---------------------------------------------------------
                for k in k_values:
                    if verbose:
                        print(f"  Evaluating k={k}")

                    try:
                        reduced_model = copy.deepcopy(topic_model)
                        reduced_model.reduce_topics(docs, nr_topics=k)

                        topic_words = _extract_topic_words(reduced_model, top_n=top_n_words)
                        cv_score, umass_score = _compute_coherence(topic_words, tokenized_docs)

                        actual_topic_ids = [
                            tid for tid in reduced_model.get_topics().keys()
                            if tid != -1
                        ]
                        actual_n_topics = len(actual_topic_ids)

                        reduced_topic_summary = _summarize_topics(reduced_model, top_n=top_n_words)

                        run_key = f"run_{run_counter}"
                        run_registry[run_key] = {
                            "topic_summary": reduced_topic_summary.copy(),
                            "model": reduced_model,
                            "dataset": dataset_name,
                            "config_id": config_id,
                            "topic_stage": "reduced"
                        }
                        run_counter += 1

                        row = {
                            "dataset": dataset_name,
                            "config_id": config_id,
                            "topic_stage": "reduced",
                            "ngram_range": ngram_range,
                            "min_df": min_df,
                            "max_df": max_df,
                            "initial_k_detected": initial_n_topics,
                            "k_requested": k,
                            "k_actual": actual_n_topics,
                            "cv_coherence": cv_score,
                            "umass_coherence": umass_score,
                            "n_noise": n_noise_initial,
                            "noise_ratio": noise_ratio_initial,
                            "run_key": run_key,
                            "error": None
                        }
                        all_rows.append(row)

                        if verbose:
                            print(row)

                    except Exception as e_k:
                        row = {
                            "dataset": dataset_name,
                            "config_id": config_id,
                            "topic_stage": "reduced",
                            "ngram_range": ngram_range,
                            "min_df": min_df,
                            "max_df": max_df,
                            "initial_k_detected": initial_n_topics,
                            "k_requested": k,
                            "k_actual": np.nan,
                            "cv_coherence": np.nan,
                            "umass_coherence": np.nan,
                            "n_noise": n_noise_initial,
                            "noise_ratio": noise_ratio_initial,
                            "run_key": None,
                            "error": str(e_k)
                        }
                        all_rows.append(row)

                        if verbose:
                            print(row)

            except Exception as e_fit:
                # If the BERTopic fit itself fails for a vectorizer config,
                # still record rows for all k values
                for k in k_values:
                    row = {
                        "dataset": dataset_name,
                        "config_id": config_id,
                        "topic_stage": "fit_failed",
                        "ngram_range": ngram_range,
                        "min_df": min_df,
                        "max_df": max_df,
                        "initial_k_detected": np.nan,
                        "k_requested": k,
                        "k_actual": np.nan,
                        "cv_coherence": np.nan,
                        "umass_coherence": np.nan,
                        "n_noise": np.nan,
                        "noise_ratio": np.nan,
                        "run_key": None,
                        "error": f"fit_failed: {str(e_fit)}"
                    }
                    all_rows.append(row)

                    if verbose:
                        print(row)

        results_df = pd.DataFrame(all_rows)
        return fitted_models, results_df, docs

    original_models, original_results, original_docs_used = _fit_and_evaluate_one(
        df[target_column], "original"
    )
    slight_models, slight_results, slight_docs_used = _fit_and_evaluate_one(
        df[slight_col], "slightly_preprocessed"
    )
    full_models, full_results, full_docs_used = _fit_and_evaluate_one(
        df[full_col], "fully_preprocessed"
    )

    all_results = pd.concat(
        [original_results, slight_results, full_results],
        axis=0,
        ignore_index=True
    )

    print("\n" + "#" * 100)
    print("ALL RESULTS")
    print("#" * 100)
    print(all_results)

    return {
        "original_results": original_results,
        "slight_results": slight_results,
        "full_results": full_results,
        "all_results": all_results,
        "original_models": original_models,
        "slight_models": slight_models,
        "full_models": full_models,
        "original_docs_used": original_docs_used,
        "slight_docs_used": slight_docs_used,
        "full_docs_used": full_docs_used,
        "run_registry": run_registry
    }






def select_and_print_best_config(results):
    import numpy as np
    import pandas as pd

    all_results = results["all_results"].copy()
    run_registry = results["run_registry"]

    required_cols = [
        "dataset",
        "config_id",
        "topic_stage",
        "ngram_range",
        "min_df",
        "max_df",
        "initial_k_detected",
        "k_requested",
        "k_actual",
        "cv_coherence",
        "umass_coherence",
        "n_noise",
        "noise_ratio",
        "run_key",
        "error"
    ]

    missing = [c for c in required_cols if c not in all_results.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in results['all_results']: {missing}. "
            "Please first update the original evaluation function to store these fields."
        )

    if "run_registry" not in results:
        raise ValueError(
            "results does not contain 'run_registry'. "
            "Please update the original function to store per-run topic details."
        )

    def _minmax_normalize(series):
        s = pd.to_numeric(series, errors="coerce")
        out = pd.Series(np.nan, index=series.index, dtype=float)

        valid = s.dropna()
        if valid.empty:
            return out

        s_min = valid.min()
        s_max = valid.max()

        if np.isclose(s_min, s_max):
            out.loc[valid.index] = 1.0
        else:
            out.loc[valid.index] = (valid - s_min) / (s_max - s_min)

        return out

    def _print_metrics(row):
        metric_cols = [
            "dataset",
            "topic_stage",
            "ngram_range",
            "min_df",
            "max_df",
            "initial_k_detected",
            "k_requested",
            "k_actual",
            "cv_coherence",
            "umass_coherence",
            "cv_norm",
            "umass_norm",
            "combined_score",
            "n_noise",
            "noise_ratio",
            "error"
        ]

        print("\nMetrics:")
        for col in metric_cols:
            if col in row.index:
                print(f"{col}: {row[col]}")

    def _print_topic_details(row):
        run_key = row["run_key"]

        if pd.isna(run_key) or run_key not in run_registry:
            print("\nNo topic details stored for this run.")
            return

        topic_summary = run_registry[run_key]["topic_summary"]

        print("\nNumber of samples in each topic:")
        if topic_summary.empty:
            print("No non-noise topics found.")
        else:
            for _, r in topic_summary.iterrows():
                print(f"Topic {int(r['topic_id'])}: {int(r['n_samples'])} samples")

        print("\nKeywords in each topic:")
        if topic_summary.empty:
            print("No non-noise topics found.")
        else:
            for _, r in topic_summary.iterrows():
                print(f"Topic {int(r['topic_id'])}: {r['keywords']}")

    # keep only successful rows with real scores
    valid_mask = (
        all_results["error"].isna() &
        all_results["cv_coherence"].notna() &
        all_results["umass_coherence"].notna()
    )

    valid_results = all_results.loc[valid_mask].copy()

    if valid_results.empty:
        print("No valid rows found with both CV and UMass coherence scores.")
        return None

    # normalize on valid rows only
    valid_results["cv_norm"] = _minmax_normalize(valid_results["cv_coherence"])
    valid_results["umass_norm"] = _minmax_normalize(valid_results["umass_coherence"])

    # equal weights
    valid_results["combined_score"] = (
        0.5 * valid_results["cv_norm"] +
        0.5 * valid_results["umass_norm"]
    )

    # put normalized columns back into all_results too
    all_results["cv_norm"] = np.nan
    all_results["umass_norm"] = np.nan
    all_results["combined_score"] = np.nan

    all_results.loc[valid_results.index, "cv_norm"] = valid_results["cv_norm"]
    all_results.loc[valid_results.index, "umass_norm"] = valid_results["umass_norm"]
    all_results.loc[valid_results.index, "combined_score"] = valid_results["combined_score"]

    # best overall row
    best_idx = valid_results["combined_score"].idxmax()
    best_row = all_results.loc[best_idx].copy()

    print("\n" + "=" * 100)
    print("BEST RESULT BASED ON 0.5 * NORMALIZED CV + 0.5 * NORMALIZED U_MASS")
    print("=" * 100)

    _print_metrics(best_row)
    _print_topic_details(best_row)

    corresponding_initial_row = None

    # if best result is reduced, print the matching initial result
    if best_row["topic_stage"] == "reduced":
        initial_match = all_results[
            (all_results["config_id"] == best_row["config_id"]) &
            (all_results["topic_stage"] == "initial_detected") &
            (all_results["error"].isna())
        ]

        if not initial_match.empty:
            corresponding_initial_row = initial_match.iloc[0].copy()

            print("\n" + "=" * 100)
            print("CORRESPONDING INITIAL RESULT BEFORE REDUCTION")
            print("=" * 100)

            _print_metrics(corresponding_initial_row)
            _print_topic_details(corresponding_initial_row)
        else:
            print("\nBest row is reduced, but no matching initial result was found.")

    return {
        "all_results_with_normalized_scores": all_results,
        "best_result": best_row,
        "corresponding_initial_result": corresponding_initial_row
    }







def plot_k_vs_cv_umass_for_best_config(best_output, use_actual_k=True):
    import pandas as pd
    import matplotlib.pyplot as plt

    all_results = best_output["all_results_with_normalized_scores"].copy()
    best_row = best_output["best_result"]

    if best_row is None:
        print("No best result found.")
        return

    # Keep only rows from the same config as the selected best result
    df_plot = all_results[
        (all_results["config_id"] == best_row["config_id"]) &
        (all_results["error"].isna()) &
        (all_results["cv_coherence"].notna()) &
        (all_results["umass_coherence"].notna())
    ].copy()

    if df_plot.empty:
        print("No valid rows found for the selected best config.")
        return

    # x-axis choice
    x_col = "k_actual" if use_actual_k else "k_requested"

    df_plot = df_plot[[
        "topic_stage",
        "initial_k_detected",
        "k_requested",
        "k_actual",
        "cv_coherence",
        "umass_coherence"
    ]].copy()

    df_plot = df_plot[df_plot[x_col].notna()].copy()

    # Sort by k
    df_plot = df_plot.sort_values(by=[x_col, "topic_stage"]).reset_index(drop=True)

    # If duplicate x exists, keep first occurrence
    df_plot_unique = df_plot.drop_duplicates(subset=[x_col], keep="first").copy()

    # Best selected point
    best_x = best_row[x_col]
    best_cv = best_row["cv_coherence"]
    best_umass = best_row["umass_coherence"]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # CV on left axis
    line1 = ax1.plot(
        df_plot_unique[x_col],
        df_plot_unique["cv_coherence"],
        marker="o",
        color="orange",
        label="CV coherence"
    )

    # UMass on right axis
    line2 = ax2.plot(
        df_plot_unique[x_col],
        df_plot_unique["umass_coherence"],
        marker="o",
        color="blue",
        label="UMass coherence"
    )

    # Mark selected best result
    best_cv_point = ax1.scatter(
        [best_x], [best_cv],
        color="red",
        s=140,
        marker="*",
        zorder=5,
        label="Selected best CV"
    )

    best_umass_point = ax2.scatter(
        [best_x], [best_umass],
        color="green",
        s=140,
        marker="*",
        zorder=5,
        label="Selected best UMass"
    )

    # Annotate best points
    ax1.annotate(
        f"Best CV\nk={int(best_x)}",
        (best_x, best_cv),
        textcoords="offset points",
        xytext=(8, 8)
    )

    ax2.annotate(
        f"Best UMass\nk={int(best_x)}",
        (best_x, best_umass),
        textcoords="offset points",
        xytext=(8, -18)
    )

    ax1.set_xlabel("Number of topics (k)")
    ax1.set_ylabel("CV coherence", color="orange")
    ax2.set_ylabel("UMass coherence", color="blue")
    ax1.set_title("Relationship between k and CV / UMass for selected best config")

    ax1.grid(True)

    # Combined legend
    handles = line1 + line2 + [best_cv_point, best_umass_point]
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc="best")

    plt.show()

    return df_plot_unique





def generate_topic_descriptions_for_best_and_initial(results, best_output, max_keywords=6):
    import pandas as pd

    def _build_for_row(row):
        if row is None:
            return None

        run_key = row["run_key"]
        if pd.isna(run_key) or run_key not in results["run_registry"]:
            return None

        topic_summary = results["run_registry"][run_key]["topic_summary"].copy()

        def _format_keywords(keywords, n=max_keywords):
            if not isinstance(keywords, list):
                return []
            return [str(k).replace("_", " ").strip() for k in keywords[:n] if str(k).strip()]

        def _join_words(words):
            if len(words) == 0:
                return ""
            if len(words) == 1:
                return words[0]
            if len(words) == 2:
                return f"{words[0]} and {words[1]}"
            return ", ".join(words[:-1]) + f", and {words[-1]}"

        def _describe_topic(row_):
            words = _format_keywords(row_["keywords"])
            n_samples = row_.get("n_samples", None)

            if len(words) == 0:
                return "No clear keywords were extracted for this topic."

            s1 = f"This topic mainly focuses on {_join_words(words[:3])}."
            s2 = f"It also covers related ideas such as {_join_words(words)}."

            if pd.notna(n_samples):
                s3 = f"It contains {int(n_samples)} samples."
            else:
                s3 = "It groups together semantically similar samples."

            return f"{s1} {s2} {s3}"

        topic_summary["topic_description"] = topic_summary.apply(_describe_topic, axis=1)
        return topic_summary

    best_desc = _build_for_row(best_output["best_result"])
    init_desc = _build_for_row(best_output.get("corresponding_initial_result"))

    return {
        "best_topic_descriptions": best_desc,
        "initial_topic_descriptions": init_desc
    }





def assign_best_topics_to_cleaned_df_v2(
    cleaned_df,
    results,
    target_column,
    target_col_,
    summary,
    best_output=None
):
    import pandas as pd

    df_out = cleaned_df.copy()

    # ---------------------------------------------------------------
    # get best result if not passed
    # ---------------------------------------------------------------
    if best_output is None:
        best_output = select_and_print_best_config(results)

    if best_output is None or best_output.get("best_result") is None:
        raise ValueError(
            "Could not find a valid best result from select_and_print_best_config(results)."
        )

    best_row = best_output["best_result"]
    corresponding_initial_row = best_output.get("corresponding_initial_result", None)
    run_registry = results["run_registry"]

    # ---------------------------------------------------------------
    # decide which dataset column was used by the best config
    # ---------------------------------------------------------------
    slight_col = f"{target_column}_slightly_preprocessed"
    full_col = f"{target_column}_fully_preprocessed"

    dataset_to_col = {
        "original": target_column,
        "slightly_preprocessed": slight_col,
        "fully_preprocessed": full_col
    }

    best_dataset = best_row["dataset"]
    if best_dataset not in dataset_to_col:
        raise ValueError(f"Unknown dataset value in best_result: {best_dataset}")

    selected_text_col = dataset_to_col[best_dataset]

    if selected_text_col not in df_out.columns:
        raise ValueError(
            f"Selected text column '{selected_text_col}' is missing from cleaned_df."
        )

    # ---------------------------------------------------------------
    # output column names
    # ---------------------------------------------------------------
    initial_topic_col = f"{target_col_}_{summary}_initial_k_detected_topic"
    reduced_topic_col = f"{target_col_}_{summary}_detected_k_reduced"

    # remove old versions if they already exist
    if initial_topic_col in df_out.columns:
        df_out = df_out.drop(columns=[initial_topic_col])

    if reduced_topic_col in df_out.columns:
        df_out = df_out.drop(columns=[reduced_topic_col])

    # ---------------------------------------------------------------
    # same filtering logic as training:
    # keep only non-empty docs in the selected dataset column
    # ---------------------------------------------------------------
    selected_texts = df_out[selected_text_col].fillna("").astype(str)
    valid_mask = selected_texts.str.strip() != ""
    valid_index = df_out.index[valid_mask]
    valid_docs = selected_texts.loc[valid_mask].tolist()

    # ---------------------------------------------------------------
    # helper to get topics from a stored model
    # prefers model.topics_ because it preserves training order
    # ---------------------------------------------------------------
    def _get_topics_for_docs(model, docs):
        if hasattr(model, "topics_") and model.topics_ is not None:
            stored_topics = list(model.topics_)
            if len(stored_topics) == len(docs):
                return stored_topics

        topics, _ = model.transform(docs)
        return list(topics)

    # ---------------------------------------------------------------
    # case 1: winner is initial_detected
    # create ONLY the initial topic column
    # ---------------------------------------------------------------
    if best_row["topic_stage"] == "initial_detected":
        best_run_key = best_row["run_key"]

        if pd.isna(best_run_key) or best_run_key not in run_registry:
            raise ValueError(
                f"Best initial run_key '{best_run_key}' is missing from results['run_registry']."
            )

        best_model = run_registry[best_run_key]["model"]
        initial_topics = _get_topics_for_docs(best_model, valid_docs)

        df_out[initial_topic_col] = pd.Series(pd.NA, index=df_out.index, dtype="Int64")
        df_out.loc[valid_index, initial_topic_col] = pd.Series(
            initial_topics, index=valid_index, dtype="Int64"
        )

        info = {
            "best_stage": "initial_detected",
            "selected_dataset": best_dataset,
            "selected_text_column": selected_text_col,
            "initial_topic_column": initial_topic_col,
            "reduced_topic_column": None,
            "best_result": best_row,
            "corresponding_initial_result": None
        }

        return df_out, info

    # ---------------------------------------------------------------
    # case 2: winner is reduced
    # create BOTH reduced topic column and corresponding initial topic column
    # ---------------------------------------------------------------
    elif best_row["topic_stage"] == "reduced":
        best_run_key = best_row["run_key"]

        if pd.isna(best_run_key) or best_run_key not in run_registry:
            raise ValueError(
                f"Best reduced run_key '{best_run_key}' is missing from results['run_registry']."
            )

        if corresponding_initial_row is None:
            raise ValueError(
                "Best result is reduced, but corresponding_initial_result is missing."
            )

        initial_run_key = corresponding_initial_row["run_key"]
        if pd.isna(initial_run_key) or initial_run_key not in run_registry:
            raise ValueError(
                f"Corresponding initial run_key '{initial_run_key}' is missing from results['run_registry']."
            )

        reduced_model = run_registry[best_run_key]["model"]
        initial_model = run_registry[initial_run_key]["model"]

        reduced_topics = _get_topics_for_docs(reduced_model, valid_docs)
        initial_topics = _get_topics_for_docs(initial_model, valid_docs)

        df_out[reduced_topic_col] = pd.Series(pd.NA, index=df_out.index, dtype="Int64")
        df_out[initial_topic_col] = pd.Series(pd.NA, index=df_out.index, dtype="Int64")

        df_out.loc[valid_index, reduced_topic_col] = pd.Series(
            reduced_topics, index=valid_index, dtype="Int64"
        )
        df_out.loc[valid_index, initial_topic_col] = pd.Series(
            initial_topics, index=valid_index, dtype="Int64"
        )

        info = {
            "best_stage": "reduced",
            "selected_dataset": best_dataset,
            "selected_text_column": selected_text_col,
            "initial_topic_column": initial_topic_col,
            "reduced_topic_column": reduced_topic_col,
            "best_result": best_row,
            "corresponding_initial_result": corresponding_initial_row
        }

        return df_out, info

    else:
        raise ValueError(
            f"best_result['topic_stage'] must be 'initial_detected' or 'reduced', got: {best_row['topic_stage']}"
        )
    








def keep_selected_columns(df, title, summary_level):
    cols_to_keep = ["repo_full_name"]

    initial_col = f"{title}_{summary_level}_initial_k_detected_topic"
    reduced_col = f"{title}_{summary_level}_detected_k_reduced"

    if initial_col in df.columns:
        cols_to_keep.append(initial_col)

    if reduced_col in df.columns:
        cols_to_keep.append(reduced_col)

    return df[cols_to_keep].copy()











def is_nonempty_cell(x):
    # missing values
    if x is None:
        return False
    if isinstance(x, float) and pd.isna(x):
        return False

    # actual Python list/tuple/set/array
    if isinstance(x, (list, tuple, set, np.ndarray)):
        return len(x) > 0

    # strings, including stringified lists like "[]" or "['a']"
    if isinstance(x, str):
        x = x.strip()
        if x == "" or x == "[]":
            return False
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, (list, tuple, set, np.ndarray)):
                return len(parsed) > 0
        except Exception:
            pass
        return True

    # anything else counts as non-empty
    return True






def keep_selected_columns(df, title, summary_level):
    cols_to_keep = ["repo_full_name"]

    initial_col = f"{title}_{summary_level}_initial_k_detected_topic"
    reduced_col = f"{title}_{summary_level}_detected_k_reduced"

    if initial_col in df.columns:
        cols_to_keep.append(initial_col)

    if reduced_col in df.columns:
        cols_to_keep.append(reduced_col)

    return df[cols_to_keep].copy()