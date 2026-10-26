"""
Teacher-stage distillation: patent prose -> structured formulation records.
===========================================================================

DDS has almost no public *labelled* formulation data — the labels exist, but
they are locked inside patent examples as prose and ASCII tables. This stage
uses a large model as a teacher to convert those passages into structured
(composition, process, performance) records, which then supervise the much
smaller DDS-Former student.

Run from the `python` tool (needs the injected `host` for host.llm):

    import distill_extract as dx
    recs = dx.run(dx.load_excerpts("hpc/<job>/out"), host, model=None, limit=500)
"""
import glob
import gzip
import json
import os
import re

SCHEMA = {
    "name": "emit_formulations",
    "description": "Emit every quantitative formulation found in the patent passage.",
    "input_schema": {
        "type": "object",
        "properties": {
            "formulations": {
                "type": "array",
                "description": ("One entry per distinct quantitative formulation actually "
                                "tabulated or stated in the passage. Ignore claim ranges, "
                                "prior-art citations, and purely qualitative lists."),
                "items": {
                    "type": "object",
                    "properties": {
                        "example_label": {"type": "string"},
                        "dosage_form": {
                            "type": "string",
                            "description": ("Normalise to one of: oral_film, oral_gum, tablet, "
                                            "capsule, transdermal_patch, cream, ointment, gel, "
                                            "lacquer, liposome_injection, microsphere_injection, "
                                            "nanoparticle_injection, solution_injection, "
                                            "suspension, implant, other")},
                        "api": {"type": "string", "description": "Active ingredient, generic name"},
                        "components": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "role": {"type": "string", "description":
                                             ("film_former, polymer, plasticiser, adhesive, "
                                              "permeation_enhancer, surfactant, solvent, "
                                              "lipid, cholesterol, pegylated_lipid, buffer, "
                                              "sweetener, flavour, colourant, filler, "
                                              "disintegrant, lubricant, preservative, api, other")},
                                    "pct_w_w": {"type": "number", "description":
                                                "Amount as %w/w if stated or derivable; else omit"},
                                    "amount_raw": {"type": "string"}},
                                "required": ["name"]}},
                        "process": {
                            "type": "array",
                            "items": {"type": "object", "properties": {
                                "param": {"type": "string"}, "value": {"type": "string"}}}},
                        "performance": {
                            "type": "array",
                            "description": "Measured outcomes with their assay conditions.",
                            "items": {"type": "object", "properties": {
                                "metric": {"type": "string", "description":
                                           ("dissolution_time, disintegration_time, "
                                            "cumulative_release, release_rate, skin_flux, "
                                            "permeation, encapsulation_efficiency, "
                                            "particle_size, zeta_potential, cmax, tmax, auc, "
                                            "adhesion, tensile_strength, viscosity, "
                                            "assay, impurity, other")},
                                "value": {"type": "number"},
                                "unit": {"type": "string"},
                                "condition": {"type": "string", "description":
                                              "medium, membrane, timepoint, temperature, site"}}}}
                    },
                    "required": ["components"]}}
        },
        "required": ["formulations"]},
}

PROMPT = ("Extract every quantitative pharmaceutical formulation from this patent passage.\n"
          "Report amounts exactly as printed in amount_raw, and additionally as pct_w_w when "
          "the passage states or implies percent-by-weight. Attach each measured outcome to "
          "the formulation it belongs to, with its assay condition. If the passage contains no "
          "quantitative formulation, return an empty list.\n\n"
          "PATENT: {title}\nCPC: {cpc}\n\nPASSAGE:\n{excerpt}")


def load_excerpts(outdir):
    rows = []
    for fp in sorted(glob.glob(os.path.join(outdir, "excerpts_*.jsonl.gz"))):
        with gzip.open(fp, "rt") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    return rows


def _requests(rows, model):
    out = []
    for r in rows:
        req = {"prompt": PROMPT.format(title=(r.get("title") or "")[:160],
                                       cpc=r.get("cpc") or "",
                                       excerpt=(r.get("excerpt") or "")[:9000]),
               "tools": [SCHEMA], "tool_choice": {"type": "tool", "name": "emit_formulations"},
               "max_tokens": 4000}
        if model:
            req["model"] = model
        out.append(req)
    return out


def run(rows, host, model=None, limit=None, batch=200, concurrency=8, out_path=None):
    """Fan out teacher extraction. Returns list of per-patent dicts."""
    rows = rows[:limit] if limit else rows
    results, usage = [], {"input_tokens": 0, "output_tokens": 0}
    fh = open(out_path, "w") if out_path else None
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        reps = host.llm(_requests(chunk, model), max_concurrency=concurrency)
        for src, rep in zip(chunk, reps):
            if not isinstance(rep, dict) or rep.get("error"):
                continue
            u = rep.get("usage") or {}
            usage["input_tokens"] += u.get("input_tokens", 0)
            usage["output_tokens"] += u.get("output_tokens", 0)
            forms = []
            for tu in (rep.get("tool_use") or []):
                if tu.get("name") == "emit_formulations":
                    forms = (tu.get("input") or {}).get("formulations", []) or []
            rec = {"id": src.get("id"), "pub": src.get("pub"), "title": src.get("title"),
                   "cpc": src.get("cpc"), "formulations": forms}
            results.append(rec)
            if fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  teacher {min(i + batch, len(rows))}/{len(rows)}  "
              f"forms={sum(len(r['formulations']) for r in results)}", flush=True)
    if fh:
        fh.close()
    return results, usage


def to_tables(records):
    """Flatten to component-level and performance-level rows."""
    comp, perf = [], []
    for rec in records:
        for k, f in enumerate(rec.get("formulations") or []):
            fid = f"{rec['id']}_{k}"
            meta = {"formulation_id": fid, "patent_id": rec["id"], "pub": rec.get("pub"),
                    "cpc": rec.get("cpc"), "dosage_form": f.get("dosage_form"),
                    "api": f.get("api"), "example_label": f.get("example_label")}
            for c in (f.get("components") or []):
                comp.append({**meta, "component": c.get("name"), "role": c.get("role"),
                             "pct_w_w": c.get("pct_w_w"), "amount_raw": c.get("amount_raw")})
            for p in (f.get("performance") or []):
                perf.append({**meta, "metric": p.get("metric"), "value": p.get("value"),
                             "unit": p.get("unit"), "condition": p.get("condition")})
    return comp, perf
