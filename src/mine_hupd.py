"""Stream HUPD year archives from HuggingFace, keep only A61K9 (DDS) patents.

A61K9 = "Medicinal preparations characterised by special physical form" — the
CPC class that covers every Galaxy Bio platform: films/ODF (Bio-FX), liposomes
and microspheres (Bio-NN), patches and creams (Bio-D3).

Writes:
  /corpus/hupd_a61k9/<year>.jsonl   full records (volume, for pre-training)
  ./out/meta.jsonl                  one metadata row per patent
  ./out/excerpts.jsonl.gz           formulation-relevant description slices
  ./out/stats.json                  per-year counts
"""
import gzip, io, json, os, re, sys, tarfile, urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed

YEARS = [int(y) for y in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(2010, 2019))
BASE = "https://huggingface.co/datasets/HUPD/hupd/resolve/main/data/{y}.tar.gz"
VOL = "/corpus/hupd_a61k9"
os.makedirs(VOL, exist_ok=True)
os.makedirs("out", exist_ok=True)

# HUPD stores CPC/IPCR unpunctuated: 'A61B603' == A61B 6/03, so A61K 9/127 -> 'A61K9127'.
# A61K subclass has no group above 9/xx that starts with '9', so the prefix is unambiguous.
def is_a61k9(label):
    return str(label).replace(" ", "").upper().startswith("A61K9")

# cues that a passage carries a quantitative formulation
EXC = re.compile(
    r"\b(polyisobutylene|acrylate|adhesive|pullulan|alginate|povidone|PVP|HPMC|HPC|"
    r"hydroxypropyl|hydroxyethyl|carbomer|Eudragit|ethylcellulose|PLGA|PLA\b|poloxamer|"
    r"polysorbate|Tween|Span|sorbitan|lecithin|phosphatidyl|DSPC|DPPC|DOTAP|cholesterol|"
    r"PEG|polyethylene glycol|glycerin|glycerol|propylene glycol|triacetin|oleic acid|"
    r"oleyl|isopropyl myristate|Transcutol|menthol|limonene|azone|mannitol|sucralose|"
    r"aspartame|xanthan|gelatin|maltodextrin|titanium dioxide|magnesium stearate|"
    r"microcrystalline cellulose|croscarmellose|crospovidone|lactose|ammonium sulfate|"
    r"sucrose octasulfate|trehalose|histidine|polyvinyl alcohol)\b", re.I)
AMT = re.compile(r"\d+(?:\.\d+)?\s*(?:%|wt\.?%|w/w|w/v|mg\b|g\b|mL\b|parts\b|µm|nm\b)", re.I)
SECT = re.compile(r"(?i)\b(example\s+\d+|TABLE\s+\d+|formulation\s+[A-Z0-9]|comparative example)\b")

PERF = re.compile(
    r"(?i)\b(cumulative release|dissolution time|disintegration|release rate|flux|permeation|"
    r"encapsulation efficiency|loading efficiency|particle size|zeta potential|Cmax|Tmax|AUC|"
    r"bioavailability|adhesion|tensile|viscosity|assay|impurit|stability)\b")


def excerpt(desc, cap=9000):
    """Windows of description that look like formulation tables / examples."""
    if not desc:
        return ""
    keep, n = [], len(desc)
    for m in SECT.finditer(desc):
        a, b = max(0, m.start() - 200), min(n, m.start() + 2200)
        keep.append((a, b))
    for m in AMT.finditer(desc):
        w = desc[max(0, m.start() - 160): m.start() + 160]
        if EXC.search(w):
            keep.append((max(0, m.start() - 400), min(n, m.start() + 400)))
    if not keep:
        return ""
    keep.sort()
    merged = [list(keep[0])]
    for a, b in keep[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out, total = [], 0
    for a, b in merged:
        s = desc[a:b]
        if total + len(s) > cap:
            s = s[: max(0, cap - total)]
        out.append(s); total += len(s)
        if total >= cap:
            break
    return "\n…\n".join(out)


def do_year(year):
    url = BASE.format(y=year)
    kept = seen = with_ex = 0
    fv = open(f"{VOL}/{year}.jsonl", "w")
    fm = open(f"out/meta_{year}.jsonl", "w")
    fe = gzip.open(f"out/excerpts_{year}.jsonl.gz", "wt")
    req = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tf:
            for mem in tf:
                if not mem.isfile() or not mem.name.endswith(".json"):
                    continue
                seen += 1
                try:
                    rec = json.loads(tf.extractfile(mem).read().decode("utf-8", "ignore"))
                except Exception:
                    continue
                labs = []
                for k in ("main_cpc_label", "main_ipcr_label"):
                    if rec.get(k):
                        labs.append(str(rec[k]))
                for k in ("cpc_labels", "ipcr_labels"):
                    v = rec.get(k) or []
                    labs += [str(x) for x in (v if isinstance(v, list) else [v])]
                if not any(is_a61k9(l) for l in labs):
                    continue
                kept += 1
                desc = rec.get("full_description") or ""
                full = {"id": rec.get("application_number") or mem.name,
                        "pub": rec.get("publication_number"),
                        "patent_number": rec.get("patent_number"),
                        "title": rec.get("title"), "date": rec.get("filing_date"),
                        "decision": rec.get("decision"),
                        "inventors": rec.get("inventor_list"),
                        "cpc": rec.get("main_cpc_label"), "cpcs": rec.get("cpc_labels"),
                        "abstract": rec.get("abstract"), "claims": rec.get("claims"),
                        "summary": rec.get("summary"), "description": desc}
                fv.write(json.dumps(full, ensure_ascii=False) + "\n")
                ex = excerpt(desc)
                fm.write(json.dumps({"id": full["id"], "pub": full["pub"],
                                     "patent_number": full["patent_number"],
                                     "title": full["title"], "date": full["date"],
                                     "decision": full["decision"], "cpc": full["cpc"],
                                     "cpcs": full["cpcs"], "n_desc": len(desc),
                                     "n_excerpt": len(ex),
                                     "has_perf": bool(PERF.search(desc))},
                                    ensure_ascii=False) + "\n")
                if ex:
                    with_ex += 1
                    fe.write(json.dumps({"id": full["id"], "pub": full["pub"],
                                         "title": full["title"], "cpc": full["cpc"],
                                         "excerpt": ex}, ensure_ascii=False) + "\n")
    fv.close(); fm.close(); fe.close()
    return {"year": year, "scanned": seen, "a61k9": kept, "with_excerpt": with_ex}


if __name__ == "__main__":
    stats = []
    with ProcessPoolExecutor(max_workers=min(8, len(YEARS))) as ex:
        futs = {ex.submit(do_year, y): y for y in YEARS}
        for f in as_completed(futs):
            y = futs[f]
            try:
                s = f.result(); stats.append(s)
                print(f"[{s['year']}] scanned={s['scanned']:7d} A61K9={s['a61k9']:6d} "
                      f"excerpt={s['with_excerpt']:6d}", flush=True)
            except Exception as e:
                print(f"[{y}] FAILED {type(e).__name__}: {e}", flush=True)
                stats.append({"year": y, "error": f"{type(e).__name__}: {e}"})
    json.dump(sorted(stats, key=lambda d: d["year"]), open("out/stats.json", "w"), indent=1)
    tot = sum(s.get("a61k9", 0) for s in stats)
    print(f"TOTAL A61K9 = {tot}")
