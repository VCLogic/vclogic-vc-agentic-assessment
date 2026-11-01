"""Reproduce every result in the 30 October proposal ("VCLogic 30 Oct NN ee.docx") from data/ alone.

    uv run python proposal-evaluation/evaluate.py            # full run, 2,000 bootstrap resamples
    uv run python proposal-evaluation/evaluate.py --quick    # 200 resamples, for a fast check

Writes results/results.md (every table, in the order of the proposal, and a check of each number
against the draft), results/tables/*.csv and results/figures/*.png. No model calls.

Intervals: episodes are resampled with replacement (all decisions on a pitch move together), seed 0.
Tables 2, A5, A6, A9 and A10 use 2,000 resamples (numpy default_rng(0)). Table A7 repeats the
pre-specified tests, which used 1,000 resamples (Python random.seed(0)).
"""
import argparse
import collections
import csv
import itertools
import json
import random
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "results"
SIGN = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
WEIGHT = {"primary": 1.0, "secondary": 0.5}
QUOTA = 0.3
MIN_OWN = 8

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true", help="200 resamples instead of 2,000 (intervals approximate)")
ARGS = ap.parse_args()
NBOOT = 200 if ARGS.quick else 2000
NBOOT_A7 = 100 if ARGS.quick else 1000


# ---------------------------------------------------------------- data
def jsonl(name):
    return [json.loads(line) for line in open(DATA / name)]


rows = jsonl("decisions.jsonl")
cat = json.load(open(DATA / "taxonomy.json"))
N = len(rows)
Y = np.array([r["actual"] == "In" for r in rows], bool)
VC = [r["investor"] for r in rows]
EP = [r["episode"] for r in rows]
byvc, byep = collections.defaultdict(list), collections.defaultdict(list)
for i, r in enumerate(rows):
    byvc[r["investor"]].append(i)
    byep[r["episode"]].append(i)
eps = sorted(byep)
A = [i for i in range(N) if rows[i]["reference_mentions"] is not None]


def score(item):
    c = item.get("confidence")
    return SIGN.get(item.get("direction"), 0.0) * WEIGHT.get(item.get("salience"), 0.5) * (1.0 if c is None else float(c))


def tally(items):
    return sum(score(x) for x in items)


def top_cutoff(values, quota):
    ranked = sorted(values, reverse=True)
    return ranked[max(1, round(quota * len(ranked))) - 1]


def avg(x):
    x = list(x)
    return sum(x) / max(1, len(x))


def top_share_decisions(S, quota=QUOTA):
    """In if the score is in the investor's top `quota` of their other pitches (all pitches if < 8 others)."""
    d = np.zeros(N, bool)
    for i in range(N):
        own = [S[j] for j in byvc[VC[i]] if j != i]
        d[i] = S[i] >= top_cutoff(own if len(own) >= MIN_OWN else [S[j] for j in range(N) if j != i], quota)
    return d


S = np.array([tally(r["investigator_rationales"]) for r in rows])  # weighted rationale score
SL = np.array([tally(r["final_rationales"]) for r in rows])  # score from the decision agent's final rationales
LK = np.array([r["agent_likelihood"] for r in rows])  # decision agent's likelihood
D = top_share_decisions(S)  # weighted-score decision (reported)
V = np.array([r["agent_call"] == "In" for r in rows])  # decision agent's call
labels = lambda items: {x["taxonomy_label"] for x in items}
F = [labels(r["final_rationales"]) for r in rows]
G = [labels(r["investigator_rationales"]) for r in rows]
ref = lambda i: {m["label"] for m in rows[i]["reference_mentions"]}
f1 = lambda X, Z: 2 * len(X & Z) / max(1, len(X) + len(Z))


# ---------------------------------------------------------------- measures
def bal(I, d):
    y, dd = Y[I], d[I]
    return ((dd & y).sum() / y.sum() + ((~dd) & ~y).sum() / (~y).sum()) / 2


def auc(groups, s):
    ok = tot = 0
    for g in groups:
        a = s[[i for i in g if Y[i]]]
        b = s[[i for i in g if not Y[i]]]
        if len(a) and len(b):
            ok += (a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()
            tot += len(a) * len(b)
    return ok / tot if tot else float("nan")


def patk(groups, k, s):
    """Share of Ins among each investor's k highest-scoring pitches (ties shared), investors with >= 11 pitches."""
    P = []
    for g in groups:
        if len(g) < 11 or not Y[g].sum():
            continue
        srt = sorted(s[g])
        thr = srt[-k]
        above = [i for i in g if s[i] > thr]
        tie = [i for i in g if s[i] == thr]
        P.append((Y[above].sum() + Y[tie].mean() * (k - len(above))) / k)
    return avg(P)


def rationale_measures(I, sets_of, prefix):
    """Recall/precision/F1 at three levels, decision-linked and investor-fit recall, on decisions I with reference."""
    out = {}
    It = [i for i in I if rows[i]["reference_mentions"] is not None]
    for lvl, name in enumerate(["dimension", "rationale", "rationale + direction"]):
        tp = nr = npd = 0
        for i in It:
            items, ms = sets_of(i), rows[i]["reference_mentions"]
            P = [{cat.get(x["taxonomy_label"]) for x in items}, labels(items),
                 {(x["taxonomy_label"], x["direction"]) for x in items}][lvl]
            R = [{m["dimension"] for m in ms}, {m["label"] for m in ms}, {(m["label"], m["direction"]) for m in ms}][lvl]
            tp += len(P & R)
            nr += len(R)
            npd += len(P)
        out[f"{prefix}{name} recall"] = tp / nr
        out[f"{prefix}{name} precision"] = tp / npd
        out[f"{prefix}{name} F1"] = 2 * tp / (nr + npd)

    def rec(pred):
        h = t = 0
        for i in It:
            R = {m["label"] for m in rows[i]["reference_mentions"] if pred(m)}
            h += len(labels(sets_of(i)) & R)
            t += len(R)
        return h / t

    out[f"{prefix}decision-linked recall"] = rec(lambda m: m["decision_link"] != "none")
    out[f"{prefix}investor-fit recall"] = rec(lambda m: m["dimension"] == "investor_fit_constraints")
    out[f"{prefix}twin rationales per pitch"] = avg(len(labels(sets_of(i))) for i in I)
    return out


def measures(sample):
    """sample: list of (cluster key, row indices). Every number in Tables 2, A5, A6 and A9."""
    I = [i for _, g in sample for i in g]
    Ia = [i for i in I if rows[i]["reference_mentions"] is not None]
    gvc = collections.defaultdict(list)
    for i in I:
        gvc[VC[i]].append(i)
    G = [g for _, g in sample]
    split = [g for g in G if 0 < Y[g].sum() < len(g)]
    multi = [m for m in ([i for i in g if rows[i]["reference_mentions"] is not None] for g in G) if len(m) >= 2]
    out = rationale_measures(I, lambda i: rows[i]["final_rationales"], "")
    out["investor rationales per pitch"] = avg(len(ref(i)) for i in Ia)
    for name, d, s in (("score", D, S), ("agent", V, LK)):
        out[f"{name} balanced accuracy, all"] = bal(I, d)
        out[f"{name} balanced accuracy, per investor (average)"] = avg(bal(g, d) for g in gvc.values() if 0 < Y[g].sum() < len(g))
        out[f"{name} balanced accuracy, split panels"] = bal([i for g in split for i in g], d)
        out[f"{name} share called In"] = d[I].mean()
        out[f"{name} Ins called In"] = d[I][Y[I]].mean()
        out[f"{name} Outs called Out"] = (~d[I])[~Y[I]].mean()
    for name, s in (("score", S), ("agent", LK), ("final-rationale score", SL)):
        out[f"{name} AUC, all"] = auc([I], s)
        out[f"{name} AUC, within investor"] = auc(gvc.values(), s)
        out[f"{name} AUC, within panel"] = auc(split, s)
        out[f"{name} top five"] = patk(gvc.values(), 5, s)
    out["random top five"] = avg(Y[g].mean() for g in gvc.values() if len(g) >= 11 and Y[g].sum())
    out["twins: overlap between twins"] = avg(f1(F[a], F[b]) for m in multi for a, b in itertools.combinations(m, 2))
    out["twins: overlap between investors"] = avg(f1(ref(a), ref(b)) for m in multi for a, b in itertools.combinations(m, 2))
    out["twins: closer to own investor"] = avg(f1(F[a], ref(a)) > f1(F[a], ref(b)) for m in multi for a, b in itertools.permutations(m, 2))
    for k in list(out):
        if k.startswith("agent ") and k.replace("agent ", "score ", 1) in out:
            out["diff " + k[6:]] = out[k] - out[k.replace("agent ", "score ", 1)]
    return out


print(f"{N} decisions ({Y.sum()} In), {len(byvc)} investors, {len(eps)} episodes, {len(A)} with reference rationales")
point = measures([(e, byep[e]) for e in eps])
rng = np.random.default_rng(0)
print(f"bootstrap: {NBOOT} resamples of episodes ...")
B = [measures([(f"{e}#{k}", byep[e]) for k, e in enumerate(rng.choice(eps, len(eps)))]) for _ in range(NBOOT)]


def ci(key):
    return tuple(np.percentile([b[key] for b in B], [2.5, 97.5]))


pct = lambda x, d=1: f"{100 * x:.{d}f}%"
pt = lambda x: f"{100 * x:+.1f}"


def pci(key, d=1):
    lo, hi = ci(key)
    return f"{pct(point[key], d)} ({100 * lo:.{d}f}-{100 * hi:.{d}f})"


def aci(key):
    lo, hi = ci(key)
    return f"{point[key]:.2f} ({lo:.2f}-{hi:.2f})"


def dci(key, auc_=False):
    lo, hi = ci(key)
    if auc_:
        return f"{point[key]:+.2f} ({lo:+.2f} to {hi:+.2f})"
    return f"{pt(point[key])} ({pt(lo)} to {pt(hi)})"


# ---------------------------------------------------------------- report helpers
OUT.mkdir(exist_ok=True)
(OUT / "tables").mkdir(exist_ok=True)
(OUT / "figures").mkdir(exist_ok=True)
md = []
checks = []  # (where, what, draft, computed)


def table(name, caption, header, body):
    md.append(f"\n### {caption}\n")
    md.append("| " + " | ".join(header) + " |")
    md.append("|" + "---|" * len(header))
    for b in body:
        md.append("| " + " | ".join(str(x) for x in b) + " |")
    with open(OUT / "tables" / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(body)


def check(where, what, draft, computed):
    checks.append((where, what, draft, computed))


def text(s):
    md.append(s)


# ---------------------------------------------------------------- Table 1: rationale identification after each agent
inv = rationale_measures(A, lambda i: rows[i]["investigator_rationales"], "")
fin = rationale_measures(A, lambda i: rows[i]["final_rationales"], "")
inv400 = avg(len(G[i]) for i in range(N))
fin400 = avg(len(F[i]) for i in range(N))
inv_pp = avg(len(ref(i)) for i in A)
text("# Results of the 30 October proposal, reproduced from data/\n")
text(f"{N} investor-pitch decisions ({Y.sum()} In, {N - Y.sum()} Out), {len(byvc)} investors, {len(eps)} episodes. "
     f"{len(A)} decisions have reference rationales ({sum(len(rows[i]['reference_mentions']) for i in A):,} mentions). "
     f"Bootstrap: {NBOOT} resamples of episodes.\n")
text("## Main text")
T1 = [("Investor dimensions recovered", pct(inv["dimension recall"], 0), pct(fin["dimension recall"], 0)),
      ("Investor rationales recovered", pct(inv["rationale recall"], 0), pct(fin["rationale recall"], 0)),
      ("Investor rationales recovered with the same direction", pct(inv["rationale + direction recall"], 0), pct(fin["rationale + direction recall"], 0)),
      ("Rationales the investor tied to the decision, recovered", pct(inv["decision-linked recall"], 0), pct(fin["decision-linked recall"], 0)),
      (f"Rationales per pitch (investors: {inv_pp:.1f})", f"{inv['twin rationales per pitch']:.1f}", f"{fin['twin rationales per pitch']:.1f}")]
table("table1_rationales_by_agent", f"Table 1. Rationale identification after each agent ({len(A)} decisions with reference rationales)",
      ["Measure", "Investigator Agent", "Decision Agent"], T1)
text(f"\nRationales per pitch over all {N} decisions: investigator {inv400:.1f}, decision agent {fin400:.1f}. "
     f"Most rationales in one pitch: investigator {max(len(g) for g in G)}, decision agent {max(len(f) for f in F)}.")
for (m, a, b), (da, db) in zip(T1, [("91%", "95%"), ("59%", "61%"), ("29%", "32%"), ("51%", "58%"), ("17.8", "18.6")]):
    check("Table 1", m + ", investigator", da, a)
    check("Table 1", m + ", decision agent", db, b)
check("Table 1", "investor rationales per pitch", "7.6", f"{inv_pp:.1f}")
check("Table 1 caption", "decisions in the rationale comparison", "400", str(len(A)))
check("A1", "most rationales per pitch ('maximum 18')", "18", str(max(len(f) for f in F)))

# ---------------------------------------------------------------- Table 2
nI, nO = Y.sum(), (~Y).sum()
T2 = [("Always Out Baseline", f"0 (0%)", f"{nO} (100%)", "50.0%", "-", "-"),
      ("Random Ranking", "-", "-", "-", "0.50", pct(point["random top five"], 0)),
      ("Decision Agent", f"{(V & Y).sum()} ({pct((V & Y).sum() / nI, 0)})", f"{(~V & ~Y).sum()} ({pct((~V & ~Y).sum() / nO, 0)})",
       pct(point["agent balanced accuracy, all"]), f"{point['agent AUC, all']:.2f}", pct(point["agent top five"], 0)),
      ("Weighted Rationale Score", f"{(D & Y).sum()} ({pct((D & Y).sum() / nI, 0)})", f"{(~D & ~Y).sum()} ({pct((~D & ~Y).sum() / nO, 0)})",
       pct(point["score balanced accuracy, all"]), f"{point['score AUC, all']:.2f}", pct(point["score top five"], 0))]
table("table2_decisions_ranking", "Table 2. Decisions and ranking",
      ["Approach", f"Investments called In (of {nI})", f"Rejections called Out (of {nO})", "Balanced accuracy", "AUC", "Backed pitches in top five"], T2)
text(f"\nDecision agent, other ranking signals: score from its final rationales AUC {point['final-rationale score AUC, all']:.2f}, "
     f"top five {pct(point['final-rationale score top five'], 0)}; balanced accuracy within split panels {pct(point['agent balanced accuracy, split panels'])}.")
check("Table 2", "decision agent balanced accuracy", "56.1%", pct(point["agent balanced accuracy, all"]))
check("Table 2", "decision agent AUC", "0.60", f"{point['agent AUC, all']:.2f}")
check("Table 2", "decision agent top five", "32%", pct(point["agent top five"], 0))
check("Table 2", "decision agent Ins / Outs", "54 / 185", f"{(V & Y).sum()} / {(~V & ~Y).sum()}")
check("Table 2", "weighted score Ins / Outs", "46 / 216", f"{(D & Y).sum()} / {(~D & ~Y).sum()}")
check("Table 2", "weighted score balanced accuracy", "57.9%", pct(point["score balanced accuracy, all"]))
check("Table 2", "weighted score AUC", "0.62", f"{point['score AUC, all']:.2f}")
check("Table 2", "weighted score top five", "40%", pct(point["score top five"], 0))
check("Table 2", "random top five", "25%", pct(point["random top five"], 0))
check("Main text", "share of decisions that are rejections", "71.5%", pct(nO / N))

# ---------------------------------------------------------------- A2: memory sizes
text("\n## Technical appendix")
mem = json.load(open(DATA / "memory_sizes.json"))
ex = sorted(v["evidence_excerpts"] for v in mem.values())
text(f"\n**A2.** Evidence excerpts per Investment Memory: {ex[0]:,} to {ex[-1]:,} (total {sum(ex):,}); "
     f"context entries total {sum(v['context_entries'] or 0 for v in mem.values()):,}.")
check("A2", "smallest / largest memory (excerpts)", "12 / 2,082", f"{ex[0]:,} / {ex[-1]:,}")

# ---------------------------------------------------------------- Table A4: funnel
fn = jsonl("funnel.jsonl")
counts = json.load(open(DATA / "funnel_counts.json"))
fb = collections.defaultdict(list)
for x in fn:
    fb[x["investor"]].append(x)
TA4, tot = [], collections.Counter()
for vc in sorted(fb, key=lambda v: -len(fb[v])):
    xs = fb[vc]
    c = [len(xs), counts[vc]["available_transcripts"], sum(x["on_panel"] for x in xs),
         sum(x["decision"] in ("In", "Out") for x in xs), sum(x["decision"] == "In" for x in xs),
         sum(x["passed_leakage_check"] for x in xs), sum(x["assessed"] for x in xs)]
    TA4.append([vc] + c)
    for k, v in enumerate(c):
        tot[k] += v
TA4.append(["Total"] + [tot[k] for k in range(7)])
table("tableA4_funnel", "Table A4. From episode records to evaluated decisions, per investor",
      ["Investor", "# Episodes", "# Episodes w/ Transcript", "On Panel", "On-Air In or Out", "Of which In", "Passed Leakage Check", "Assessed"], TA4)
for k, (name, d) in enumerate(zip(["episodes", "with transcript", "on panel", "on-air In or Out", "of which In", "passed leakage check", "assessed"],
                                  ["750", "675", "598", "482", "132", "413", "400"])):
    check("Table A4", name + " (total)", d, str(tot[k]))
resp = list(csv.DictReader(open(DATA / "no_decision_responses.csv")))
rc = collections.Counter(r["category"] for r in resp)
text(f"\nOn panel without an on-air In or Out: {tot[2] - tot[3]}; with a recorded response classification: {len(resp)}: "
     + "; ".join(f"{k} ({v})" for k, v in rc.most_common()) + ".")
check("A4", "on-panel cases without In/Out", "116", str(tot[2] - tot[3]))
check("A4", "with a response classification", "73", str(len(resp)))

# reference rationales and agreement between two codings
sc2 = {(x["investor"], x["episode"]): x["mentions"] for x in jsonl("reference_second_coding.jsonl")}
C = collections.defaultdict(collections.Counter)
for i in A:
    o = sc2.get((VC[i], EP[i]))
    if o is None:
        continue
    ms = rows[i]["reference_mentions"]
    ol = {m["label"] for m in o}
    for lvl, X, R in (("dimension", {cat.get(l) for l in ol}, {m["dimension"] for m in ms}), ("rationale", ol, ref(i)),
                      ("rationale + direction", {(m["label"], m["direction"]) for m in o}, {(m["label"], m["direction"]) for m in ms})):
        C[lvl]["tp"] += len(X & R)
        C[lvl]["fp"] += len(X - R)
        C[lvl]["fn"] += len(R - X)
agree = {k: 2 * c["tp"] / (2 * c["tp"] + c["fp"] + c["fn"]) for k, c in C.items()}
nm = sum(len(rows[i]["reference_mentions"]) for i in A)
text(f"\nReference rationales: {len(A)} decisions, {nm:,} mentions. Agreement of two independent codings "
     f"({sum((VC[i], EP[i]) in sc2 for i in A)} decisions), F1: " + ", ".join(f"{k} {v:.2f}" for k, v in agree.items()) + ".")
check("A4", "reference set: decisions / mentions", "400 / 4,777", f"{len(A)} / {nm:,}")
for k, d in zip(["dimension", "rationale", "rationale + direction"], ["0.86", "0.73", "0.61"]):
    check("A4", f"agreement F1, {k}", d, f"{agree[k]:.2f}")

# ---------------------------------------------------------------- Table A5 and Figure A2
TA5 = [(lvl.capitalize() if lvl != "rationale + direction" else "Rationale and direction",
        pci(f"{lvl} recall", 0), pci(f"{lvl} precision", 0), pci(f"{lvl} F1", 0)) for lvl in ("dimension", "rationale", "rationale + direction")]
TA5 += [("Reasons tied to the decision", pci("decision-linked recall", 0), "-", "-"),
        ("Investor-fit reasons", pci("investor-fit recall", 0), "-", "-")]
table("tableA5_rationales", f"Table A5. Rationale identification after the decision agent ({len(A)} decisions with reference, 95% intervals)",
      ["Level", "Recall", "Precision", "F1"], TA5)
for (lvl, r_, p_, f_), d in zip(TA5, [("95% (94-96)", "57% (55-59)", "71% (70-73)"), ("61% (59-62)", "25% (23-26)", "35% (34-36)"),
                                       ("32% (30-33)", "16% (15-17)", "21% (20-22)"), ("58% (55-61)", "-", "-"), ("32% (27-36)", "-", "-")]):
    for what, dv, cv in zip(("recall", "precision", "F1"), d, (r_, p_, f_)):
        if dv != "-":
            check("Table A5", f"{lvl} {what}", dv, cv)
same = n_ = 0
for i in A:
    invd = collections.defaultdict(set)
    for m in rows[i]["reference_mentions"]:
        invd[m["label"]].add(m["direction"])
    for x in rows[i]["final_rationales"]:
        if x["taxonomy_label"] in invd:
            n_ += 1
            same += x["direction"] in invd[x["taxonomy_label"]]
shared_dir = point["rationale + direction recall"] / point["rationale recall"]
text(f"\nDirection on shared rationales: of the investor's rationales the twin also raises, the twin gives the investor's direction "
     f"for {pct(shared_dir, 0)}; of the twin's rationales the investor also raised, its direction matches one the investor gave in "
     f"{pct(same / n_, 0)} ({n_} cases).")
check("A5", "direction agreement on shared rationales (investor's side)", "50%", pct(shared_dir, 0))
check("A5", "direction agreement on shared rationales (twin's side)", "50%", pct(same / n_, 0))

DIMS = [("founder_team", "Founder/team"), ("business_model_economics", "Business model"), ("investor_fit_constraints", "Investor fit"),
        ("product_solution", "Product"), ("market_opportunity", "Market"), ("competition_defensibility", "Competition"),
        ("deal_terms_valuation", "Valuation/deal terms"), ("traction_growth", "Traction"), ("timing", "Timing")]
dimrec = collections.defaultdict(collections.Counter)
for i in A:
    for l in ref(i):
        c = dimrec[cat.get(l)]
        c["n"] += 1
        c["inv"] += l in G[i]
        c["fin"] += l in F[i]
FA2 = [(name, dimrec[d]["n"], pct(dimrec[d]["inv"] / dimrec[d]["n"], 0), pct(dimrec[d]["fin"] / dimrec[d]["n"], 0)) for d, name in DIMS]
table("figureA2_recall_by_dimension", "Figure A2 data. Share of the investor's rationales identified, by dimension",
      ["Dimension", "Investor rationales", "Investigator agent", "Decision agent"], FA2)
fa2 = {name: (a, b) for name, _, a, b in FA2}
for name, d in (("Investor fit", ("13%", "32%")), ("Competition", ("61%", "70%")), ("Founder/team", ("67%", "58%")), ("Timing", ("51%", "35%"))):
    check("A5 text", f"{name} recall, investigator -> decision agent", f"{d[0]} -> {d[1]}", f"{fa2[name][0]} -> {fa2[name][1]}")

# ---------------------------------------------------------------- Table A6 and Figure A3
TA6 = [("Balanced accuracy, all decisions", pci("score balanced accuracy, all"), pci("agent balanced accuracy, all"), dci("diff balanced accuracy, all")),
       ("Balanced accuracy, average per investor", pci("score balanced accuracy, per investor (average)"), pci("agent balanced accuracy, per investor (average)"),
        dci("diff balanced accuracy, per investor (average)")),
       ("Balanced accuracy, split panels", pci("score balanced accuracy, split panels"), pci("agent balanced accuracy, split panels"),
        dci("diff balanced accuracy, split panels")),
       ("AUC, all decisions", aci("score AUC, all"), aci("agent AUC, all"), dci("diff AUC, all", True)),
       ("AUC, within investor", aci("score AUC, within investor"), aci("agent AUC, within investor"), dci("diff AUC, within investor", True)),
       ("AUC, within panel", aci("score AUC, within panel"), aci("agent AUC, within panel"), dci("diff AUC, within panel", True)),
       ("Share of pitches called In", pci("score share called In"), pci("agent share called In"), dci("diff share called In")),
       ("Investments called In", pci("score Ins called In"), pci("agent Ins called In"), dci("diff Ins called In")),
       ("Rejections called Out", pci("score Outs called Out"), pci("agent Outs called Out"), dci("diff Outs called Out"))]
table("tableA6_two_methods", f"Table A6. Decisions and ranking for the two methods ({N} decisions, 95% intervals)",
      ["Measure", "Weighted rationale score", "Decision agent", "Difference (agent - score)"], TA6)
for row, d in zip(TA6, [("57.9% (52.0-63.8)", "56.0% (50.5-61.8)", "-1.9"), ("58.3% (51.4-66.2)", "58.8% (50.0-64.3)", "+0.5"), None,
                        ("0.62 (0.55-0.68)", "0.57 (0.51-0.64)", "-0.04"), ("0.59 (0.51-0.68)", "0.56 (0.48-0.64)", "-0.03"), None,
                        ("29.0%", "38.8%", "+9.8"), ("40.4%", "47.4%", "+7.0"), ("75.5%", "64.7%", "-10.8")]):
    if d is None:
        continue
    check("Table A6", row[0] + ", score", d[0], row[1] if "(" in d[0] else row[1].split(" ")[0])
    check("Table A6", row[0] + ", agent", d[1], row[2] if "(" in d[1] else row[2].split(" ")[0])
    check("Table A6", row[0] + ", difference", d[2], row[3].split(" ")[0])
check("Table A6 / A7", "decision agent minus score, balanced accuracy interval", "-8.7 to +5.0", TA6[0][3].split("(")[1].rstrip(")"))
check("Main text", "within split panels: weighted score / decision agent", "52.3% / 56.1%",
      f"{pct(point['score balanced accuracy, split panels'])} / {pct(point['agent balanced accuracy, split panels'])}")
check("A6", "decision agent: distinct likelihood values", "42", str(len(set(LK))))

FA3 = []
for vc in sorted(byvc, key=lambda v: -len(byvc[v])):
    g = byvc[vc]
    FA3.append((vc, len(g), pct(Y[g].mean(), 0), pct(V[g].mean(), 0), pct(D[g].mean(), 0)))
table("figureA3_share_called_in", "Figure A3 data. Share of pitches called In against the investor's share of In decisions",
      ["Investor", "Pitches", "Investor's share In", "Decision agent calls In", "Weighted score calls In"], FA3)
hi_ = max(FA3, key=lambda x: float(x[3][:-1]) - float(x[2][:-1]))
lo_ = min(FA3, key=lambda x: float(x[3][:-1]) - float(x[2][:-1]))
check("A6 text", "largest over-call: agent In vs investor In", "73% vs 9%", f"{hi_[3]} vs {hi_[2]}")
check("A6 text", "largest under-call: agent In vs investor In", "12% vs 35%", f"{lo_[3]} vs {lo_[2]}")

# ---------------------------------------------------------------- Table A7: alternative decision models (pre-specified tests)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DN = sorted({v for v in cat.values() if v})
SG = {"positive": 1, "negative": -1}
ann = A


def featurize(items):
    x = [0.0] * (2 * len(DN) + 3)
    for l, d, w in items:
        j = DN.index(cat[l]) if cat.get(l) in DN else None
        if j is not None:
            x[j] += SG.get(d, 0) * w
            x[len(DN) + j] += 1
        x[2 * len(DN) + {"positive": 0, "negative": 1}.get(d, 2)] += 1
    return x


def instats(i, excl):
    own = [o for o in range(N) if VC[o] == VC[i] and EP[o] not in excl]
    allr = [o for o in range(N) if EP[o] not in excl]
    g = avg(Y[o] for o in allr)
    rate = (sum(Y[o] for o in own) + 5 * g) / (len(own) + 5)
    ms = [m for o in ann if VC[o] == VC[i] and EP[o] not in excl for m in rows[o]["reference_mentions"]]
    if not ms:
        ms = [m for o in ann if EP[o] not in excl for m in rows[o]["reference_mentions"]]
    return rate, avg(m["direction"] == "negative" for m in ms)


print("Table A7: alternative models ...")
Xtwin = np.array([featurize([(x["taxonomy_label"], x["direction"], 2 if x["salience"] == "primary" else 1)
                             for x in rows[i]["investigator_rationales"]]) for i in range(N)])
Xref = {i: featurize([(m["label"], m["direction"], 1) for m in rows[i]["reference_mentions"]]) for i in ann}
Cm, Es = np.zeros(N), np.zeros(N)
for e in eps:
    tr = [i for i in ann if EP[i] != e]
    te = byep[e]
    X = np.array([Xref[i] for i in tr])
    sc = StandardScaler().fit(X)
    Cm[te] = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(X), Y[tr].astype(int)).predict_proba(sc.transform(Xtwin[te]))[:, 1]
ST = {i: instats(i, {EP[i]}) for i in range(N)}
for e in eps:
    te = byep[e]
    tr = [i for i in range(N) if EP[i] != e]
    f = lambda i, ex: [LK[i], S[i], Cm[i], *instats(i, ex)]
    X = np.array([f(i, {EP[i], e}) for i in tr])
    sc = StandardScaler().fit(X)
    Es[te] = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(X), Y[tr].astype(int)).predict_proba(
        sc.transform(np.array([f(i, {e}) for i in te])))[:, 1]


def cut_rate(s):
    d = np.zeros(N, bool)
    for i in range(N):
        own = [s[j] for j in byvc[VC[i]] if j != i]
        d[i] = s[i] >= top_cutoff(own if len(own) >= MIN_OWN else [s[j] for j in range(N) if j != i], ST[i][0])
    return d


# pooled and per-investor calibration of score and likelihood
XL = np.column_stack([S, LK])
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def fit(Xtr, ytr):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(Xtr), ytr)
    return lambda Z: m.predict_proba(sc.transform(Z))[:, 1]


Pc = {k: np.zeros(N) for k in ("pooled", "per-investor")}
Dc = {k: np.zeros(N, bool) for k in Pc}
for e in eps:
    te = byep[e]
    tr = [i for i in range(N) if EP[i] != e]
    pool = fit(XL[tr], Y[tr].astype(int))
    g = Y[tr].mean()
    Pc["pooled"][te] = pool(XL[te])
    Dc["pooled"][te] = Pc["pooled"][te] >= g
    for i in te:
        own = [j for j in tr if VC[j] == VC[i]]
        if len(own) < MIN_OWN or Y[own].sum() < 2 or Y[own].sum() == len(own):
            Pc["per-investor"][i], Dc["per-investor"][i] = Pc["pooled"][i], Dc["pooled"][i]
            continue
        Pc["per-investor"][i] = fit(XL[own], Y[own].astype(int))(XL[[i]])[0]
        Dc["per-investor"][i] = Pc["per-investor"][i] >= Y[own].mean()

# share of Ins (quota): per investor or one for everyone, chosen on other episodes
GRID = [round(0.05 * k, 2) for k in range(1, 17)]
EPI = np.array([eps.index(e) for e in EP])


def cut_excl_episode(idx, q, s=S):
    idx = np.asarray(idx)
    out = np.zeros(len(idx), bool)
    for n, i in enumerate(idx):
        vals = s[idx[EPI[idx] != EPI[i]]]
        out[n] = s[i] >= top_cutoff(list(vals), q)
    return out


def bal_y(y, d):
    p, n = y.sum(), len(y) - y.sum()
    if p == 0 or n == 0:
        return None
    return ((d & y).sum() / p + ((~d) & ~y).sum() / n) / 2


def best_q(idx):
    sc = [(bal_y(Y[idx], cut_excl_episode(idx, q)), q) for q in GRID]
    sc = [x for x in sc if x[0] is not None]
    if not sc:
        return None
    m = max(x for x, _ in sc)
    return float(np.median([q for x, q in sc if x == m]))


qcache = {}
Dq = {"per-investor": np.zeros(N, bool), "single": np.zeros(N, bool)}
for i in range(N):
    own = [j for j in byvc[VC[i]] if EP[j] != EP[i]]
    allo = [j for j in range(N) if EP[j] != EP[i]]
    if EP[i] not in qcache:
        qcache[EP[i]] = best_q(allo)
    gq = qcache[EP[i]]
    q = best_q(own) if len(own) >= MIN_OWN else None
    q = gq if q is None else q
    pool = own if len(own) >= MIN_OWN else allo
    Dq["per-investor"][i] = S[i] >= top_cutoff([S[j] for j in pool], q)
    Dq["single"][i] = S[i] >= top_cutoff([S[j] for j in pool], gq)
Dbase = np.array([S[i] >= top_cutoff([S[j] for j in ([j for j in byvc[VC[i]] if EP[j] != EP[i]]
                                                      if len([j for j in byvc[VC[i]] if EP[j] != EP[i]]) >= MIN_OWN
                                                      else [j for j in range(N) if EP[j] != EP[i]])], QUOTA) for i in range(N)])

ALT = [("Regression on the twin's rationales", cut_rate(Cm), Cm, D),
       ("Regression combining the score, the agent's likelihood and the investor's past share of Ins", cut_rate(Es), Es, D),
       ("Pooled calibration of score and likelihood", Dc["pooled"], Pc["pooled"], D),
       ("Per-investor regression models", Dc["per-investor"], Pc["per-investor"], D),
       ("Per-investor share of Ins, chosen on other episodes", Dq["per-investor"], S, Dbase),
       ("One share of Ins for all investors, chosen on other episodes", Dq["single"], S, Dbase)]
random.seed(0)
RB = [[i for e in random.choices(eps, k=len(eps)) for i in byep[e]] for _ in range(NBOOT_A7)]


def balI(idx, d):
    return bal_y(Y[idx], d[idx])


TA7 = [("Weighted rationale score (reported)", pct(bal(list(range(N)), D)), "-", f"{auc([range(N)], S):.2f}"),
       ("Decision agent", pct(point["agent balanced accuracy, all"]), dci("diff balanced accuracy, all"), f"{point['agent AUC, all']:.2f}")]
for name, d, s, base in ALT:
    diffs = sorted(balI(ix, d) - balI(ix, base) for ix in RB)
    lo, hi = diffs[int(0.025 * NBOOT_A7) - 1], diffs[int(0.975 * NBOOT_A7) - 1]
    TA7.append((name, pct(balI(list(range(N)), d)), f"{pt(balI(list(range(N)), d) - balI(list(range(N)), base))} ({pt(lo)} to {pt(hi)})",
                f"{auc([range(N)], s):.2f}"))
table("tableA7_alternative_models", f"Table A7. Alternative decision models ({N} decisions; differences against the weighted rationale score)",
      ["Approach", "Balanced accuracy", "Difference (points)", "AUC"], TA7)
for row, d in zip(TA7[2:], [("53.9%", "-4.0 (-8.5 to +0.8)", "0.61"), ("53.0%", "-4.9 (-9.0 to -1.4)", "0.59"),
                            ("57.5%", "-0.4 (-5.0 to +4.2)", "0.59"), ("53.5%", "-4.4 (-10.4 to +0.7)", "0.54"),
                            ("54.3%", "-3.6 (-8.4 to +1.5)", "0.62"), ("58.7%", "+0.7 (-2.2 to +3.7)", "0.62")]):
    check("Table A7", row[0], " | ".join(d), " | ".join(row[1:]))

# ---------------------------------------------------------------- Table A8: per investor
TA8 = []
for vc in sorted(byvc, key=lambda v: -len(byvc[v])):
    g = byvc[vc]
    both = 0 < Y[g].sum() < len(g)
    TA8.append((vc, len(g), int(Y[g].sum()), pct(bal(g, V), 0) if both else "-", pct(bal(g, D), 0) if both else "-",
                f"{auc([g], LK):.2f}" if both else "-", f"{auc([g], S):.2f}" if both else "-"))
pv = lambda d: avg(bal(g, d) for g in byvc.values() if 0 < Y[g].sum() < len(g))
pa = lambda s: avg(auc([g], s) for g in byvc.values() if 0 < Y[g].sum() < len(g))
TA8.append(("Average across investors", "", "", pct(pv(V)), pct(pv(D)), f"{pa(LK):.2f}", f"{pa(S):.2f}"))
table("tableA8_per_investor", "Table A8. Decisions and ranking per investor",
      ["Investor", "Pitches", "In", "Balanced accuracy: decision agent", "Balanced accuracy: weighted score", "AUC: decision agent", "AUC: weighted score"], TA8)
check("Table A8", "average: agent / score balanced accuracy", "58.8% / 58.3%", f"{pct(pv(V))} / {pct(pv(D))}")
check("Table A8", "average: agent / score AUC", "0.55 / 0.60", f"{pa(LK):.2f} / {pa(S):.2f}")

# ---------------------------------------------------------------- Table A9 and Figure A4
TA9 = [("AUC, all decisions", aci("score AUC, all"), aci("final-rationale score AUC, all"), aci("agent AUC, all")),
       ("AUC, within investor", aci("score AUC, within investor"), aci("final-rationale score AUC, within investor"), aci("agent AUC, within investor")),
       (f"Backed pitches in the top five (random: {pct(point['random top five'], 0)})", pct(point["score top five"], 0),
        pct(point["final-rationale score top five"], 0), pct(point["agent top five"], 0))]
table("tableA9_ranking", f"Table A9. Ranking across all decisions and within investors ({N} decisions, 95% intervals)",
      ["Measure", "Weighted rationale score", "Score from the decision agent's rationales", "Decision agent's likelihood"], TA9)
for row, d in zip(TA9, [("0.62 (0.55-0.68)", "0.60 (0.54-0.66)", "0.57 (0.51-0.64)"), ("0.59 (0.51-0.68)", "0.59 (0.50-0.67)", "0.56 (0.48-0.64)"),
                        ("40%", "32%", "23%")]):
    for what, dv, cv in zip(("score", "final-rationale score", "likelihood"), d, row[1:]):
        check("Table A9", f"{row[0][:30]}, {what}", dv, cv)
big = [g for g in byvc.values() if len(g) > 10]
FA4 = [(k, pct(patk(big, k, S), 0), pct(patk(big, k, SL), 0), pct(patk(big, k, LK), 0), pct(avg(Y[g].mean() for g in big if Y[g].sum()), 0))
       for k in range(1, 11)]
table("figureA4_precision_at_k", f"Figure A4 data. Share of backed pitches among the k highest-ranked, {len(big)} investors with more than ten pitches",
      ["k", "Weighted rationale score", "Score from the decision agent's rationales", "Decision agent's likelihood", "Random"], FA4)
check("A7", "investors with more than ten pitches", "13", str(len(big)))

# ---------------------------------------------------------------- Table A10: single call
scd = jsonl("single_call.jsonl")
scrun = json.load(open(DATA / "single_call_run.json"))
key = {(VC[i], EP[i]): i for i in range(N)}
M = [key[(x["investor"], x["episode"])] for x in scd]
SCV = np.zeros(N, bool)
SCL = np.full(N, np.nan)
for x, i in zip(scd, M):
    SCV[i] = x["call"] == "In"
    SCL[i] = x["likelihood"]
Meps = sorted({EP[i] for i in M})
Mby = collections.defaultdict(list)
for i in M:
    Mby[EP[i]].append(i)


def a10(I):
    return {"sc bal": bal(I, SCV), "score bal": bal(I, D), "agent bal": bal(I, V),
            "sc auc": auc([I], SCL), "score auc": auc([I], S), "agent auc": auc([I], LK)}


p10 = a10(M)
rng10 = np.random.default_rng(0)
B10 = [a10([i for e in rng10.choice(Meps, len(Meps)) for i in Mby[e]]) for _ in range(NBOOT)]
q10 = lambda f: tuple(np.percentile([f(b) for b in B10], [2.5, 97.5]))
lo1, hi1 = q10(lambda b: b["score bal"] - b["sc bal"])
lo2, hi2 = q10(lambda b: b["score auc"] - b["sc auc"])
lo3, hi3 = q10(lambda b: b["agent bal"] - b["sc bal"])
lo4, hi4 = q10(lambda b: b["agent auc"] - b["sc auc"])
yM = Y[M]
TA10 = [("Balanced accuracy", pct(p10["sc bal"]), pct(p10["score bal"]), pct(p10["agent bal"])),
        ("Difference from single call (points)", "-", f"{pt(p10['score bal'] - p10['sc bal'])} ({pt(lo1)} to {pt(hi1)})",
         f"{pt(p10['agent bal'] - p10['sc bal'])} ({pt(lo3)} to {pt(hi3)})"),
        ("AUC", f"{p10['sc auc']:.2f}", f"{p10['score auc']:.2f}", f"{p10['agent auc']:.2f}"),
        ("Difference from single call", "-", f"{p10['score auc'] - p10['sc auc']:+.2f} ({lo2:+.2f} to {hi2:+.2f})",
         f"{p10['agent auc'] - p10['sc auc']:+.2f} ({lo4:+.2f} to {hi4:+.2f})"),
        ("Share of pitches called In", pct(SCV[M].mean(), 0), pct(D[M].mean(), 0), pct(V[M].mean(), 0)),
        ("Investments called In", pct(SCV[M][yM].mean(), 0), pct(D[M][yM].mean(), 0), pct(V[M][yM].mean(), 0)),
        ("Rejections called Out", pct((~SCV[M])[~yM].mean(), 0), pct((~D[M])[~yM].mean(), 0), pct((~V[M])[~yM].mean(), 0))]
table("tableA10_single_call", f"Table A10. A single model call against the prototype ({len(M)} decisions completed by both, "
      f"{len(Meps)} episodes, {yM.sum()} In; 95% intervals)", ["Measure", "Single call", "VCLogic + weighted rationale score", "VCLogic decision agent"], TA10)
text(f"\nSingle-call run: {scrun['planned']} planned, status {scrun['status']}, recorded cost USD {scrun['cost_usd']}.")
for (m, a, b, _), d in zip(TA10, [("55.0%", "59.5%"), ("-", "+4.5 (-3.8 to +13.0)"), ("0.60", "0.62"), ("-", "+0.01 (-0.08 to +0.10)"), ("91%", "31%"),
                                   ("98%", None), ("12%", None)]):
    if d[0] != "-":
        check("Table A10", m + ", single call", d[0], a)
    if d[1]:
        check("Table A10", m + ", weighted score", d[1], b)
check("A8 text", "matched decisions / episodes / Ins", "200 / 107 / 56", f"{len(M)} / {len(Meps)} / {yM.sum()}")
check("A8 text", "decision agent on the matched set: calls In", "39%", pct(V[M].mean(), 0))

# ---------------------------------------------------------------- Table A11: GPT-6 Sol
sol = json.load(open(DATA / "sol_pilots.json"))
cols = []
for p in sol:
    cs = p["cases"]
    cols.append({"n": len(cs), "luna": sum(c["luna_call"] == c["actual"] for c in cs), "sol": sum(c["sol_call"] == c["actual"] for c in cs),
                 "changed": sum(c["sol_call"] != c["luna_call"] for c in cs),
                 "fixed": sum(c["sol_call"] != c["luna_call"] and c["sol_call"] == c["actual"] for c in cs),
                 "broke": sum(c["sol_call"] != c["luna_call"] and c["luna_call"] == c["actual"] for c in cs),
                 "sol_in": sum(c["sol_call"] == "In" for c in cs), "luna_in": sum(c["luna_call"] == "In" for c in cs), "cost": p["sol_cost_usd"]})
both = {k: sum(c[k] for c in cols) for k in cols[0]}
TA11 = [("Luna correct", *(f"{c['luna']}/{c['n']}" for c in cols), f"{both['luna']}/{both['n']}"),
        ("Sol correct", *(f"{c['sol']}/{c['n']}" for c in cols), f"{both['sol']}/{both['n']}"),
        ("Calls Sol changed", *(c["changed"] for c in cols), both["changed"]),
        ("Changes that fixed an error", *(c["fixed"] for c in cols), both["fixed"]),
        ("Changes that broke a correct call", *(c["broke"] for c in cols), both["broke"]),
        ("Calls In: Luna / Sol", *(f"{c['luna_in']} / {c['sol_in']}" for c in cols), f"{both['luna_in']} / {both['sol_in']}"),
        ("Sol cost", *(f"USD {c['cost']:.2f}" for c in cols), f"About USD {both['cost'] / both['n']:.2f} per decision")]
table("tableA11_sol", "Table A11. GPT-6 Sol against GPT-5.6 Luna in the decision agent (two diagnostic pilots of 16 decisions)",
      ["Measure", "Pilot 1", "Pilot 2", "Both"], TA11)
for row, d in zip(TA11, [("8/16", "8/16", "16/32"), ("7/16", "9/16", "16/32"), ("7", "5", "12"), ("3", "3", "6"), ("4", "2", "6"), (None, None, "16 / 10"),
                         ("USD 8.94", "USD 8.27", "About USD 0.53 per decision")]):
    for what, dv, cv in zip(("pilot 1", "pilot 2", "both"), d, row[1:]):
        if dv:
            check("Table A11", f"{row[0]}, {what}", dv, str(cv))

# ---------------------------------------------------------------- telling twins apart (next steps / earlier drafts)
text(f"\n**Telling investors apart** ({sum(1 for e in eps if sum(rows[i]['reference_mentions'] is not None for i in byep[e]) >= 2)} pitches "
     f"with two or more investors with reference rationales): overlap between twins {pci('twins: overlap between twins')}, "
     f"between investors {pci('twins: overlap between investors')}; twin closer to its own investor than to a panel-mate "
     f"{pci('twins: closer to own investor')}.")

# ---------------------------------------------------------------- figures
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, LIGHT, ORANGE, INK, INK2, GRIDC, SURF = "#2a78d6", "#9cc2ee", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRIDC, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.dpi": 220, "savefig.bbox": "tight"})


def style(ax, axis="x"):
    ax.grid(axis=axis, color=GRIDC, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


fig, ax = plt.subplots(figsize=(6.4, 3.4))
y = np.arange(len(DIMS))[::-1]
h = 0.36
iv = [dimrec[d]["inv"] / dimrec[d]["n"] for d, _ in DIMS]
fv = [dimrec[d]["fin"] / dimrec[d]["n"] for d, _ in DIMS]
ax.barh(y + h / 2, iv, height=h, color=LIGHT, edgecolor=SURF, lw=1.5, label="Investigator agent")
ax.barh(y - h / 2, fv, height=h, color=BLUE, edgecolor=SURF, lw=1.5, label="Decision agent")
for yi, v in zip(y - h / 2, fv):
    ax.text(v + 0.01, yi, f"{v:.0%}", va="center", fontsize=7.5, color=INK)
ax.set_yticks(y)
ax.set_yticklabels([f"{name} ({dimrec[d]['n']})" for d, name in DIMS], color=INK)
ax.set_xlim(0, 1)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
ax.set_xlabel("Share of the investor's rationales the twin identifies (number of investor rationales)")
style(ax)
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=2)
fig.savefig(OUT / "figures" / "figureA2_recall_by_dimension.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.4, 4.0))
order = sorted(byvc, key=lambda v: Y[byvc[v]].mean())
yy = np.arange(len(order))
for k, vc in enumerate(order):
    g = byvc[vc]
    ax.plot([Y[g].mean(), V[g].mean()], [k, k], color=GRIDC, lw=2.5, zorder=1)
sz = [max(14, 2.2 * len(byvc[v])) for v in order]
ax.scatter([Y[byvc[v]].mean() for v in order], yy, s=sz, color=INK2, edgecolor=SURF, lw=1, zorder=3, label="Investor's share of In decisions")
ax.scatter([V[byvc[v]].mean() for v in order], yy, s=sz, color=ORANGE, edgecolor=SURF, lw=1, zorder=3, label="Share the decision agent calls In")
ax.set_yticks(yy)
ax.set_yticklabels([f"Investor #{k + 1}" for k in range(len(order))], fontsize=7.5, color=INK)
ax.set_xlim(0, 1)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
style(ax)
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.07), ncol=2)
fig.savefig(OUT / "figures" / "figureA3_share_called_in.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.4, 3.0))
ks = list(range(1, 11))
for s, c, lab in ((S, BLUE, "Weighted rationale score"), (LK, ORANGE, "Decision agent's likelihood")):
    ax.plot(ks, [patk(big, k, s) for k in ks], "-o", color=c, ms=5, lw=2, mec=SURF, mew=1.2, label=lab)
base = avg(Y[g].mean() for g in big if Y[g].sum())
ax.axhline(base, color=INK2, lw=1, ls=(0, (3, 2)))
ax.text(10.35, base, "random", va="center", fontsize=7.5, color=INK2)
ax.set_xlim(0.6, 11.2)
ax.set_xticks(ks)
ax.set_xlabel("k (pitches at the top of each twin's ranking)")
ax.set_ylim(0, 0.6)
ax.set_yticks([0, 0.2, 0.4, 0.6])
ax.set_yticklabels(["0", "20%", "40%", "60%"])
style(ax, "y")
ax.legend(frameon=False, fontsize=8, loc="upper right")
fig.savefig(OUT / "figures" / "figureA4_precision_at_k.png")
plt.close(fig)

# ---------------------------------------------------------------- check against the draft
norm = lambda s: str(s).replace("−", "-").replace(" ", "")
text("\n## Check against the draft\n")
text("Each number in the draft next to the value computed here. 'differs' rows need a decision before submission.\n")
text("| Where | What | Draft | Computed | |\n|---|---|---|---|---|")
nd = 0
for where, what, d, c in checks:
    ok = norm(d) == norm(c)
    nd += not ok
    text(f"| {where} | {what} | {d} | {c} | {'ok' if ok else '**differs**'} |")
text(f"\n{len(checks) - nd} of {len(checks)} numbers match; {nd} differ.")
text("\nNot reproduced here: Table A3 (sources per investor) comes from the VC Trace Collector's collection records, "
     "which are not part of this repository.")
(OUT / "results.md").write_text("\n".join(md) + "\n")
print(f"{len(checks) - nd}/{len(checks)} draft numbers match; see {OUT / 'results.md'}")
