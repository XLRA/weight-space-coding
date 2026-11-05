"""Paired per-problem differences between two evaluations' edited models: python3 paired.py <B> <A> [...]
(B minus A; results/<name>). Code is split by source; the SE is over problems."""
import collections, json, math, sys
R = "/opt/wsp/eval/results"
FILES = {"code": "gen_edited.jsonl", "mmlupro": "gen_mmlupro_edited.jsonl", "ifeval": "gen_ifeval_edited.jsonl",
         "lcb": "gen_lcb_edited.jsonl", "math500": "gen_math500_edited.jsonl"}
def load(name, suite):
    d = collections.defaultdict(list)
    for l in open(f"{R}/{name}/{FILES[suite]}"):
        r = json.loads(l); d[r["id"]].append(r)
    return d
def stats(B, A, ids):
    acc = lambda rs: sum(r["status"] == "pass" for r in rs) / len(rs)
    d = [acc(B[i]) - acc(A[i]) for i in ids]
    m = sum(d) / len(d); se = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d))
    a = 100 * sum(acc(A[i]) for i in ids) / len(ids); b = 100 * sum(acc(B[i]) for i in ids) / len(ids)
    cnt = lambda D, k: sum(r["status"] == k for i in ids for r in D[i])
    return a, b, 100 * m, 100 * se, len(ids), {k: (cnt(A, k), cnt(B, k)) for k in ("cut_at_max_tokens", "no_answer", "syntax_error", "wrong_interface")}
b = sys.argv[1]
for a in sys.argv[2:]:
    print(f"\n{b} minus {a}")
    print(f"{'suite':22s} {a[:14]:>14s} {b[:14]:>14s}  paired (SE)      n   cut A->B   no-answer A->B  syntax A->B")
    for suite in FILES:
        A, B = load(a, suite), load(b, suite)
        groups = {suite: sorted(set(A) & set(B))}
        if suite == "code":
            src = {i: A[i][0]["source"] for i in A}
            for s in sorted(set(src.values())):
                groups[s] = sorted(i for i in groups["code"] if src[i] == s)
        for g, ids in groups.items():
            x, y, m, se, n, c = stats(B, A, ids)
            print(f"{g:22s} {x:13.1f}% {y:13.1f}%  {m:+6.2f} ({se:4.2f})  {n:4d}  {c['cut_at_max_tokens'][0]:4d}->{c['cut_at_max_tokens'][1]:<4d} {c['no_answer'][0]:4d}->{c['no_answer'][1]:<4d}  {c['syntax_error'][0]:4d}->{c['syntax_error'][1]:<4d}")
