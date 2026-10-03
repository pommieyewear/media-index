"""Build the title-to-title neighbour graph the app's recommendation engine reads.

The app recommends on the device (`com.anilili.data.recs.TasteEngine`): it adds up the neighbours
of everything the viewer liked. This script is the expensive half — deciding, once a day and for
the whole catalogue, which titles neighbour which — and publishes the answer as one compact file:

    data/recs/neighbours-v1.tsv.gz

Two sources go into an edge, because each covers the other's blind spot:

  * **AniList community recommendations** — users voting "if you liked X, watch Y". The strongest
    signal there is, and it links titles metadata never would (Frieren and Mushishi share almost no
    tags). The catalogue walk already holds each title's top 10 *with* vote counts; `fetch` asks
    for up to 50 for the 4,000-odd titles that hit that cap. Measured 2026-10-03: two aliased
    `Page(id_in: [50 ids])` fields return 50 titles x 50 recommendations for ONE rate-limit unit,
    so the deep pass is ~90 requests, not 4,000.
  * **Tag similarity** — TF-IDF cosine over AniList's ranked tags and genres. It is what the long
    tail has: only 35% of titles below popularity rank 10,000 have a single community vote.

Every title also carries its franchise root (the first entry of its prequel/sequel component) and
two flags, so the device can collapse seasons and filter adult/unreleased titles without asking
AniList anything.

Stages, run after `konoha_build.py catalog` has written build/konoha/catalog.json:

    fetch   Deep recommendations for titles at the catalogue's cap. Cached per title; only titles
            older than --stale-days are re-asked, so a daily run touches a seventh of them.
    emit    Write data/recs/neighbours-v1.tsv.gz from the catalogue and the cache.

Usage:
    scripts/.venv/Scripts/python.exe scripts/recs_build.py fetch
    scripts/.venv/Scripts/python.exe scripts/recs_build.py emit --out build/konoha/data
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import pathlib
import sys
import time
from datetime import datetime, timezone

import numpy as np
import requests

from konoha_build import CATALOG_FILE, MIN_INTERVAL, WORK, AniList

DEEP_FILE = WORK / "recs-deep.json"
OUTPUT_NAME = "recs/neighbours-v1.tsv.gz"
HEADER = "#anilili-neighbours"
FORMAT_VERSION = 1

# What the catalogue query asks for (`recommendations(perPage: 10)` in konoha_build.CATALOG_QUERY).
# A title with fewer than this many has nothing more to fetch.
CATALOGUE_REC_CAP = 10
# AniList clamps a recommendations page to 25, and two pages is where the votes have thinned to
# single digits even for the most-recommended titles (Frieren's 50th has 45 votes, its 1st 1,191).
DEEP_PAGES = 2
DEEP_PER_PAGE = 25
DEEP_BATCH = 50

MAX_NEIGHBOURS = 40
MIN_WEIGHT = 0.03
# Vote counts at which a title's own best recommendation is fully believed. A title whose best has
# one vote is mostly noise — 23% of all edges are single votes.
VOTE_CONFIDENCE = 30
REC_SHARE = 0.8
TAG_SHARE_WITH_REC = 0.2
TAG_SHARE_ALONE = 0.45
TAG_NEIGHBOURS = 30
MIN_TAG_COSINE = 0.35
MIN_TAGS = 3
# Tag-only neighbours must be titles people have heard of: cosine alone happily pairs a show with an
# obscure music video that shares its three tags.
MIN_TAG_TARGET_POPULARITY = 2000
TAG_TARGET_EXCLUDED_FORMATS = {"MUSIC"}
GENRE_WEIGHT = 0.6
SPOILER_TAG_WEIGHT = 0.5
# Formats preferred as a franchise's root, so "Gintama" is recommended rather than a 2005 festival OVA.
ROOT_FORMATS = {"TV", "TV_SHORT", "ONA", "MOVIE"}
# Relations that make two entries one franchise: its seasons, and the recap films cut from them.
FRANCHISE_RELATIONS = {"PREQUEL", "SEQUEL", "SUMMARY", "COMPILATION"}
# Relations that keep a title off its franchise's neighbour lists without merging it into the
# franchise. "Attack on Titan: Junior High" is not a discovery for an Attack on Titan viewer — but
# folding every side story into its parent would weld the whole Gundam universe into one title.
ADJACENT_RELATIONS = {"SIDE_STORY", "SPIN_OFF", "PARENT", "ALTERNATIVE", "CHARACTER", "OTHER"}

FLAG_ADULT = 1
FLAG_UNRELEASED = 2

DEEP_QUERY = """
query ($ids: [Int]) {
  %s
}
""" % "\n  ".join(
    f"p{page}: Page(perPage: {DEEP_BATCH}) {{ media(id_in: $ids, type: ANIME) {{ id "
    f"recommendations(page: {page}, perPage: {DEEP_PER_PAGE}, sort: RATING_DESC) "
    f"{{ nodes {{ rating mediaRecommendation {{ id }} }} }} }} }}"
    for page in range(1, DEEP_PAGES + 1)
)


# --------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------


def load_catalog(path: pathlib.Path = CATALOG_FILE) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} is missing; run `konoha_build.py catalog` first")
    return json.loads(path.read_text(encoding="utf-8"))


def catalogue_recs(entry: dict) -> list[tuple[int, int]]:
    """(target id, votes) from the catalogue walk's own top-10, best first, positive votes only."""
    out = []
    for node in ((entry.get("recommendations") or {}).get("nodes")) or []:
        target = (node.get("mediaRecommendation") or {}).get("id")
        rating = node.get("rating") or 0
        if target and rating > 0:
            out.append((target, rating))
    return out


def load_deep(path: pathlib.Path = DEEP_FILE) -> dict[int, dict]:
    if not path.exists():
        return {}
    return {int(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def save_deep(deep: dict[int, dict], path: pathlib.Path = DEEP_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(k): v for k, v in sorted(deep.items())}
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def rec_lists(catalog: list[dict], deep: dict[int, dict]) -> dict[int, list[tuple[int, int]]]:
    """Each title's recommendations: the deep fetch where there is one, the catalogue's otherwise.

    The deep list is a superset of the catalogue's top 10 when both are fresh; when the deep copy is
    older, the catalogue's newer vote counts win for the targets both name.
    """
    out: dict[int, list[tuple[int, int]]] = {}
    for entry in catalog:
        shallow = catalogue_recs(entry)
        cached = deep.get(entry["id"])
        if not cached:
            out[entry["id"]] = shallow
            continue
        merged = {target: votes for target, votes in cached.get("r") or []}
        merged.update(dict(shallow))
        out[entry["id"]] = sorted(merged.items(), key=lambda kv: -kv[1])
    return out


# --------------------------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------------------------


def needs_deep(entry: dict) -> bool:
    return len(catalogue_recs(entry)) >= CATALOGUE_REC_CAP


def deep_rows(data: dict) -> dict[int, list[tuple[int, int]]]:
    """Fold every aliased page of one response into (target, votes) lists per title."""
    rows: dict[int, dict[int, int]] = {}
    for page in range(1, DEEP_PAGES + 1):
        for media in ((data.get(f"p{page}") or {}).get("media")) or []:
            bucket = rows.setdefault(media["id"], {})
            for node in ((media.get("recommendations") or {}).get("nodes")) or []:
                target = (node.get("mediaRecommendation") or {}).get("id")
                rating = node.get("rating") or 0
                if target and rating > 0:
                    bucket[target] = max(bucket.get(target, 0), rating)
    return {mid: sorted(b.items(), key=lambda kv: -kv[1]) for mid, b in rows.items()}


def cmd_fetch(args: argparse.Namespace) -> int:
    catalog = load_catalog()
    deep = load_deep()
    now = int(time.time())
    stale_before = now - args.stale_days * 86400
    wanted = [e["id"] for e in catalog if needs_deep(e)]
    due = [i for i in wanted if (deep.get(i) or {}).get("t", 0) < stale_before]
    # Oldest first, so a capped run still works through the backlog in order.
    due.sort(key=lambda i: (deep.get(i) or {}).get("t", 0))
    batches = [due[i : i + DEEP_BATCH] for i in range(0, len(due), DEEP_BATCH)]
    if args.max_batches is not None:
        batches = batches[: args.max_batches]
    print(f"{len(wanted)} titles at the cap, {len(due)} due, {len(batches)} requests", flush=True)

    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "Anilili-konoha-build (AniList client 45552)",
            # See cmd_catalog: no Referer and no Authorization is a "temporarily disabled" 403.
            "Referer": "android-app://com.miruronative",
        }
    )
    anilist = AniList(session, args.min_interval)
    done = 0
    try:
        for index, batch in enumerate(batches, 1):
            data = anilist.post(DEEP_QUERY, {"ids": batch}, label=f"deep recs batch {index}")
            rows = deep_rows(data)
            for media_id in batch:
                # A title AniList no longer answers for keeps its old row rather than an empty one.
                if media_id in rows:
                    deep[media_id] = {"t": now, "r": rows[media_id]}
                    done += 1
            if index % 10 == 0:
                print(f"  {index}/{len(batches)} requests, {done} titles", flush=True)
                save_deep(deep)
    except KeyboardInterrupt:
        print("\ninterrupted — saving what was fetched", flush=True)
    save_deep(deep)
    print(f"wrote {done} titles to {DEEP_FILE} ({len(deep)} cached)")
    return 0


# --------------------------------------------------------------------------------------------
# emit
# --------------------------------------------------------------------------------------------


def start_key(entry: dict) -> tuple:
    start = entry.get("startDate") or {}
    return (start.get("year") or 9999, start.get("month") or 99, start.get("day") or 99, entry["id"])


def franchise_roots(catalog: list[dict]) -> dict[int, int]:
    """Each title's franchise root: the earliest TV/ONA/film of its season-and-recap component.

    A component, not the app's bounded 16-entry walk: the root has to be the same answer from every
    member, or "watched season 3" would fail to rule out season 1.
    """
    by_id = {e["id"]: e for e in catalog}
    parent = {i: i for i in by_id}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for entry in catalog:
        for edge in ((entry.get("relations") or {}).get("edges")) or []:
            if edge.get("relationType") not in FRANCHISE_RELATIONS:
                continue
            other = (edge.get("node") or {}).get("id")
            if other in parent:
                a, b = find(entry["id"]), find(other)
                if a != b:
                    parent[a] = b

    members: dict[int, list[dict]] = {}
    for media_id, entry in by_id.items():
        members.setdefault(find(media_id), []).append(entry)
    roots: dict[int, int] = {}
    for group in members.values():
        preferred = [e for e in group if e.get("format") in ROOT_FORMATS] or group
        root = min(preferred, key=start_key)["id"]
        for entry in group:
            roots[entry["id"]] = root
    return roots


def adjacent_roots(catalog: list[dict], roots: dict[int, int]) -> dict[int, set[int]]:
    """For each franchise root, the roots of titles any of its members side-stories or spins off."""
    out: dict[int, set[int]] = {}
    for entry in catalog:
        mine = roots.get(entry["id"], entry["id"])
        for edge in ((entry.get("relations") or {}).get("edges")) or []:
            if edge.get("relationType") not in ADJACENT_RELATIONS:
                continue
            other = (edge.get("node") or {}).get("id")
            if other:
                theirs = roots.get(other, other)
                if theirs != mine:
                    out.setdefault(mine, set()).add(theirs)
                    out.setdefault(theirs, set()).add(mine)
    return out


def pair_votes(recs: dict[int, list[tuple[int, int]]], known: set[int]) -> dict[int, dict[int, int]]:
    """Each title's recommendation partners and the votes on the pair, from either side.

    An AniList recommendation is one object linking two titles, so its votes belong to the pair;
    the two titles' lists can still disagree on whether it is in their top N, which is why both
    sides are read.
    """
    out: dict[int, dict[int, int]] = {}
    for source, items in recs.items():
        for target, votes in items:
            if target not in known or target == source:
                continue
            for a, b in ((source, target), (target, source)):
                row = out.setdefault(a, {})
                row[b] = max(row.get(b, 0), votes)
    return out


def rec_strengths(recs: list[tuple[int, int]]) -> dict[int, float]:
    """Votes as 0..1 strengths, relative to the title's own best and scaled by how believable it is.

    Relative, because vote counts track popularity: Frieren's 10th recommendation has more votes
    than an obscure title's 1st. Scaled, because a title whose best has one vote should not hand
    that vote full weight.
    """
    if not recs:
        return {}
    best = max(v for _, v in recs)
    confidence = min(1.0, math.log1p(best) / math.log1p(VOTE_CONFIDENCE))
    return {t: (math.log1p(v) / math.log1p(best)) * confidence for t, v in recs}


def tag_matrix(catalog: list[dict]) -> tuple[list[int], np.ndarray]:
    """L2-normalised TF-IDF rows over ranked tags and genres, for titles with enough tags."""
    rows = []
    for entry in catalog:
        tags = entry.get("tags") or []
        if len(tags) < MIN_TAGS:
            continue
        features: dict[str, float] = {}
        for tag in tags:
            weight = (tag.get("rank") or 0) / 100.0
            if tag.get("isMediaSpoiler") or tag.get("isGeneralSpoiler"):
                weight *= SPOILER_TAG_WEIGHT
            if weight > 0:
                features["t:" + tag["name"]] = weight
        for genre in entry.get("genres") or []:
            features["g:" + genre] = GENRE_WEIGHT
        rows.append((entry["id"], features))
    vocabulary: dict[str, int] = {}
    df: dict[str, int] = {}
    for _, features in rows:
        for name in features:
            vocabulary.setdefault(name, len(vocabulary))
            df[name] = df.get(name, 0) + 1
    total = len(rows)
    matrix = np.zeros((total, len(vocabulary)), dtype=np.float32)
    for r, (_, features) in enumerate(rows):
        for name, weight in features.items():
            matrix[r, vocabulary[name]] = weight * math.log(total / df[name])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return [i for i, _ in rows], matrix / norms


def tag_neighbours(
    ids: list[int],
    matrix: np.ndarray,
    eligible: np.ndarray,
    roots: dict[int, int],
    k: int = TAG_NEIGHBOURS,
    block: int = 1024,
) -> dict[int, list[tuple[int, float]]]:
    """Each title's top-k tag-similar titles, excluding itself and its own franchise."""
    out: dict[int, list[tuple[int, float]]] = {}
    root_of = np.array([roots.get(i, i) for i in ids])
    take = min(k + 16, len(ids))
    for start in range(0, len(ids), block):
        sims = matrix[start : start + block] @ matrix.T
        sims[:, ~eligible] = -1
        top = np.argpartition(-sims, take - 1, axis=1)[:, :take]
        for row, candidates in enumerate(top):
            me = start + row
            ranked = sorted(candidates, key=lambda c: -sims[row, c])
            picked = []
            for c in ranked:
                if c == me or root_of[c] == root_of[me]:
                    continue
                cosine = float(sims[row, c])
                if cosine < MIN_TAG_COSINE:
                    break
                picked.append((ids[c], cosine))
                if len(picked) == k:
                    break
            out[ids[me]] = picked
    return out


def build_graph(
    catalog: list[dict],
    recs: dict[int, list[tuple[int, int]]],
    tags: dict[int, list[tuple[int, float]]],
    roots: dict[int, int],
) -> dict[int, list[tuple[int, int]]]:
    """Blend both sources into each title's weighted neighbour list (weights 1..100, best first).

    Votes are read on the title's own scale ([rec_strengths]): Frieren's 30-vote partner is a weak
    one, an obscure title's 30-vote partner its strongest. Taking the stronger of the two sides
    instead let every obscure title that names Frieren first sit at the top of Frieren's list.

    Targets are written as their franchise root, keeping the best weight any member earned: the app
    recommends franchises, and five Demon Slayer entries would otherwise spend five of the forty
    slots. Neighbours in the title's own franchise, or one side story away from it, are dropped —
    "you watched season 1, try season 2" is not a discovery.
    """
    known = {e["id"] for e in catalog}
    adjacent = adjacent_roots(catalog, roots)
    partners = pair_votes(recs, known)

    graph: dict[int, list[tuple[int, int]]] = {}
    for entry in catalog:
        me = entry["id"]
        my_root = roots.get(me, me)
        excluded = adjacent.get(my_root, set())
        rec = rec_strengths(sorted((partners.get(me) or {}).items(), key=lambda kv: -kv[1]))
        tag = dict(tags.get(me) or [])
        blended: dict[int, float] = {}
        for target in set(rec) | set(tag):
            root = roots.get(target, target)
            if root == my_root or root in excluded:
                continue
            if target in rec:
                weight = REC_SHARE * rec[target] + TAG_SHARE_WITH_REC * tag.get(target, 0.0)
            else:
                weight = TAG_SHARE_ALONE * tag[target]
            if weight >= MIN_WEIGHT and weight > blended.get(root, 0.0):
                blended[root] = weight
        best = sorted(blended.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_NEIGHBOURS]
        if best:
            graph[me] = [(t, max(1, min(100, round(w * 100)))) for t, w in best]
    return graph


def flags_of(entry: dict) -> int:
    flags = 0
    if entry.get("isAdult"):
        flags |= FLAG_ADULT
    if entry.get("status") == "NOT_YET_RELEASED":
        flags |= FLAG_UNRELEASED
    return flags


def render(
    catalog: list[dict],
    graph: dict[int, list[tuple[int, int]]],
    roots: dict[int, int],
    generated: str,
) -> str:
    """The file body. A title is written when it says anything a default would not: it has
    neighbours, a root other than itself, or a flag."""
    lines = [f"{HEADER}\t{FORMAT_VERSION}\t{generated}"]
    for entry in sorted(catalog, key=lambda e: e["id"]):
        me = entry["id"]
        root = roots.get(me, me)
        flags = flags_of(entry)
        edges = graph.get(me) or []
        if not edges and root == me and not flags:
            continue
        lines.append(f"{me}\t{root}\t{flags}\t" + ",".join(f"{t}:{w}" for t, w in edges))
    return "\n".join(lines) + "\n"


def gzip_bytes(text: str) -> bytes:
    """Deterministic gzip — no timestamp, no filename — so an unchanged graph is an unchanged file
    and the daily commit does not churn a binary that says the same thing."""
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0, compresslevel=9) as out:
        out.write(text.encode("utf-8"))
    return buffer.getvalue()


def cmd_emit(args: argparse.Namespace) -> int:
    started = time.time()
    catalog = load_catalog()
    deep = load_deep()
    roots = franchise_roots(catalog)
    recs = rec_lists(catalog, deep)

    ids, matrix = tag_matrix(catalog)
    by_id = {e["id"]: e for e in catalog}
    eligible = np.array(
        [
            (by_id[i].get("popularity") or 0) >= MIN_TAG_TARGET_POPULARITY
            and by_id[i].get("format") not in TAG_TARGET_EXCLUDED_FORMATS
            for i in ids
        ]
    )
    tags = tag_neighbours(ids, matrix, eligible, roots)

    graph = build_graph(catalog, recs, tags, roots)
    # The date only, so a rebuild on the same day with the same inputs is byte-identical.
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    body = render(catalog, graph, roots, generated)
    out = pathlib.Path(args.out) / OUTPUT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = gzip_bytes(body)
    out.write_bytes(payload)

    edges = sum(len(v) for v in graph.values())
    with_recs = sum(1 for v in recs.values() if v)
    print(
        f"wrote {out}: {len(graph)} titles with neighbours, {edges} edges, "
        f"{len(body.encode()) / 1e6:.1f} MB text, {len(payload) / 1e6:.2f} MB gzip, "
        f"{len(deep)} deep / {with_recs} with votes, {time.time() - started:.0f}s"
    )
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch", help="deep AniList recommendations for titles at the cap")
    fetch.add_argument("--stale-days", type=int, default=7)
    fetch.add_argument("--max-batches", type=int, default=None)
    fetch.add_argument("--min-interval", type=float, default=MIN_INTERVAL)
    fetch.set_defaults(func=cmd_fetch)

    emit = commands.add_parser("emit", help="write data/recs/neighbours-v1.tsv.gz")
    emit.add_argument("--out", required=True)
    emit.set_defaults(func=cmd_emit)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
