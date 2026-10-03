"""Checks for recs_build.py — the neighbour graph the app's TasteEngine reads.

The file format is shared with `NeighbourGraph.kt`; `test_render_matches_the_app_format` pins the
exact bytes the Kotlin parser's tests also use. Plain asserts, no pytest, like test_konoha_build.py.

    scripts/.venv/Scripts/python.exe scripts/test_recs_build.py
"""
from __future__ import annotations

import gzip
import sys

import numpy as np

from recs_build import (
    adjacent_roots,
    build_graph,
    deep_rows,
    franchise_roots,
    gzip_bytes,
    needs_deep,
    pair_votes,
    rec_lists,
    rec_strengths,
    render,
    tag_matrix,
    tag_neighbours,
)


def media(mid, *, fmt="TV", year=2020, relations=(), recs=(), tags=(), genres=(), adult=False, status="FINISHED", popularity=10000):
    return {
        "id": mid,
        "format": fmt,
        "status": status,
        "isAdult": adult,
        "popularity": popularity,
        "startDate": {"year": year, "month": 1, "day": 1},
        "relations": {"edges": [{"relationType": t, "node": {"id": o}} for t, o in relations]},
        "recommendations": {"nodes": [{"rating": v, "mediaRecommendation": {"id": t}} for t, v in recs]},
        "tags": [{"name": n, "rank": r, "isMediaSpoiler": False, "isGeneralSpoiler": False} for n, r in tags],
        "genres": list(genres),
    }


def test_franchise_root_is_the_earliest_series_entry():
    catalog = [
        media(1, fmt="OVA", year=2005, relations=[("SEQUEL", 2)]),
        media(2, year=2006, relations=[("PREQUEL", 1), ("SEQUEL", 3)]),
        media(3, year=2010, relations=[("PREQUEL", 2), ("SIDE_STORY", 9)]),
        media(4, fmt="MOVIE", year=2011, relations=[("SUMMARY", 3)]),
        media(9, year=2012),
    ]
    roots = franchise_roots(catalog)
    # The 2005 OVA is older, but a TV entry is what a viewer knows the franchise as.
    assert roots == {1: 2, 2: 2, 3: 2, 4: 2, 9: 9}, roots
    assert adjacent_roots(catalog, roots) == {2: {9}, 9: {2}}


def test_votes_are_read_on_each_titles_own_scale():
    # Popular title 1: best partner 1000 votes. Obscure title 5 names 1 as its best, with 30 votes.
    recs = {1: [(2, 1000), (3, 500)], 5: [(1, 30)]}
    partners = pair_votes(recs, {1, 2, 3, 5})
    assert partners[1] == {2: 1000, 3: 500, 5: 30}
    assert partners[5] == {1: 30}
    on_one = rec_strengths(sorted(partners[1].items(), key=lambda kv: -kv[1]))
    on_five = rec_strengths(list(partners[5].items()))
    assert on_one[2] == 1.0
    assert on_one[5] < 0.6, on_one
    assert on_five[1] == 1.0


def test_a_single_vote_is_not_believed_fully():
    assert rec_strengths([(2, 1)])[2] < 0.25
    assert rec_strengths([(2, 30)])[2] == 1.0


def test_graph_collapses_franchises_and_drops_its_own():
    catalog = [
        media(1, recs=[(10, 100), (11, 90), (2, 80), (20, 50)], relations=[("SEQUEL", 2), ("SPIN_OFF", 30)]),
        media(2, relations=[("PREQUEL", 1)]),
        media(10, year=2010, relations=[("SEQUEL", 11)]),
        media(11, year=2012, relations=[("PREQUEL", 10)]),
        media(20),
        media(30, recs=[(1, 5)]),
    ]
    roots = franchise_roots(catalog)
    recs = rec_lists(catalog, {})
    graph = build_graph(catalog, recs, {}, roots)
    targets = [t for t, _ in graph[1]]
    # 10 and 11 are one franchise, written once as 10; 2 is 1's own sequel; 30 its spin-off.
    assert targets == [10, 20], graph[1]
    assert graph[1][0][1] == 80  # REC_SHARE * 1.0


def test_tag_neighbours_skip_own_franchise_and_unknown_targets():
    catalog = [
        media(1, tags=[("Travel", 90), ("Magic", 80), ("Elf", 70)], genres=["Fantasy"], relations=[("SEQUEL", 2)]),
        media(2, tags=[("Travel", 90), ("Magic", 80), ("Elf", 70)], genres=["Fantasy"], relations=[("PREQUEL", 1)]),
        media(3, tags=[("Travel", 85), ("Magic", 75), ("Elf", 60)], genres=["Fantasy"]),
        media(4, tags=[("Travel", 85), ("Magic", 75), ("Elf", 60)], genres=["Fantasy"], popularity=50),
        media(5, tags=[("Mecha", 90), ("Space", 80), ("War", 70)], genres=["Action"]),
        media(6, tags=[("Mecha", 90)]),  # too few tags to be compared at all
    ]
    roots = franchise_roots(catalog)
    ids, matrix = tag_matrix(catalog)
    assert 6 not in ids
    by_id = {e["id"]: e for e in catalog}
    eligible = np.array([by_id[i]["popularity"] >= 2000 for i in ids])
    near = tag_neighbours(ids, matrix, eligible, roots)
    assert [t for t, _ in near[1]] == [3], near[1]


def test_deep_rows_merge_pages_and_rec_lists_prefer_newer_catalogue_votes():
    data = {
        "p1": {"media": [{"id": 1, "recommendations": {"nodes": [{"rating": 50, "mediaRecommendation": {"id": 2}}]}}]},
        "p2": {"media": [{"id": 1, "recommendations": {"nodes": [
            {"rating": 3, "mediaRecommendation": {"id": 3}},
            {"rating": 0, "mediaRecommendation": {"id": 4}},
            {"rating": 9, "mediaRecommendation": None},
        ]}}]},
    }
    assert deep_rows(data) == {1: [(2, 50), (3, 3)]}
    catalog = [media(1, recs=[(2, 60)])]
    assert rec_lists(catalog, {1: {"t": 0, "r": [[2, 50], [3, 3]]}}) == {1: [(2, 60), (3, 3)]}


def test_only_titles_at_the_cap_need_the_deep_pass():
    assert needs_deep(media(1, recs=[(i, 5) for i in range(2, 12)]))
    assert not needs_deep(media(1, recs=[(i, 5) for i in range(2, 11)]))


def test_render_matches_the_app_format():
    catalog = [
        media(1),
        media(2, adult=True),
        media(3, status="NOT_YET_RELEASED"),
        media(4),  # nothing to say: left out
        media(5, relations=[("PREQUEL", 1)], year=2021),
    ]
    roots = {1: 1, 2: 2, 3: 3, 4: 4, 5: 1}
    body = render(catalog, {1: [(2, 80), (3, 7)]}, roots, "2026-10-03")
    assert body == (
        "#anilili-neighbours\t1\t2026-10-03\n"
        "1\t1\t0\t2:80,3:7\n"
        "2\t2\t1\t\n"
        "3\t3\t2\t\n"
        "5\t1\t0\t\n"
    ), body
    packed = gzip_bytes(body)
    assert gzip.decompress(packed).decode() == body
    assert packed == gzip_bytes(body), "gzip output must be deterministic"


def main() -> int:
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok   {name}")
        except AssertionError as error:
            failed += 1
            print(f"FAIL {name}: {error}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
