"""Gold steps with list values in ``str`` parameters become one scalar call each.

The rule is the second contract in opendatalab/Earth-Agent issue 13: keep the
published scalar tool API and expand the ground truth. These cases do not need
the Earth-Agent checkout; the annotation map is passed in.
"""

from agentic_evals.tasks.earth_bench import metrics

TVDI = {"compute_tvdi": frozenset({"ndvi_path", "lst_path", "output_path"})}


def test_equal_str_lists_become_one_call_per_index():
    gold = [
        {
            "name": "compute_tvdi",
            "input": {
                "ndvi_path": ["a.tif", "b.tif"],
                "lst_path": ["c.tif", "d.tif"],
                "output_path": ["e.tif", "f.tif"],
                "note": "kept",
            },
        }
    ]
    expanded, unexpanded = metrics.expand_scalar_gold(gold, TVDI)
    assert unexpanded == []
    assert expanded == [
        {
            "name": "compute_tvdi",
            "input": {"ndvi_path": "a.tif", "lst_path": "c.tif", "output_path": "e.tif", "note": "kept"},
        },
        {
            "name": "compute_tvdi",
            "input": {"ndvi_path": "b.tif", "lst_path": "d.tif", "output_path": "f.tif", "note": "kept"},
        },
    ]
    assert gold[0]["input"]["ndvi_path"] == ["a.tif", "b.tif"]


def test_list_parameter_stays_one_call():
    gold = [
        {
            "name": "calculate_tif_average",
            "input": {"file_list": ["a.tif", "b.tif"], "output_path": "year.tif"},
        }
    ]
    # file_list is list[str] in the published source, so it is not a scalar param.
    expanded, unexpanded = metrics.expand_scalar_gold(
        gold, {"calculate_tif_average": frozenset({"output_path"})}
    )
    assert unexpanded == []
    assert len(expanded) == 1
    assert expanded[0]["input"]["file_list"] == ["a.tif", "b.tif"]


def test_union_annotation_stays_one_call():
    # image_paths is ``str | list[str]``, so the loader omits it from the map.
    gold = [{"name": "calculate_threshold_ratio", "input": {"image_paths": ["a.tif"], "threshold": 0.75}}]
    expanded, unexpanded = metrics.expand_scalar_gold(gold, {})
    assert unexpanded == []
    assert len(expanded) == 1
    assert expanded[0]["input"]["image_paths"] == ["a.tif"]


def test_unequal_lengths_stay_one_call():
    gold = [
        {
            "name": "compute_tvdi",
            "input": {
                "ndvi_path": ["a.tif"],
                "lst_path": ["b.tif", "c.tif"],
                "output_path": ["d.tif"],
            },
        }
    ]
    expanded, unexpanded = metrics.expand_scalar_gold(gold, TVDI)
    assert len(expanded) == 1
    assert expanded[0]["input"]["lst_path"] == ["b.tif", "c.tif"]
    assert unexpanded == [
        {"step": 1, "tool": "compute_tvdi", "lengths": {"ndvi_path": 1, "lst_path": 2, "output_path": 1}}
    ]
