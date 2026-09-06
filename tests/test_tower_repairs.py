from cftn_v3.tower_repairs import dataset,verify,ORDER

def test_repair_panels():
    for tower in ORDER:
        data=dataset(tower)
        seen=set()
        for items in data.values():
            assert len(items)>=64
            for r in items:
                assert verify(r)
                assert r['semantic_id'] not in seen
                seen.add(r['semantic_id'])
