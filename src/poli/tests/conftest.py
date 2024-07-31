def pytest_collection_modifyitems(items, config):
    for item in items:
        markers = item.iter_markers()
        is_unmarked = True
        for marker in markers:
            if marker.name != "parametrize":
                is_unmarked = False
                break

        if is_unmarked:
            item.add_marker("unmarked")
