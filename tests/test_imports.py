def test_imports_all():
    pass


def test_SKAModel_methods_entropy():
    import inspect

    import ska.wrappers
    from ska.model import SKAModel

    model_methods_pre_entropy = inspect.getmembers(
        SKAModel, predicate=inspect.isfunction
    ).copy()

    import ska.entropy

    model_methods_after_entropy = inspect.getmembers(
        SKAModel, predicate=inspect.isfunction
    ).copy()

    for method in model_methods_pre_entropy:
        assert method in model_methods_after_entropy

    model_methods_after_entropy_names = [
        method[0] for method in model_methods_after_entropy
    ]

    for element in inspect.getmembers(ska.entropy):
        if inspect.isfunction(element[1]) and element not in inspect.getmembers(
            ska.wrappers
        ):
            assert element[0] in model_methods_after_entropy_names

    pass


def test_SKAModel_methods_visualize():
    import inspect

    import ska.wrappers
    from ska.model import SKAModel

    model_methods_pre_vis = inspect.getmembers(
        SKAModel, predicate=inspect.isfunction
    ).copy()

    import ska.visualization

    model_methods_after_vis = inspect.getmembers(
        SKAModel, predicate=inspect.isfunction
    ).copy()

    for method in model_methods_pre_vis:
        assert method in model_methods_after_vis

    model_methods_after_vis_names = [method[0] for method in model_methods_after_vis]

    for element in inspect.getmembers(ska.visualization):
        if inspect.isfunction(element[1]) and element not in inspect.getmembers(
            ska.wrappers
        ):
            assert element[0] in model_methods_after_vis_names

    pass
