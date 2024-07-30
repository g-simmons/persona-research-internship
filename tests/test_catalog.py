from contrastive_tda.catalog import Catalog

def test_load_functions():
    catalog = Catalog()
    load_fns = [getattr(catalog, load_fn) for load_fn in [x for x in dir(catalog) if x.startswith("load_")]]
    for load_fn in load_fns:
        assert callable(load_fn)
        assert load_fn() is not None
    