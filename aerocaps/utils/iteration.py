import typing


def flatten_arbitrarily_nested_list_of_lists(xs):
    for x in xs:
        if isinstance(x, typing.Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten_arbitrarily_nested_list_of_lists(x)
        else:
            yield x
