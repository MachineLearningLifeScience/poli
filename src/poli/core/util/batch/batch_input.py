from itertools import islice


def batched(iterable, chunk_size):
    iterator = iter(iterable)
    while chunk := tuple(islice(iterator, chunk_size)):
        yield chunk
