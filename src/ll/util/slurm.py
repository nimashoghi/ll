class SlurmParseException(Exception):
    pass


def _expand_id_suffix(suffix_parts: str) -> list[str]:
    """Parse the a suffix formatted like "1-3,5,8" into
    the list of numeric values 1,2,3,5,8.
    """
    suffixes = []
    for suffix_part in suffix_parts.split(","):
        if "-" in suffix_part:
            low, high = suffix_part.split("-")
            int_length = len(low)
            for num in range(int(low), int(high) + 1):
                suffixes.append(f"{num:0{int_length}}")
        else:
            suffixes.append(suffix_part)
    return suffixes


def _parse_node_group(node_list: str, pos: int, parsed: list[str]) -> int:
    """Parse a node group of the form PREFIX[1-3,5,8] and return
    the position in the string at which the parsing stopped
    """
    prefixes = [""]
    while pos < len(node_list):
        c = node_list[pos]
        if c == ",":
            parsed.extend(prefixes)
            return pos + 1
        if c == "[":
            last_pos = node_list.index("]", pos)
            suffixes = _expand_id_suffix(node_list[pos + 1 : last_pos])
            prefixes = [prefix + suffix for prefix in prefixes for suffix in suffixes]
            pos = last_pos + 1
        else:
            for i, prefix in enumerate(prefixes):
                prefixes[i] = prefix + c
            pos += 1
    parsed.extend(prefixes)
    return pos


def parse_slurm_node_list(node_list: str):
    pos = 0
    parsed: list[str] = []
    while pos < len(node_list):
        pos = _parse_node_group(node_list, pos, parsed)
    return parsed
