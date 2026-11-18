def budget_count(full_count: int, fraction: float, minimum: int = 2) -> int:
    if full_count < minimum:
        raise ValueError("full_count is smaller than the required minimum")
    if not 0 < fraction <= 1:
        raise ValueError("budget fraction must be in (0, 1]")
    return min(full_count, max(minimum, int(round(full_count * fraction))))
