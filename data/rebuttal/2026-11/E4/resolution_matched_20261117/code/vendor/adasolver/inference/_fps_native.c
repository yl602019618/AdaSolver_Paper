/* Same Euclidean FPS recurrence and first-index tie rule as NumPy.
 * Compile without fast-math or fused multiply-add reassociation. */
#include <stdint.h>
#include <math.h>

int adasolver_fps(const double *points, double *nearest, int64_t count,
                 int64_t dimensions, int64_t take, int64_t first_if_infinite,
                 int64_t *selected) {
    if (count < 1 || dimensions < 1 || take < 0 || take > count) return 1;
    for (int64_t step = 0; step < take; ++step) {
        int64_t best = 0;
        for (int64_t j = 1; j < count; ++j)
            if (nearest[j] > nearest[best]) best = j;
        if (!isfinite(nearest[best])) {
            if (step != 0 || first_if_infinite < 0 || first_if_infinite >= count) return 2;
            best = first_if_infinite;
        }
        selected[step] = best;
        const double *anchor = points + best * dimensions;
        for (int64_t j = 0; j < count; ++j) {
            const double *point = points + j * dimensions;
            double squared = 0.;
            if (dimensions == 3) {
                double dx = point[0] - anchor[0];
                double dy = point[1] - anchor[1];
                double dz = point[2] - anchor[2];
                squared = (dx * dx + dy * dy) + dz * dz;
            } else {
                for (int64_t d = 0; d < dimensions; ++d) {
                    double delta = point[d] - anchor[d];
                    squared += delta * delta;
                }
            }
            double distance = sqrt(squared);
            if (distance < nearest[j]) nearest[j] = distance;
        }
        nearest[best] = -1.;
    }
    return 0;
}
