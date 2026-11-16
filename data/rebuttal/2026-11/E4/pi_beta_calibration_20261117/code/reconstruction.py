"""E4-rule-preserving reconstruction of several fixed-point predictions.

Columns may contain different beta predictions for the same selected points.
SciPy builds one geometry for these columns. This module neither changes the
interpolation rule nor caches a map across requests. It does not quantize or
regenerate coordinates, backfill retained points, or use a dense distance array.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

IMPLEMENTATION_VERSION = 'e4_scipy_multichannel_v1'


def reconstruct_many(source_coords, predictions, query_coords, *, return_metadata=False):
    """Return a float64 ``(N_query, C)`` field under the original E4 rule.

    ``source_coords`` and ``query_coords`` are passed to SciPy at their provided
    precision. E4's caller supplies float32-quantized values stored as float64.
    Predictions may be FP32 or FP64; no input is downcast. Non-finite predictions
    are rejected before interpolation because they are outside this experiment's
    finite-field contract. QHull degeneracy errors propagate as in original E4.
    """
    source = np.asarray(source_coords)
    query = np.asarray(query_coords)
    values = np.asarray(predictions)
    if source.ndim != 2 or source.shape[1] != 2:
        raise ValueError('source_coords must have shape (N_source, 2)')
    if query.ndim != 2 or query.shape[1] != 2:
        raise ValueError('query_coords must have shape (N_query, 2)')
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != len(source) or values.shape[1] < 1:
        raise ValueError('predictions must have shape (N_source, C), C >= 1')
    if any(np.iscomplexobj(a) for a in (source, query, values)):
        raise ValueError('E4 uses real coordinates and fields')
    if not all(np.isfinite(a).all() for a in (source, query, values)):
        raise ValueError('coordinates and predictions must be finite')

    linear = LinearNDInterpolator(source, values, fill_value=np.nan)
    result = np.asarray(linear(query), dtype=np.float64).reshape(len(query), values.shape[1])
    missing = ~np.isfinite(result)
    missing_rows = missing.any(axis=1)
    if missing_rows.any():
        fallback = np.asarray(NearestNDInterpolator(source, values)(query[missing_rows]), dtype=np.float64)
        # The original scalar routine only replaces missing entries. Keep this
        # exact rule even if a future numeric edge case affects just one column.
        result[missing_rows] = np.where(missing[missing_rows], fallback, result[missing_rows])
    if not np.isfinite(result).all():
        raise ValueError('reconstruction produced non-finite output')
    if return_metadata:
        return result, {
            'implementation': IMPLEMENTATION_VERSION,
            'source_points': len(source), 'query_points': len(query),
            'channels': values.shape[1], 'source_coordinate_dtype': str(source.dtype),
            'query_coordinate_dtype': str(query.dtype), 'prediction_dtype': str(values.dtype),
            'output_dtype': str(result.dtype), 'fallback_rows': int(missing_rows.sum()),
            'fallback_count_per_channel': missing.sum(axis=0).astype(int).tolist(),
            'output_bytes': result.nbytes,
            'memory_scope': 'output array bytes only; not a process or GPU peak',
        }
    return result


def _validate(output_path=None):
    """Independent scalar-reference equivalence, using no GPU or model calls."""
    import ast
    import datetime as dt
    import hashlib
    import json
    from pathlib import Path
    import scipy

    current = Path(__file__).resolve().parents[2]
    original = current / 'rebuttal_20261112/experiments/E4_darcy_resolution/code/run_resolution.py'
    tree = ast.parse(original.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'reconstruct')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(original), 'exec'), namespace)
    scalar_reference = namespace['reconstruct']
    cases = []

    def synthetic_values(coords, channels=11, dtype=np.float32, strided=False):
        x, y = coords[:, 0], coords[:, 1]
        count = channels * 2 if strided else channels
        columns = [np.sin((j+1)*.37*x) + np.cos((j+1)*.23*y)
                   + ((-1)**j)*.17*x*y + j*.013 for j in range(count)]
        result = np.column_stack(columns).astype(dtype)
        return result[:, ::2] if strided else result

    def compare(name, source, values, query, scope, source_file=None):
        originals = [np.array(a, copy=True) for a in (source, values, query)]
        expected_columns = []
        counts = []
        for j in range(values.shape[1]):
            field, count = scalar_reference(source, values[:, j], query)
            expected_columns.append(field); counts.append(count)
        expected = np.column_stack(expected_columns)
        actual, metadata = reconstruct_many(source, values, query, return_metadata=True)
        difference = np.abs(actual - expected)
        bound = 1e-6 + 1e-5*np.abs(expected)
        violations = int(np.count_nonzero(difference > bound))
        assert violations == 0
        assert actual.dtype == np.float64 and actual.shape == expected.shape
        assert metadata['fallback_count_per_channel'] == counts
        assert all(np.array_equal(a, b) and a.dtype == b.dtype
                   for a, b in zip((source, values, query), originals))
        # A normalized downstream-error diagnostic on a fixed finite target.
        # This target is a test fixture, not a physical solution or test result.
        target = 2. + np.sin(query[:, 0]*.41) + np.cos(query[:, 1]*.29)
        old_metrics = np.linalg.norm(expected-target[:, None], axis=0)/np.linalg.norm(target)
        new_metrics = np.linalg.norm(actual-target[:, None], axis=0)/np.linalg.norm(target)
        metric_difference = float(np.max(np.abs(old_metrics-new_metrics)))
        assert metric_difference <= 1e-8 + 1e-5*float(np.max(np.abs(old_metrics)))
        record = dict(name=name, scope=scope, source_file=None if source_file is None else str(source_file),
            source_file_sha256=None if source_file is None else hashlib.sha256(source_file.read_bytes()).hexdigest(),
            synthetic_prediction_values=True, physical_accuracy_evidence=False,
            max_absolute_difference=float(difference.max()),
            relative_norm_difference=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-300)),
            absolute_difference_quantiles={str(q):float(np.quantile(difference,q)) for q in [.5,.9,.99,.999,1.]},
            elementwise_violations=violations, downstream_fixture_metric_difference=metric_difference,
            inputs_unchanged=True, metadata=metadata,
            source_coordinate_sha256=hashlib.sha256(np.ascontiguousarray(source).tobytes()).hexdigest(),
            query_coordinate_sha256=hashlib.sha256(np.ascontiguousarray(query).tobytes()).hexdigest())
        cases.append(record)

    square = np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]], dtype=np.float64)
    queries = np.array([[0.,0.],[.5,0.],[.5,.5],[1.,1.],[.5,-1.],[1.5,.5],[-.5,.5]], dtype=np.float64)
    compare('edges_exact_points_and_equidistant_fallback', square, synthetic_values(square, 4), queries,
            'numeric contract: vertices, shared edge, equal nearest distances and outside hull')
    duplicate = np.vstack([square, square[0]])
    values = synthetic_values(duplicate, 4); values[-1] += np.array([1.,-2.,3.,-4.],np.float32)
    compare('duplicate_coordinates', duplicate, values, queries,
            'numeric contract: duplicate coordinate values and original QHull/KDTree source ordering')
    precise = np.vstack([square, [[.5+3e-9,.5-4e-9]]])
    precise_query = np.vstack([queries, precise[-1], [.5,.5]])
    compare('native_float64_coordinates_and_values', precise, synthetic_values(precise, 5, np.float64), precise_query,
            'numeric contract: sub-FP32 coordinate offsets preserved; no silent coordinate downcast')

    examples = current / 'rebuttal_20261112/experiments/E4_darcy_resolution/records/examples'
    for n, k in [(85,3612),(141,9940),(211,22260)]:
        path = examples / f'case1000_n{n}_pi_k{k}.npz'
        with np.load(path) as item:
            source = item['reference_coords'][item['indices']]
            query = item['evaluation_coords']
        assert np.array_equal(source.astype(np.float32).astype(np.float64), source)
        values = synthetic_values(source, 11, np.float32, strided=True)
        assert not values.flags.c_contiguous
        compare(f'real_E4_PI_geometry_{n}_50pct', source, values, query,
                'actual saved E4 selected geometry and 211² query grid; synthetic 11-channel values only', path)

    # This call additionally confirms the root runner's single-channel interface.
    single = reconstruct_many(square, synthetic_values(square, 1)[:, 0], queries)
    np.testing.assert_array_equal(single[:, 0], scalar_reference(square, synthetic_values(square, 1)[:, 0], queries)[0])
    receipt = dict(status='passed_for_rule_preserving_multichannel_reconstruction', passed=True,
        implementation=IMPLEMENTATION_VERSION, checked_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        implementation_path=str(Path(__file__).resolve()),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_scalar_source=str(original), original_scalar_source_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        original_scalar_function='AST-extracted unmodified reconstruct(source, values, query) from E4 runner',
        numpy_version=np.__version__, scipy_version=scipy.__version__,
        cases=cases, case_count=len(cases), total_output_elements=sum(c['metadata']['query_points']*c['metadata']['channels'] for c in cases),
        max_absolute_difference=max(c['max_absolute_difference'] for c in cases),
        max_downstream_fixture_metric_difference=max(c['downstream_fixture_metric_difference'] for c in cases),
        elementwise_violations=sum(c['elementwise_violations'] for c in cases),
        starting_tolerance='abs(new-old) <= 1e-6 + 1e-5*abs(old); unchanged after observing outcomes',
        coordinate_policy='pass provided coordinate values to SciPy; no quantization, regeneration or float64-to-float32 downcast',
        retained_points='No additional exact-point backfill; original E4 interpolation behavior retained',
        fallback_policy='Original SciPy nearest-neighbor result at each non-finite linear output entry; original ties/source order',
        dense_distance_matrix=False, geometry_reuse='Within one call, beta predictions occupy independent field channels on the same point set',
        cache_reuse_across_requests=False, single_channel_interface_passed=True,
        evidence_scope='CPU engineering validation; synthetic values on actual geometry and numeric contracts; no model inference, physical accuracy, GPU correctness or speed claim')
    destination = Path(output_path) if output_path else Path(__file__).resolve().parents[1] / 'reconstruction_validation.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(receipt, indent=2) + '\n'); temporary.replace(destination)
    print(json.dumps({k:receipt[k] for k in ['status','case_count','total_output_elements','max_absolute_difference','elementwise_violations']}, indent=2))
    return receipt


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--validate',action='store_true');parser.add_argument('--output')
    args=parser.parse_args()
    if not args.validate:parser.error('Use --validate to run the CPU equivalence check')
    _validate(args.output)
