"""VTK-based ShapeNet-Car drag coefficient metric.

The physical convention follows the audited NSL-Car implementation: z is the
streamwise direction, pressure and wall shear are integrated over the surface,
and the frontal reference area is the x-y convex-hull area.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable
from dataclasses import dataclass
import os

import numpy as np


DEFAULT_SPEED = 72.0 / 3.6
DEFAULT_AIR_DENSITY = 0.3
DEFAULT_VISCOSITY = 1.8e-5


def _require_vtk():
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError as exc:
        raise RuntimeError(
            "Cd integration requires optional VTK or an integrator callback"
        ) from exc
    return vtk, vtk_to_numpy


def _read_grid(vtk, path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()
    if grid is None or grid.GetNumberOfPoints() == 0:
        raise ValueError(f"VTK mesh is empty: {path}")
    return grid


def _surface(vtk, grid):
    filt = vtk.vtkDataSetSurfaceFilter()
    filt.SetInputData(grid)
    filt.Update()
    output = filt.GetOutput()
    if output is None or output.GetNumberOfCells() == 0:
        raise ValueError("VTK mesh has no surface cells")
    return output


def _cell_normals(vtk, vtk_to_numpy, grid):
    poly = _surface(vtk, grid)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(poly)
    normals.SetAutoOrientNormals(1)
    normals.SetConsistency(1)
    normals.SetComputeCellNormals(1)
    normals.SetComputePointNormals(0)
    normals.Update()
    values = normals.GetOutput().GetCellData().GetNormals()
    if values is None:
        raise ValueError("VTK failed to compute surface cell normals")
    return vtk_to_numpy(values), poly


def _cell_areas(vtk, poly) -> np.ndarray:
    points = poly.GetPoints()
    cells = poly.GetPolys()
    areas = np.zeros(cells.GetNumberOfCells(), dtype=np.float64)
    cells.InitTraversal()
    cell = vtk.vtkIdList()
    index = 0
    while cells.GetNextCell(cell):
        if cell.GetNumberOfIds() != 4:
            raise ValueError("ShapeNet-Car Cd mesh must contain quadrilateral surface cells")
        p1 = np.asarray(points.GetPoint(cell.GetId(0)))
        p2 = np.asarray(points.GetPoint(cell.GetId(1)))
        p3 = np.asarray(points.GetPoint(cell.GetId(2)))
        p4 = np.asarray(points.GetPoint(cell.GetId(3)))
        areas[index] = 0.5 * (
            np.linalg.norm(np.cross(p2 - p1, p3 - p1))
            + np.linalg.norm(np.cross(p3 - p1, p4 - p1))
        )
        index += 1
    if not np.isfinite(areas).all() or np.any(areas <= 0):
        raise ValueError("VTK surface contains non-positive cell areas")
    return areas


def _velocity_gradient(vtk, grid, velocity: np.ndarray) -> np.ndarray:
    if velocity.shape != (grid.GetNumberOfPoints(), 3):
        raise ValueError(
            f"velo_surf must have shape ({grid.GetNumberOfPoints()}, 3), got {velocity.shape}"
        )
    velocity_array = vtk.vtkDoubleArray()
    velocity_array.SetNumberOfComponents(3)
    velocity_array.SetNumberOfTuples(grid.GetNumberOfPoints())
    velocity_array.SetName("Velocity")
    for index, value in enumerate(velocity):
        velocity_array.SetTuple(index, value)
    grid.GetPointData().AddArray(velocity_array)
    poly = _surface(vtk, grid)
    velocity_array = poly.GetPointData().GetArray("Velocity")
    if velocity_array is None:
        raise ValueError("VTK failed to propagate surface velocity values")

    points = poly.GetPoints()
    cells = poly.GetPolys()
    gradient = np.zeros((cells.GetNumberOfCells(), 3), dtype=np.float64)
    cells.InitTraversal()
    cell = vtk.vtkIdList()
    index = 0
    while cells.GetNextCell(cell):
        if cell.GetNumberOfIds() != 4:
            raise ValueError("ShapeNet-Car Cd mesh must contain quadrilateral surface cells")
        points4 = [np.asarray(points.GetPoint(cell.GetId(i))) for i in range(4)]
        velocity4 = [
            np.asarray(velocity_array.GetTuple(cell.GetId(i))) for i in range(4)
        ]
        p1, p2, p3, p4 = points4
        u1, u2, u3, u4 = velocity4
        du_dx = (u2 - u1 + u3 - u4) / (np.linalg.norm(p2 - p1 + p3 - p4) + 1e-8)
        du_dy = (u3 - u1 + u4 - u2) / (np.linalg.norm(p3 - p1 + p4 - p2) + 1e-8)
        du_dz = (u4 - u1 + u2 - u3) / (np.linalg.norm(p4 - p1 + p2 - p3) + 1e-8)
        gradient[index] = du_dx + du_dy + du_dz
        index += 1
    return gradient


def surface_pressure_drag_influence(
    obj_name: str,
    data_root: str | Path,
    *,
    fold: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return VTK surface coordinates and fixed pressure-drag influence.

    The returned point influence distributes each cell's ``area * normal_z``
    equally over its four vertices.  It is geometry-only and therefore safe for
    acquisition; callers must still validate coordinate alignment with the
    adapter cloud before using it.
    """
    vtk, vtk_to_numpy = _require_vtk()
    mesh_path = Path(data_root).expanduser().resolve() / f"param{fold}" / str(obj_name) / "quadpress_smpl.vtk"
    grid = _read_grid(vtk, mesh_path)
    normals, poly = _cell_normals(vtk, vtk_to_numpy, grid)
    areas = _cell_areas(vtk, poly)
    if len(normals) != len(areas):
        raise ValueError("VTK normal and area counts do not match")
    coordinates = vtk_to_numpy(grid.GetPoints().GetData()).astype(np.float64, copy=False)
    influence = np.zeros(len(coordinates), dtype=np.float64)
    cells = grid.GetCells()
    cells.InitTraversal()
    cell = vtk.vtkIdList()
    cell_index = 0
    while cells.GetNextCell(cell):
        if cell.GetNumberOfIds() != 4:
            raise ValueError("ShapeNet-Car Cd mesh must contain quadrilateral cells")
        contribution = float(areas[cell_index] * normals[cell_index, -1] / 4.0)
        for vertex in range(cell.GetNumberOfIds()):
            influence[cell.GetId(vertex)] += contribution
        cell_index += 1
    if cell_index != len(areas) or not np.isfinite(influence).all():
        raise ValueError("failed to construct finite VTK pressure-drag influence")
    return coordinates, influence


def compute_cd(
    obj_name: str,
    press_surf,
    velo_surf,
    data_root: str | Path,
    *,
    fold: int = 0,
    speed: float = DEFAULT_SPEED,
    air_density: float = DEFAULT_AIR_DENSITY,
    viscosity: float = DEFAULT_VISCOSITY,
    integrator: Callable | None = None,
) -> float:
    """Compute one physical Cd from surface pressure and velocity.

    ``data_root`` contains ``param{fold}/{obj_name}/quadpress_smpl.vtk``.
    Inputs must follow the VTK surface point order and be in physical units.
    ``integrator`` is an optional test/deployment callback with the same
    arguments; it avoids importing VTK when a caller supplies another backend.
    """
    if integrator is not None:
        return float(integrator(obj_name, press_surf, velo_surf, data_root, fold=fold))
    vtk, vtk_to_numpy = _require_vtk()
    pressure = np.asarray(press_surf, dtype=np.float64)
    velocity = np.asarray(velo_surf, dtype=np.float64)
    if pressure.ndim == 2 and pressure.shape[1] == 1:
        pressure = pressure[:, 0]
    if pressure.ndim != 1 or velocity.ndim != 2 or velocity.shape[1] != 3:
        raise ValueError("press_surf must be (N,) or (N,1), and velo_surf must be (N,3)")
    if not np.isfinite(pressure).all() or not np.isfinite(velocity).all():
        raise ValueError("surface pressure or velocity contains NaN or Inf")
    if speed <= 0 or air_density <= 0 or viscosity < 0:
        raise ValueError("speed and air_density must be positive; viscosity cannot be negative")

    mesh_path = Path(data_root).expanduser().resolve() / f"param{fold}" / str(obj_name) / "quadpress_smpl.vtk"
    grid = _read_grid(vtk, mesh_path)
    normals, poly = _cell_normals(vtk, vtk_to_numpy, grid)
    points = vtk_to_numpy(grid.GetPoints().GetData())
    if len(points) != len(pressure) or len(points) != len(velocity):
        raise ValueError(
            f"surface fields must match VTK points ({len(points)}), got pressure={len(pressure)}, velocity={len(velocity)}"
        )
    from scipy.spatial import ConvexHull

    frontal_area = float(ConvexHull(points[:, :2]).volume)
    if frontal_area <= 0 or not np.isfinite(frontal_area):
        raise ValueError("surface frontal area is not positive")
    areas = _cell_areas(vtk, poly)
    gradient = _velocity_gradient(vtk, grid, velocity)
    if len(normals) != len(areas) or len(gradient) != len(areas):
        raise ValueError("VTK surface arrays have inconsistent cell counts")

    pressure_array = vtk.vtkDoubleArray()
    pressure_array.SetNumberOfComponents(1)
    pressure_array.SetNumberOfTuples(len(pressure))
    pressure_array.SetName("my_press")
    for index, value in enumerate(pressure):
        pressure_array.SetTuple1(index, float(value))
    grid.GetPointData().AddArray(pressure_array)
    point_to_cell = vtk.vtkPointDataToCellData()
    point_to_cell.SetInputData(grid)
    point_to_cell.Update()
    pressure_cell_array = point_to_cell.GetOutput().GetCellData().GetArray("my_press")
    if pressure_cell_array is None:
        raise ValueError("VTK failed to interpolate pressure from points to cells")
    pressure_cell = vtk_to_numpy(pressure_cell_array).reshape(-1)
    if len(pressure_cell) != len(areas):
        raise ValueError("VTK pressure cell count does not match surface cells")

    normal_z = normals[:, -1]
    pressure_force = -np.dot(pressure_cell * areas, normal_z)
    wall_shear = -np.dot(gradient[:, -1] * areas, normal_z) * viscosity
    drag = float(pressure_force + wall_shear)
    return float(2.0 * drag / (speed * speed * frontal_area * air_density))


cal_cd = compute_cd


@dataclass
class CarCdGeometry:
    """Cached linear form of the audited pressure + wall-shear VTK metric.

    Only geometry is cached here. Ground-truth fields are supplied separately
    by evaluation, and are never available to acquisition through this class.
    """
    coordinates: np.ndarray
    pressure_coefficients: np.ndarray
    velocity_z_coefficients: np.ndarray
    pressure_influence: np.ndarray
    frontal_area: float
    mesh_sha256: str

    @classmethod
    def load(cls, mesh_path, cache_dir=None):
        from adasolver.models.checkpoint import sha256_file
        mesh_path = Path(mesh_path)
        digest = sha256_file(mesh_path)
        cache = Path(cache_dir) / f"cd_geometry_v1_{digest}.npz" if cache_dir else None
        if cache is not None and cache.exists():
            with np.load(cache) as data:
                return cls(data["coordinates"], data["pressure_coefficients"],
                           data["velocity_z_coefficients"], data["pressure_influence"],
                           float(data["frontal_area"]), digest)
        vtk, vtk_to_numpy = _require_vtk()
        grid = _read_grid(vtk, mesh_path)
        normals, poly = _cell_normals(vtk, vtk_to_numpy, grid)
        areas = _cell_areas(vtk, poly)
        coords = vtk_to_numpy(grid.GetPoints().GetData()).astype(np.float64)
        vertices = []
        for index in range(grid.GetNumberOfCells()):
            cell = grid.GetCell(index)
            if cell.GetNumberOfPoints() != 4:
                raise ValueError("Cd requires quadrilateral cells")
            vertices.append([cell.GetPointId(i) for i in range(4)])
        vertices = np.asarray(vertices, dtype=np.int64)
        if len(vertices) != len(areas) or len(normals) != len(areas):
            raise ValueError("Cd surface cells do not align")
        p1, p2, p3, p4 = (coords[vertices[:, i]] for i in range(4))
        den = np.stack((np.linalg.norm(p2-p1+p3-p4, axis=1),
                        np.linalg.norm(p3-p1+p4-p2, axis=1),
                        np.linalg.norm(p4-p1+p2-p3, axis=1)), axis=1) + 1e-8
        derivative = (np.array([-1., 1., 1., -1.])[None] / den[:, 0, None] +
                      np.array([-1., -1., 1., 1.])[None] / den[:, 1, None] +
                      np.array([-1., 1., -1., 1.])[None] / den[:, 2, None])
        normal_area = areas * normals[:, -1]
        influence = np.zeros(len(coords))
        velocity = np.zeros(len(coords))
        np.add.at(influence, vertices.ravel(), np.repeat(normal_area / 4, 4))
        np.add.at(velocity, vertices.ravel(), (-DEFAULT_VISCOSITY * normal_area[:, None] * derivative).ravel())
        from scipy.spatial import ConvexHull
        frontal = float(ConvexHull(coords[:, :2]).volume)
        factor = 2. / (DEFAULT_SPEED ** 2 * DEFAULT_AIR_DENSITY * frontal)
        result = cls(coords, -influence * factor, velocity * factor, influence, frontal, digest)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_name(f".{cache.stem}.{os.getpid()}.npz")
            np.savez(temporary, coordinates=coords, pressure_coefficients=result.pressure_coefficients,
                     velocity_z_coefficients=result.velocity_z_coefficients,
                     pressure_influence=influence, frontal_area=frontal)
            os.replace(temporary, cache)
        return result

    def validate_surface(self, surface_coords, atol=1e-6):
        coords = np.asarray(surface_coords)
        if coords.shape != self.coordinates.shape or not np.allclose(coords, self.coordinates, atol=atol, rtol=0):
            raise ValueError("adapter surface and VTK point order/coordinates do not match")

    def compute(self, pressure, velocity):
        pressure, velocity = np.asarray(pressure), np.asarray(velocity)
        if pressure.shape != (len(self.coordinates),) or velocity.shape != self.coordinates.shape:
            raise ValueError("Cd field shape does not match surface geometry")
        if not np.isfinite(pressure).all() or not np.isfinite(velocity).all():
            raise ValueError("nonfinite Cd fields")
        return float(self.pressure_coefficients @ pressure + self.velocity_z_coefficients @ velocity[:, 2])
