"""Recompute the DreamPoly notebook and publish its chart data to Grafana."""
import argparse
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
NOTEBOOK = PROJECT / "notebooks" / "DreamPoly.ipynb"
SNAPSHOT = HERE / "data" / "charts.json"


def run_notebook():
    import nbformat
    from nbclient import NotebookClient

    work = HERE / ".work"
    work.mkdir(exist_ok=True)
    matplotlib_cache = work / "matplotlib"
    matplotlib_cache.mkdir(exist_ok=True)
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    paths = [str(PROJECT / "src"), str(HERE), str(work / "python")]
    setup = (
        "import os, sys\n"
        f"sys.path[:0] = {paths!r}\n"
        "os.environ['MPLBACKEND'] = 'Agg'\n"
        f"os.environ['MPLCONFIGDIR'] = {str(matplotlib_cache)!r}"
    )
    notebook.cells.insert(0, nbformat.v4.new_code_cell(setup))
    notebook.cells.append(nbformat.v4.new_code_cell(
        f"from export_notebook import export\nexport(globals(), {str(SNAPSHOT)!r})"
    ))

    def progress(cell, cell_index, **_):
        if cell.cell_type == "code":
            first_line = cell.source.splitlines()[0]
            print(f"  {cell_index:02d}  {first_line[:88]}", flush=True)

    with TemporaryDirectory(prefix="run-", dir=work) as scratch:
        notebooks = Path(scratch) / "notebooks"
        raw = notebooks / "data" / "raw"
        raw.mkdir(parents=True)
        shutil.copyfile(NOTEBOOK, notebooks / NOTEBOOK.name)
        archive = "dreambank.tar.xz"
        for candidate in (NOTEBOOK.parent / "data" / "raw" / archive, work / archive):
            if candidate.is_file():
                shutil.copyfile(candidate, raw / archive)
                break
        client = NotebookClient(
            notebook, kernel_name="python3", timeout=900, on_cell_start=progress,
            resources={"metadata": {"path": scratch}},
        )
        kernel = client.create_kernel_manager()
        kernel.kernel_spec.argv[0] = sys.executable
        # Keep the notebook's statistical routines reproducible and avoid BLAS oversubscription.
        env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        client.execute(env=env)
        if not (work / archive).exists():
            shutil.copyfile(raw / archive, work / archive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--publish-only", action="store_true", help="Upload the existing data/charts.json")
    mode.add_argument("--export-only", action="store_true", help="Run the notebook and save data without Grafana")
    args = parser.parse_args()

    if not args.publish_only:
        run_notebook()
        print(f"Экспорт данных: {SNAPSHOT}", flush=True)
    if not args.export_only:
        from dashboard import publish
        print(f"Grafana: {publish(SNAPSHOT)}", flush=True)


if __name__ == "__main__":
    main()
