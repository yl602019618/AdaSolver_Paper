"""Render the unchanged plotting functions from published CSVs in a fresh folder."""
from pathlib import Path
import argparse,gzip,hashlib,importlib.util,json,os,shutil,sys,types
sys.dont_write_bytecode=True
from verify_published import verify

def render(package,output):
    package=Path(package).resolve();output=Path(output).resolve()
    if output==package or package in output.parents:raise ValueError('Use an output directory outside the source package')
    if output.exists() and any(output.iterdir()):raise FileExistsError('Output directory must be new or empty')
    verify(package);output.mkdir(parents=True,exist_ok=True)
    source_files={str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest() for p in package.rglob('*') if p.is_file()}
    for folder in ['results','figures/source_data','checkpoints']:
        for source in (package/folder).rglob('*'):
            if not source.is_file():continue
            if folder=='checkpoints' and not source.name.startswith('history'):continue
            relative=source.relative_to(package);target=output/relative
            if source.name.endswith('.jsonl.gz') or source.name.endswith('.csv.gz') and folder=='checkpoints':
                target=target.with_suffix('');payload=gzip.decompress(source.read_bytes())
            else:payload=source.read_bytes()
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
    os.environ['MPLCONFIGDIR']=str(output/'runtime/matplotlib')
    # The plot module only needs these I/O helpers from common.py. Supplying
    # them avoids importing Torch or model weights for a CSV-only rendering.
    common=types.ModuleType('common');common.OUT=output;common.CODE=package/'code'
    common.read=lambda path:json.loads(Path(path).read_text())
    common.digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def write(path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')
    common.write=write;sys.modules['common']=common
    spec=importlib.util.spec_from_file_location('matched_darcy_published_plot',package/'code/plot_results.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.export_examples=lambda:(output/'figures/source_data/qualitative_values.csv.gz',
        output/'figures/source_data/qualitative_bins.csv',output/'figures/source_data/qualitative_sources.csv')
    module.main()
    assert 'torch' not in sys.modules,'CSV-only rendering unexpectedly imported Torch'
    receipt=common.read(output/'results/figures_receipt.json')
    assert receipt['passed'] and receipt['exported_result_files']==39
    current={str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest() for p in package.rglob('*') if p.is_file()}
    assert current==source_files,'Public source package changed during figure reproduction'
    return dict(passed=True,output=str(output),figure_groups=13,files=39,
        unchanged_plot_source_sha256=common.digest(package/'code/plot_results.py'),
        source_package_modified=False,private_arrays_loaded=False,model_weights_loaded=False,torch_imported=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--package',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--receipt',type=Path);args=parser.parse_args()
    result=render(args.package,args.output)
    if args.receipt:args.receipt.parent.mkdir(parents=True,exist_ok=True);args.receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
