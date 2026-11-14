"""Interface and exact weighted-key checks; these are not test-set results."""
from common import *
import argparse,time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    report={'scope':'model API and backward correctness, not trained accuracy','device':args.device,'checks':[]}
    for arch in ['t3','linearno']:
        model=build_model(arch).to(args.device)
        for n in [17,43]:
            pos=torch.rand(2,n,2,device=args.device)
            output=model(pos,pos)
            assert output.shape==(2,n,1) and torch.isfinite(output).all()
            loss=(output-torch.rand_like(output)).square().mean();loss.backward()
            assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
            report['checks'].append(dict(architecture=arch,points=n,shape=list(output.shape),backward_finite=True,
                                         parameters=sum(p.numel() for p in model.parameters())))
            model.zero_grad(set_to_none=True)
        model.eval();pos=torch.rand(1,7,2,device=args.device)
        with torch.no_grad():
            native=model(pos,pos)
            context,patched=install_compensation(model);context.set(torch.ones(7,device=args.device))
            unit=model(pos,pos);context.clear()
            torch.testing.assert_close(unit,native,rtol=2e-5,atol=1e-6)
            if arch=='linearno':
                mass=torch.tensor([1,2,3,1,2,1,3],device=args.device,dtype=torch.float32)
                context.set(mass);weighted=model(pos,pos);context.clear()
                expanded=torch.repeat_interleave(pos,mass.to(torch.long),dim=1)
                duplicate=model(expanded,expanded)
                starts=torch.cat((mass.new_zeros(1),mass.cumsum(0)[:-1])).long()
                duplicate=duplicate[:,starts]
                torch.testing.assert_close(weighted,duplicate,rtol=2e-5,atol=1e-6)
                report['checks'].append(dict(architecture=arch,weighted_duplicate_max_abs=float((weighted-duplicate).abs().max())))
        report['checks'].append(dict(architecture=arch,unit_compensation_max_abs=float((unit-native).abs().max()),patched_layers=patched))
    conv=build_model('linearno_conv').to(args.device).eval()
    try:
        conv(torch.rand(1,43,2,device=args.device),None)
    except Exception as exc:
        report['conv_variable_points']=dict(compatible=False,error_type=type(exc).__name__,error=str(exc))
    else:raise AssertionError('Expected native fixed-grid conv to reject irregular point count')
    with torch.no_grad():
        y=conv(torch.rand(1,11271,2,device=args.device),None)
        assert y.shape==(1,11271,1) and torch.isfinite(y).all()
    report['conv_full_grid']=dict(compatible=True,shape=list(y.shape),parameters=sum(p.numel() for p in conv.parameters()))
    report['status']='passed'
    write(OUT/'results/interface_validation.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
