#!/usr/bin/env python3
"""Build the manuscript and publish paper.pdf without regenerating figures or tables."""
from pathlib import Path
import argparse,json,re,shutil,subprocess,sys

ROOT=Path(__file__).resolve().parents[1]
NAME='iclr2027_conference'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args()
    build=ROOT/'.build';build.mkdir(exist_ok=True)
    cmd=['latexmk','-pdf','-synctex=1','-interaction=nonstopmode','-halt-on-error',
         '-file-line-error','-outdir=.build',NAME+'.tex']
    if args.force:cmd.insert(1,'-g')
    result=subprocess.run(cmd,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (build/'build.log').write_text(result.stdout)
    if result.returncode:
        print(result.stdout[-10000:],file=sys.stderr)
        raise SystemExit(result.returncode)
    log=(build/(NAME+'.log')).read_text(errors='replace')
    unresolved=[s for s in log.splitlines() if
        ('undefined' in s.lower() and any(w in s for w in ['Reference','Citation','There were']))
        or 'multiply defined' in s]
    if unresolved:raise RuntimeError('\n'.join(unresolved))
    for name in ['paper.pdf',NAME+'.pdf']:shutil.copy2(build/(NAME+'.pdf'),ROOT/name)
    info=subprocess.check_output(['pdfinfo',str(ROOT/'paper.pdf')],text=True)
    pages=int(re.search(r'Pages:\s+(\d+)',info).group(1))
    subprocess.run(['pdftotext','-layout',str(ROOT/'paper.pdf'),str(build/'paper.txt')],check=True)
    qa=dict(entrypoint=NAME+'.tex',pdf='paper.pdf',pages=pages,undefined_references=unresolved,
            overfull_boxes=[s for s in log.splitlines() if s.startswith('Overfull')])
    (build/'build_qa.json').write_text(json.dumps(qa,indent=2)+'\n')
    print(f"Built paper.pdf: {pages} pages, no undefined references; {len(qa['overfull_boxes'])} overfull boxes")


if __name__=='__main__':main()
