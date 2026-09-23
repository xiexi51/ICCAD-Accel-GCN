"""Stream the official archive, extracting only missing original CSR files."""
import argparse
import re
import shutil
import tarfile
import urllib.request
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('graph')
args=parser.parse_args()
if not re.fullmatch(r'[A-Za-z0-9_.-]+',args.graph):
    parser.error('graph must be a basename')
dest=Path(__file__).resolve().parents[1]/'graphs'
dest.mkdir(exist_ok=True)
wanted={args.graph+'.'+suffix for suffix in ('config','graph.ptrdump','graph.edgedump')}
wanted={name for name in wanted if not (dest/name).exists()}
if wanted:
    url='https://drive.usercontent.google.com/download?id=1_sE65oveGpzRdCcExBmUaNG982lUB-Cx&export=download&confirm=t'
    with urllib.request.urlopen(url,timeout=120) as response:
        with tarfile.open(fileobj=response,mode='r|gz') as archive:
            for entry in archive:
                name=Path(entry.name).name
                if name not in wanted or not entry.isfile():
                    continue
                tmp=dest/(name+'.partial')
                with archive.extractfile(entry) as source,tmp.open('wb') as target:
                    shutil.copyfileobj(source,target)
                tmp.replace(dest/name)
                wanted.remove(name)
                print('Extracted',name,entry.size,'bytes',flush=True)
                if not wanted:
                    break
    if wanted:
        raise RuntimeError(f'Missing files: {sorted(wanted)}')
