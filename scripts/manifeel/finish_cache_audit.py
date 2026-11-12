"""Finish an interrupted immutable-cache audit in bounded, restartable array shards."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--start',type=int,required=True)
    parser.add_argument('--stop',type=int,required=True)
    args=parser.parse_args()
    root=Path('/data/wangzihao/cache/manifeel/full_sensor_float32_9task_v1_seed42_20261017')
    manifest=json.loads((root/'manifest.json').read_text())
    prefix=json.loads((args.output/'verified_prefix.json').read_text())
    assert prefix['manifest_sha256']==hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
    for i,entry in enumerate(prefix['entries']):
        assert entry['sha256']==manifest['arrays'][i]['sha256']
        stat=(root/entry['filename']).stat()
        assert entry['identity']==[stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns]
    receipts=args.output/'hash_receipts';receipts.mkdir(exist_ok=True)
    buffer=bytearray(8*1024**2)
    for i in range(args.start,args.stop):
        record=manifest['arrays'][i];path=root/record['filename']
        before=path.stat();identity=[before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns]
        receipt=receipts/f'{i}.json'
        if receipt.exists():
            old=json.loads(receipt.read_text())
            if old['identity']==identity and old['sha256']==record['sha256']:continue
        array=np.load(path,mmap_mode='r',allow_pickle=False)
        size=array.nbytes;offset=array.offset;del array
        digest=hashlib.sha256()
        with path.open('rb',buffering=0) as stream:
            stream.seek(offset);remaining=size
            while remaining:
                count=stream.readinto(memoryview(buffer)[:min(len(buffer),remaining)])
                if not count:raise IOError('Truncated cache')
                digest.update(memoryview(buffer)[:count]);remaining-=count
        assert digest.hexdigest()==record['sha256'],record['filename']
        after=path.stat();assert identity==[after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns]
        temporary=receipt.with_suffix('.partial')
        temporary.write_text(json.dumps(dict(index=i,filename=record['filename'],identity=identity,
                                            bytes=size,sha256=digest.hexdigest()))+'\n')
        os.replace(temporary,receipt)
        print(json.dumps(dict(event='hash_verified',index=i,filename=record['filename'],bytes=size)),flush=True)
    with (args.output/'hash_finish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        n=prefix['verified_arrays']
        if all((receipts/f'{i}.json').exists() for i in range(n,len(manifest['arrays']))):
            for i in range(n,len(manifest['arrays'])):
                assert json.loads((receipts/f'{i}.json').read_text())['sha256']==manifest['arrays'][i]['sha256']
            report=json.loads((args.output/'WINDOWS_PASS.json').read_text())
            report.update(status='PASS',full_cache_hash_verification=True,
                          manifest_sha256=prefix['manifest_sha256'],
                          verified_prefix_from_interrupted_scan=n,remaining_arrays_verified=len(manifest['arrays'])-n)
            temporary=args.output/'PASS.json.partial';temporary.write_text(json.dumps(report,indent=2)+'\n')
            os.replace(temporary,args.output/'PASS.json')
            print('FULL_CACHE_AUDIT_PASS',flush=True)


if __name__=='__main__':main()
