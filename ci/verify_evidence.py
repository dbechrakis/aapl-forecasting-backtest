"""Validate committed evidence without pretending to retrain external-data models."""
from pathlib import Path
import ast, csv, json, math
ROOT=Path(__file__).resolve().parents[1]
for p in ROOT.rglob('*.py'):
    if not any(x in p.parts for x in ['.git','.venv','node_modules']): ast.parse(p.read_text())
for p in ROOT.rglob('*.ipynb'):
    n=json.loads(p.read_text()); assert n['nbformat']==4
    for c in n['cells']:
        assert c['cell_type'] in ['code','markdown','raw']
        assert not any(o.get('output_type')=='error' for o in c.get('outputs',[])), str(p)
def table(path):
    with (ROOT/path).open(newline='',encoding='latin1' if str(path).startswith('data/') else 'utf-8') as f:return list(csv.DictReader(f))
def close(a,b,tol=1e-6): assert math.isclose(float(a),float(b),abs_tol=tol,rel_tol=tol),(a,b)

# Backtest evidence is optional until the first refresh run; when present it must be consistent.
manifest_path=ROOT/'data/manifest.json'
if manifest_path.exists():
    import hashlib
    m=json.loads(manifest_path.read_text())
    assert hashlib.sha256((ROOT/m['path']).read_bytes()).hexdigest()==m['sha256'],'snapshot fingerprint'
    run=json.loads((ROOT/'outputs/backtest/run_manifest.json').read_text())
    assert run['data_sha256']==m['sha256'],'backtest ran on a different snapshot'
    acc=table('outputs/backtest/accuracy.csv')
    for r in acc:
        if r['model']=='Random walk': close(r['mae_vs_rw'],1); close(r['rmse_vs_rw'],1)
        assert float(r['mae'])>0 and 0<=float(r['always_up_hit_rate'])<=1
    for r in table('outputs/backtest/interval_calibration.csv'):
        assert 0<=float(r['coverage'])<=1 and float(r['mean_width'])>0
live=ROOT/'outputs/live/forecast_log.csv'
if live.exists():
    for r in table('outputs/live/forecast_log.csv'):
        assert float(r['lower_return'])<0<float(r['upper_return'])
        assert r['inside'] in ('','0.0','1.0','0','1')

print('Committed evidence and syntax checks passed; see VALIDATION.md for rerun scope.')
