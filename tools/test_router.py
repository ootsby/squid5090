"""Router self-check with a stub backend (no GPU): python tools/test_router.py"""
import json, sys, threading, time, urllib.error, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import router

STUB = '''import sys,time,json
from http.server import *
class H(BaseHTTPRequestHandler):
    def log_message(s,*a): pass
    def do_GET(s): s.send_response(200); s.end_headers(); s.wfile.write(b"{}")
    def do_POST(s):
        d=json.loads(s.rfile.read(int(s.headers["Content-Length"]))); time.sleep(d.get("sleep",0))
        s.send_response(200); s.send_header("Content-Type","application/json"); s.end_headers()
        s.wfile.write(json.dumps({"tag":sys.argv[2],"got":d}).encode())
time.sleep(1); ThreadingHTTPServer(("127.0.0.1",int(sys.argv[1])),H).serve_forever()'''
cmd = lambda tag: {'cmd': [sys.executable, '-c', STUB, '8791', tag], 'ctx': 4096}
CFG = {'default': 'a', 'image': 'x', 'model_dir': '.', 'ablation_file': 'f', 'backend_port': 8791, 'idle_unload_seconds': 0, 'load_timeout': 20,
       'models': {'a': cmd('a'), 'b': cmd('b'), 'bad': {'cmd': [sys.executable, '-c', 'raise SystemExit(3)']}, 'sq': {'mode': 'nvfp4', 'ctx': 8192}}}
R = router.Router(CFG)
S = router.Server(('127.0.0.1', 8790), router.make_handler(R)); threading.Thread(target=S.serve_forever, daemon=True).start()

def call(path, d=None, **h):
    q = urllib.request.Request('http://127.0.0.1:8790' + path, None if d is None else json.dumps(d).encode(), {'Content-Type': 'application/json', **h})
    try:
        with urllib.request.urlopen(q, timeout=40) as r: return r.status, json.load(r)
    except urllib.error.HTTPError as e: return e.code, json.load(e)
chat = lambda **d: call('/v1/chat/completions', d)
state = lambda: {m['id']: m['status']['value'] for m in call('/models')[1]['data']}
def until(f, t=15):
    while not f(): time.sleep(.1); t -= .1; assert t > 0, 'timed out'

try:
    # names and options
    e = lambda m: (lambda: R.resolve(m))
    def fails(f, code):
        try: f()
        except router.Bad as x: assert x.code == code, x; return str(x)
        raise AssertionError('accepted')
    s = R.resolve('sq:abl,ctx=16384,mtp=0'); assert (s['abl'], s['ctx'], s['mtp'], s['mode']) == (True, 16384, 0, 'nvfp4')
    assert '--ablation' in R.command(s) and '--ablation' not in R.command(R.resolve('sq'))
    assert R.resolve('sq:abl=0') == R.resolve('sq')
    for m in ('sq:ctx=1000', 'sq:ctx=x', 'sq:ctx=-1024', 'sq:mode=exact', 'sq:kv=q2', 'sq:mtp=9', 'sq:foo=1', 'sq:image=evil', 'a:abl'): fails(e(m), 400)
    assert 'GiB' in fails(e('sq:ctx=131072'), 400); fails(e('nope'), 404)
    # load, swap, reuse
    assert set(state().values()) == {'unloaded'}
    assert chat(model='a')[1]['tag'] == 'a' and state()['a'] == 'loaded'
    n = R.load_id; assert chat(model='a')[1]['tag'] == 'a' and chat()[1]['tag'] == 'a' and R.load_id == n
    assert chat(model='b')[1]['tag'] == 'b' and state() == {'a': 'unloaded', 'b': 'loaded', 'bad': 'unloaded', 'sq': 'unloaded'}
    g = chat(model='b', chat_template_kwargs={'enable_thinking': True})[1]['got']; assert g['enable_thinking'] is True and g['reasoning_effort'] == 'medium' and 'chat_template_kwargs' not in g
    # busy: unload refused, a swap waits for the running reply, force cuts it off
    out = []; t = threading.Thread(target=lambda: out.append(chat(model='b', sleep=3))); t.start(); until(lambda: R.active)
    assert call('/models/unload', {'model': 'b'})[0] == 409 and call('/models/load', {'model': 'a'})[0] == 409
    assert chat(model='a')[1]['tag'] == 'a'; t.join(); assert out[0][1]['tag'] == 'b'
    t = threading.Thread(target=lambda: out.append(chat(model='a', sleep=8))); t.start(); until(lambda: R.active); t0 = time.time()
    assert call('/models/unload', {'model': 'a', 'force': True})[0] == 200; t.join(); assert time.time() - t0 < 5 and out[1][0] == 502
    # explicit load / cancel / failure / crash recovery
    assert call('/models/load', {'model': 'a'})[0] == 200 and state()['a'] == 'loading'
    assert call('/models/unload', {'model': 'a'})[0] == 200; until(lambda: state()['a'] == 'unloaded'); assert 'a' not in R.failed
    assert chat(model='bad')[0] == 502 and [m for m in call('/models')[1]['data'] if m['id'] == 'bad'][0]['status']['failed']
    assert chat(model='a')[0] == 200; R.proc.kill(); R.proc.wait(); assert state()['a'] == 'unloaded' and chat(model='a')[0] == 200
    # idle unload
    CFG['idle_unload_seconds'] = 1; threading.Thread(target=R.idle_loop, daemon=True).start(); until(lambda: state()['a'] == 'unloaded')
    # local only
    assert call('/models', Host='evil.example')[0] == 403 and call('/models/load', {'model': 'a'}, Origin='https://evil.example')[0] == 403
    assert call('/models/load', {'model': 5})[0] == 400 and call('/nope')[0] == 404
    try: router.Server(('127.0.0.1', 8790), None); raise AssertionError('port shared')
    except OSError: pass
    print('router self-check OK')
finally: R.shutdown()
