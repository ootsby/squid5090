"""Model router for squid5090: loads, unloads and swaps backends on demand, speaking the OpenAI API and the
llama.cpp router API (so pi's /login llama.cpp and /llama work). Stdlib only; localhost only.

    python router.py [--port 8000] [--config router.json]

A model is a preset from router.json, optionally with options: "inkling-2-mini:abl,ctx=65536".
Options (squid presets): ctx=N  mode=mix|f8|f18|nvfp4  kv=q4|fp8|bf16  mtp=0..7  batch=512|1024 (prefill)  abl | abl=0  vision | vision=0 (image input, +0.6 GiB)

GET  /models /v1/models /health /props
POST /models/load {"model"}   /models/unload {"model", "force": true to cut off a running reply}
POST /v1/chat/completions /v1/completions   (loads or swaps to "model" first)
"""
import argparse, ctypes, http.client, json, os, subprocess, sys, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'tools'))
import vram_budget

CONTAINER, CTX_MAX, BODY_MAX = 'squid-router', 131072, 32 << 20
DEFAULTS = {'ctx': 32768, 'mode': 'mix', 'kv': 'q4', 'mtp': 7, 'batch': 512, 'abl': False, 'vision': False}
MIXF = {'f8': '8', 'f18': '18'}  # SQUID_IQ3 value: layers with 3-bit gate/up experts, plus an FP8 lm_head


class Bad(Exception):
    def __init__(self, code, msg): super().__init__(msg); self.code = code


class Router:
    def __init__(self, cfg):
        self.cfg, self.presets = cfg, cfg['models']
        self.cv = threading.Condition()
        self.status, self.name, self.spec, self.proc = 'unloaded', None, None, None
        self.active, self.cancel, self.last_used, self.failed, self.load_id = 0, False, 0.0, {}, 0
        self.closing = False

    # ---- model names -> specs
    def resolve(self, model):
        base, _, opts = (model or self.name or self.cfg['default']).partition(':')
        if base not in self.presets: raise Bad(404, f'unknown model {base!r}; presets: {", ".join(self.presets)}')
        spec = dict(self.presets[base])
        if 'cmd' in spec:
            if opts: raise Bad(400, f'{base} takes no options')
            return spec
        spec = {**DEFAULTS, **spec}
        for o in filter(None, opts.split(',')):
            k, eq, v = o.partition('=')
            if k not in DEFAULTS: raise Bad(400, f'unknown option {k!r}; use {", ".join(DEFAULTS)}')
            if k in ('abl', 'vision'): spec[k] = v not in ('0', 'off', 'false') if eq else True
            elif k in ('ctx', 'mtp', 'batch'):
                if not v.isdecimal(): raise Bad(400, f'{k} must be a number')
                spec[k] = int(v)
            else: spec[k] = v
        if not 1024 <= spec['ctx'] <= CTX_MAX or spec['ctx'] % 1024: raise Bad(400, f'ctx must be a multiple of 1024, up to {CTX_MAX}')
        if spec['mode'] not in ('mix', 'nvfp4', *MIXF): raise Bad(400, f'mode must be mix, nvfp4, {", ".join(MIXF)}')
        if spec['batch'] not in (512, 1024): raise Bad(400, 'batch must be 512 or 1024')
        if spec['kv'] not in ('q4', 'fp8', 'bf16'): raise Bad(400, 'kv must be q4, fp8 or bf16')
        if not 0 <= spec['mtp'] <= 7: raise Bad(400, 'mtp must be 0..7')
        gib, cap = self.arena_gib(spec), self.cfg.get('max_arena_gib', 14.5)
        if gib > cap: raise Bad(400, f'needs about {gib:.2f} GiB of VRAM arenas, over the {cap} GiB limit (router.json max_arena_gib)')
        return spec

    def arena_gib(self, s):  # measured weights with --embed-host and the drafter
        mix = 12.76 if (Path(self.cfg['model_dir']) / 'iq3.safetensors').exists() else 14.13  # B: D's measured drafter growth added
        w = {'mix': mix, 'f8': 13.45, 'f18': 12.31, 'nvfp4': 13.81}[s['mode']]
        return w + 0.60 * s['vision'] + vram_budget.runtime(s['ctx'], s['kv'], 1, s['batch']) / 2 ** 30

    # ---- backend processes
    def command(self, s):
        if 'cmd' in s: return s['cmd']
        c = ['wslc', 'run', '--rm', '--name', CONTAINER, '--gpus', 'all', '--ulimit', 'memlock=-1',
             '-p', f'127.0.0.1:{self.port(s)}:8000', '-v', f'{self.cfg.get("volume") or self.cfg["model_dir"]}:/model:ro',
             *(['-e', 'SQUID_IQ3=' + MIXF[s['mode']]] if s['mode'] in MIXF else []), self.cfg['image'].format(batch=s['batch']),
             'serve', '/model', '--host', '0.0.0.0', '--port', '8000', '--mode', 'mix' if s['mode'] in MIXF else s['mode'], '--kv', s['kv'],
             '--ctx', str(s['ctx']), '--mtp', str(s['mtp']), '--embed-host', *([] if s['vision'] else ['--no-vision'])]
        return c + (['--ablation', '/model/' + self.cfg['ablation_file']] if s['abl'] else [])

    def port(self, s): return s.get('port', self.cfg.get('backend_port', 8001))

    def alive(self): return self.proc is not None and self.proc.poll() is None

    def stop_backend(self):
        if self.proc is None: return
        for _ in range(10):  # killing the wslc client would leave its container running, so keep stopping the container
            if not self.alive(): break
            if 'cmd' in (self.spec or {}): self.proc.terminate()
            else: wslc('stop', '-t', '2', CONTAINER)  # squid ignores SIGTERM until it has loaded
            try: self.proc.wait(2)
            except subprocess.TimeoutExpired: pass
        else: self.proc.kill()
        self.proc = None

    def start_load(self, name, spec):  # cv held, nothing active
        self.load_id += 1
        self.status, self.cancel = 'loading', False
        self.failed.pop(name, None)
        threading.Thread(target=self.run_load, args=(name, spec), daemon=True).start()

    def run_load(self, name, spec):
        err, t0 = None, time.time()
        try:
            self.stop_backend()
            self.name, self.spec = name, spec
            if 'cmd' not in spec and any(l.split()[-1:] == ['squid'] for l in wslc('container', 'ls').splitlines()):
                raise RuntimeError('squid.bat is already running; close it first')
            log(f'loading {name}')
            with open(HERE / 'router-backend.log', 'wb') as f:
                self.proc = subprocess.Popen(self.command(spec), stdin=subprocess.DEVNULL, stdout=f, stderr=f, creationflags=NOWIN)
            url = f'http://127.0.0.1:{self.port(spec)}{spec.get("health", "/health")}'
            while True:
                if self.cancel: err = 'cancelled'; break
                if not self.alive(): err = 'backend exited: ' + self.log_tail(); break
                if time.time() - t0 > self.cfg.get('load_timeout', 180): err = 'load timed out'; break
                try:
                    with urllib.request.urlopen(url, timeout=2) as r:
                        if r.status == 200: break
                except OSError: pass
                time.sleep(0.5)
        except Exception as e: err = str(e)
        with self.cv:
            if err is None and self.cancel: err = 'cancelled'
            if err:
                self.stop_backend()
                if err != 'cancelled': self.failed[name] = err
                self.status, self.name, self.spec = 'unloaded', None, None
            else: self.status, self.last_used = 'loaded', time.time()
            self.error = err
            self.cv.notify_all()
        log(f'{name}: {err or f"ready in {time.time() - t0:.0f} s"}')

    def log_tail(self):
        try: return (HERE / 'router-backend.log').read_text(errors='replace').strip()[-400:]
        except OSError: return '(no log)'

    def check_alive(self):  # cv held
        if self.status == 'loaded' and not self.alive():
            log(f'{self.name} exited: {self.log_tail()}')
            self.status, self.name, self.spec, self.proc = 'unloaded', None, None, None

    # ---- operations
    def acquire(self, model):  # one request at a time: squid is serial. ponytail: no parallel slots for cmd backends
        mine = None
        with self.cv:
            while True:
                self.check_alive()
                if self.closing: raise Bad(503, 'router is shutting down')
                if mine and self.status == 'unloaded' and self.load_id == mine: raise Bad(502, f'load failed: {self.error}')
                spec = self.resolve(model)
                if self.status != 'loading' and not self.active:
                    if self.status == 'loaded' and self.spec == spec:
                        self.active = 1
                        return self.port(spec)
                    self.start_load(model or self.name or self.cfg['default'], spec)
                    mine = self.load_id
                self.cv.wait(1)

    def release(self):
        with self.cv:
            self.active, self.last_used = 0, time.time()
            self.cv.notify_all()

    def load(self, model):
        with self.cv:
            self.check_alive()
            spec = self.resolve(model)
            if self.status == 'loaded' and self.spec == spec: return
            if self.status == 'loading': raise Bad(409, 'another load is in progress')
            if self.active: raise Bad(409, 'busy: a reply is being generated')
            self.start_load(model or self.cfg['default'], spec)

    def unload(self, model=None, force=False):
        with self.cv:
            self.check_alive()
            if self.status == 'unloaded': return
            if model and self.status == 'loaded' and self.resolve(model) != self.spec: raise Bad(400, f'{model} is not loaded')
            if self.status == 'loading': self.cancel = True; return
            if self.active and not force: raise Bad(409, 'busy: a reply is being generated (send "force": true to cut it off)')
            log(f'unloading {self.name}')
            self.stop_backend()
            self.status, self.name, self.spec = 'unloaded', None, None
            self.cv.notify_all()

    def idle_loop(self):
        while True:
            time.sleep(2)
            ttl = self.cfg.get('idle_unload_seconds', 900)
            with self.cv:
                idle = self.status == 'loaded' and not self.active and time.time() - self.last_used > ttl
            if ttl and idle: self.unload()

    def catalog(self):
        with self.cv:
            self.check_alive()
            names = list(self.presets)
            if self.name and self.name not in names: names.append(self.name)
            out = []
            for n in names:
                base = self.presets[n.partition(':')[0]]
                ctx = (self.spec if n == self.name and self.spec else {**DEFAULTS, **base}).get('ctx')
                st = {'value': self.status if n == self.name else 'unloaded', 'args': ['--ctx-size', str(ctx)] if ctx else []}
                if n in self.failed: st.update(failed=True, message=self.failed[n])
                out.append({'id': n, 'object': 'model', 'owned_by': 'squid-router', 'source': 'preset', 'status': st,
                            'max_model_len': ctx, 'meta': {'n_ctx': ctx}})
            return {'object': 'list', 'data': out}

    def shutdown(self):
        self.closing = self.cancel = True
        try: self.stop_backend()
        except Exception: pass


NOWIN = getattr(subprocess, 'CREATE_NO_WINDOW', 0) | getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)


def wslc(*a): return subprocess.run(['wslc', *a], capture_output=True, text=True, errors='replace', creationflags=NOWIN).stdout


def log(m): print(time.strftime('%H:%M:%S'), m, flush=True)


def make_handler(R):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def reply(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            for k, v in (('Content-Type', 'application/json'), ('Content-Length', str(len(b))), ('Access-Control-Allow-Origin', '*')): self.send_header(k, v)
            self.end_headers(); self.wfile.write(b)

        def guard(self, control=False):  # DNS rebinding; and web pages must not load or unload models
            local = lambda h: h.split('//')[-1].rsplit(':', 1)[0] in ('127.0.0.1', 'localhost')
            if not local(self.headers.get('Host', '')): raise Bad(403, 'local requests only')
            if control and not local(self.headers.get('Origin', 'http://localhost')): raise Bad(403, 'not allowed from a web page')

        def body(self):
            n = int(self.headers.get('Content-Length') or 0)
            if not 0 <= n <= BODY_MAX: raise Bad(413, 'body too large')
            try: d = json.loads(self.rfile.read(n) or b'{}')
            except ValueError: raise Bad(400, 'invalid JSON')
            if not isinstance(d, dict): raise Bad(400, 'JSON object expected')
            if not isinstance(d.get('model', ''), str): raise Bad(400, 'model must be a string')
            return d

        def run(self, f):
            try: f()
            except Bad as e: self.reply(e.code, {'error': {'message': str(e), 'type': 'router_error', 'code': e.code}})
            except (ConnectionError, BrokenPipeError): pass

        def do_OPTIONS(self):
            self.send_response(204)
            for k, v in (('Access-Control-Allow-Origin', '*'), ('Access-Control-Allow-Headers', 'Authorization, Content-Type'),
                         ('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')): self.send_header(k, v)
            self.end_headers()

        def do_GET(self): self.run(self.get)

        def get(self):
            self.guard()
            p = self.path.partition('?')[0]
            if p in ('/models', '/v1/models'): self.reply(200, R.catalog())
            elif p == '/health': self.reply(200, {'status': 'ok', 'model': R.name, 'state': R.status})
            # pi reads chat_template only to see whether thinking can be switched
            elif p == '/props': self.reply(200, {'models_autoload': True, 'chat_template': '{# enable_thinking #}'})
            else: raise Bad(404, 'unknown endpoint')

        def do_POST(self): self.run(self.post)

        def post(self):
            p = self.path.partition('?')[0]
            self.guard(control=p.startswith('/models'))
            d = self.body()
            if p == '/models/load': R.load(d.get('model')); self.reply(200, {'success': True})
            elif p == '/models/unload': R.unload(d.get('model'), d.get('force') is True); self.reply(200, {'success': True})
            elif p.removeprefix('/v1') in ('/chat/completions', '/completions'): self.proxy(p, d)
            else: raise Bad(404, 'unknown endpoint')

        def proxy(self, path, d):
            kw = d.pop('chat_template_kwargs', None)  # squid wants these at the top level
            if isinstance(kw, dict) and 'enable_thinking' in kw:
                d['enable_thinking'] = bool(kw['enable_thinking'])
                if d['enable_thinking']: d.setdefault('reasoning_effort', 'medium')
            port = R.acquire(d.get('model'))
            c, started = http.client.HTTPConnection('127.0.0.1', port, timeout=600), False
            try:
                c.request('POST', path, json.dumps(d).encode(), {'Content-Type': 'application/json'})
                r = c.getresponse()
                started = True
                self.send_response(r.status)
                self.send_header('Content-Type', r.getheader('Content-Type', 'application/json'))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                while chunk := r.read1(65536): self.wfile.write(chunk)
            except (OSError, http.client.HTTPException) as e:
                if not started: raise Bad(502, f'backend error: {e}')
            finally:
                c.close(); R.release()
    return H


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False  # on Windows, reuse would let two servers share the port


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--config', default=str(HERE / 'router.json'))
    a = ap.parse_args()
    R = Router(json.loads(Path(a.config).read_text()))
    for n in R.presets: R.resolve(n)  # fail at startup on a bad preset
    S = Server(('127.0.0.1', a.port), make_handler(R))
    if any('cmd' not in p for p in R.presets.values()):  # left over if a previous router was killed
        wslc('rm', '-f', CONTAINER)
    threading.Thread(target=R.idle_loop, daemon=True).start()
    if os.name == 'nt':  # closing the window: stop the backend within the few seconds Windows allows
        cb = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)(lambda ev: (R.shutdown(), 1)[1] if ev in (2, 5, 6) else 0)
        ctypes.windll.kernel32.SetConsoleCtrlHandler(cb, 1)
    log(f'router on http://127.0.0.1:{a.port}/v1  presets: {", ".join(R.presets)}')
    try: S.serve_forever()
    except KeyboardInterrupt: log('stopping')
    finally: R.shutdown()


if __name__ == '__main__': main()
