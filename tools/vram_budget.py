"""VRAM arena budget for 16 GB cards: transcribes compute_total_vram_bytes (src/weights.zig)
and compute_runtime_vram_bytes (src/engine.zig). Excludes the CUDA context. `python tools/vram_budget.py`"""
G=2**30; H=2304;QR=6552;KV=1024;Q=3072;VU=200058;VP=201024;I=1152
A=lambda n:(n+255)//256*256
def lin(r,c,k): return {'bf16':A(r*c*2),'fp8':A(r*c)+A(r*4),'nvfp4':A(r*(c//2))+A(r*(c//16))}[k]
glob=lambda l:(l+1)%6==0
def weights(mode,host,mtp,dv):
    d={'exact':'bf16','fp8':'fp8','nvfp4':'nvfp4'}[mode]; m='bf16' if mode=='exact' else 'nvfp4'
    t=0 if host else A(VP*H*2)
    t+=2*A(H*2)+lin(VU,H,d)
    t+=sum(A((3*H if glob(l) else H)*4) for l in range(1,24))+24*A(H*4)+A(H*4)+2*A(24*H*4)
    for l in range(24):
        g=glob(l); t+=(3 if g else 1)*A(H*2)+A(H*2)
        t+=lin(QR,H,d)+lin(2*KV,H,d)+lin(H,Q,d)
        t+=2*A(4*KV*2)+(2*A(8*4*H*2) if g else 0)+2*A(H*4*2)+A(24*128*2)+A(24*16*2)+A(16*1024*2)
        if 6<=l<18:
            t+=4*(lin(64,2*H,'nvfp4')+2*lin(H,64,'nvfp4')) if mode=='nvfp4' else 4*(A(64*2*H*2)+2*A(H*64*2))
        t+=lin(128,H,'nvfp4') if mode=='nvfp4' else A(128*H*2)
        t+=A(128*4)
        for r,c in((2*I,H),(H,I)):
            t+=(A(128*r*(c//2))+A(128*r*(c//16))+A(512)) if (l<20 or mode=='nvfp4') else A(128*r*c)+A(128*r*(c//32))
    if mtp:
        n=dv or VU
        t+=A(64)+3*A(H*2)+A(VP*256*2)+A(768*2)+A(H*256*2)+A(H*2)+A(768*512*2)+2*A(768*H*2)
        if dv: t+=A(dv*4)+A(VU*4)+lin(dv,H,m)
        t+=A(n*4)+lin(n,256,m)
        t+=3*(3*A(H*2)+3*A(256)+lin(5120,H,m)+lin(H,Q,m)+lin(18432,H,m)+lin(H,9216,m)+lin(2*KV,H,m))
    return t
def kv(c,f): return {'bf16':2*A(c*KV*2),'fp8':2*A(c*KV)+2*A(c*16),'q4':2*A(c*KV//2)+2*A(c*32)}[f]
def runtime(ctx,f,mtp,pb):
    t=0
    for v in range(48):
        l=v if v<6 else (6+(v-6)%12 if v<42 else v-24)
        g=glob(l); t+=kv(ctx if g else 1024+max(512,pb//4),f if g else 'bf16')+2*A(16*KV*4)+2*A(16*H*4)
    if mtp: t+=2*kv(1032,'bf16')+kv(65536 if ctx>524288 else ctx,f)+A(16*H*4)
    t+=A(2050000*4); b=8
    t+=b*H*4*(3+4+2+3+4+3+6+1+8+4+3)+b*H*2*5+b*QR*4+2*b*KV*4+b*Q*6+3*b*64*4+b*128*4+b*I*8*2+b*2*KV*4+12800000+2*b*4096+(b+7)*VU*4+2*b*H*4+16384*4
    t+=pb*H*4*(3+4+2+3+4+8+4+1)+pb*H*2*(3+1+2)+pb*QR*4+2*pb*KV*4+pb*Q*2+pb*128*4+pb*8*4*5+pb*8*I*2+QR*H*2+pb*64*6+pb*H*4
    return t
LAYER=128*(2*I*H+H*I)*9//16   # one NVFP4 layer's experts
if __name__=='__main__':
    # self-check against the README/LEARNINGS published figures
    for args,want in((('exact',0,1,0),18.28),(('fp8',0,1,32768),16.85),(('nvfp4',0,1,32768),14.57),(('nvfp4',1,1,32768),13.70)):
        assert round(weights(*args)/G,2)==want,args
    w=weights('nvfp4',1,1,32768)
    # runtime arena measured on the 5070 Ti after the upstream 124afb8 merge
    assert round(runtime(32768,'q4',1,512)/G,2)==0.80 and round(runtime(131072,'q4',1,1024)/G,2)==1.87
    print('nvfp4 + q4 KV + host embed + MTP, arena GiB')
    print('load peak (weights + staging):',round((w+VU*H*2)/G,2))
    print('prefill     ctx  off=0  off=1  off=2')
    for pb in(2048,512):
        for c in(8192,32768,65536):
            print(f'{pb:7d} {c:7d}',*(f'{(w+runtime(c,"q4",1,pb)-n*LAYER)/G:6.2f}' for n in(0,1,2)))
