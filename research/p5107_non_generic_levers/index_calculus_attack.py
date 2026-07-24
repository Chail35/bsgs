#!/usr/bin/env python3
"""Strict fixed-instance index-calculus experiment for E/F_5107.

Construction inputs: curve, generator, target point, coordinate factor-base rule,
and deterministic randomness.  No target scalar, orbit index, PhaseWord, atlas, or
factor-base logarithm is embedded or consulted.  Factor-base logarithms are solved
from independently generated random-multiple relations.
"""
from __future__ import annotations
import hashlib, json, math, random, signal, time
from pathlib import Path

P=5107; N=4993; B=7; G=(2,530); TARGET=(3711,748)
OUT=Path(__file__).resolve().parent/'outputs'; OUT.mkdir(parents=True,exist_ok=True)
OPS={'add':0,'dbl':0,'mul_calls':0,'inversions':0}

def inv(a): OPS['inversions']+=1; return pow(a%P,-1,P)
def neg(Q): return None if Q is None else (Q[0],(-Q[1])%P)
def add(Q,R):
    if Q is None:return R
    if R is None:return Q
    x1,y1=Q;x2,y2=R
    if x1==x2 and (y1+y2)%P==0:return None
    if Q==R:
        OPS['dbl']+=1
        if y1==0:return None
        s=3*x1*x1*inv(2*y1)%P
    else:
        OPS['add']+=1
        s=(y2-y1)*inv(x2-x1)%P
    x3=(s*s-x1-x2)%P
    return x3,(s*(x1-x3)-y1)%P

def mul(k,Q=G):
    OPS['mul_calls']+=1
    k%=N;R=None;T=Q
    while k:
        if k&1:R=add(R,T)
        T=add(T,T);k>>=1
    return R

def key(Q): return 'O' if Q is None else f'{Q[0]},{Q[1]}'
def hres(a): a%=P; return min(a,P-a)
def sqrt_fp(a):
    a%=P
    if a==0:return 0
    y=pow(a,(P+1)//4,P)
    return y if y*y%P==a else None

def rank_mod(rows,mod=N):
    A=[[x%mod for x in r] for r in rows];m=len(A);n=len(A[0]) if A else 0
    rank=0
    for c in range(n):
        piv=next((i for i in range(rank,m) if A[i][c]),None)
        if piv is None:continue
        A[rank],A[piv]=A[piv],A[rank]
        iv=pow(A[rank][c],-1,mod);A[rank]=[v*iv%mod for v in A[rank]]
        for i in range(m):
            if i!=rank and A[i][c]:
                q=A[i][c];A[i]=[(A[i][j]-q*A[rank][j])%mod for j in range(n)]
        rank+=1
        if rank==m:return rank
    return rank

def solve_mod(A,b,mod=N):
    n=len(A[0]);M=[[x%mod for x in row]+[bb%mod] for row,bb in zip(A,b)];r=0
    pivcols=[]
    for c in range(n):
        piv=next((i for i in range(r,len(M)) if M[i][c]),None)
        if piv is None:continue
        M[r],M[piv]=M[piv],M[r]
        iv=pow(M[r][c],-1,mod);M[r]=[x*iv%mod for x in M[r]]
        for i in range(len(M)):
            if i!=r and M[i][c]:
                q=M[i][c];M[i]=[(M[i][j]-q*M[r][j])%mod for j in range(n+1)]
        pivcols.append(c);r+=1
        if r==n:break
    if r<n:raise RuntimeError(f'rank {r} < {n}')
    x=[0]*n
    for i,c in enumerate(pivcols):x[c]=M[i][-1]
    return x

def build_factor_base(height=16):
    reps=[]
    for x in range(P):
        if hres(x)>height:continue
        y=sqrt_fp((x*x%P*x+B)%P)
        if y is None or y==0:continue
        yc=min(y,(-y)%P)
        reps.append((x,yc))
    reps=sorted(set(reps))
    signed=[]
    for i,Q in enumerate(reps):
        signed.append((Q,i,1));signed.append((neg(Q),i,-1))
    return reps,signed

REPS,SIGNED=build_factor_base(16)
assert len(REPS)==18 and len(SIGNED)==36

# Build the 2-sum table from point values only.  Coefficients record variables/signs,
# not discrete logarithms.
pair={}; pair_ops_before=dict(OPS)
for a in range(len(SIGNED)):
    Q,i,si=SIGNED[a]
    for b in range(a,len(SIGNED)):
        R,j,sj=SIGNED[b]
        S=add(Q,R)
        coeff=[0]*len(REPS);coeff[i]+=si;coeff[j]+=sj
        pair.setdefault(key(S),coeff)
pair_build_ops={k:OPS[k]-pair_ops_before[k] for k in OPS}

def decompose3(R):
    for Q,i,si in SIGNED:
        need=add(R,neg(Q));hit=pair.get(key(need))
        if hit is not None:
            c=hit.copy();c[i]+=si
            return [x%N for x in c]
    return None

# Relation collection. r is chosen by us, so log([r]G)=r is known independently.
rng=random.Random(0x5107)
rows=[];rhs=[];trials=0;decomposed=0;rank_history=[]
t0=time.perf_counter()
while len(rows)<len(REPS) and trials<10000:
    trials+=1;r=rng.randrange(1,N);R=mul(r)
    c=decompose3(R)
    if c is None:continue
    decomposed+=1
    old=rank_mod(rows) if rows else 0
    new=rank_mod(rows+[c])
    if new>old:
        rows.append(c);rhs.append(r);rank_history.append({'trial':trials,'rank':new})
relation_seconds=time.perf_counter()-t0
if len(rows)<len(REPS):raise RuntimeError('failed to obtain full relation rank')
logs=solve_mod(rows,rhs)
log_verification=[mul(logs[i])==REPS[i] for i in range(len(REPS))]
assert all(log_verification)

# Target decomposition and recovery.  No known target log is used.
target_coeff=decompose3(TARGET)
if target_coeff is None:raise RuntimeError('target did not decompose over selected base')
derived_k=sum(a*b for a,b in zip(target_coeff,logs))%N
point_verification=(mul(derived_k)==TARGET)
assert point_verification

# Point-side BSGS baseline with the same arithmetic counter.
def bsgs_point(H):
    m=math.isqrt(N)+1
    table={};R=None
    for j in range(m):table.setdefault(key(R),j);R=add(R,G)
    step=neg(mul(m));cur=H
    for i in range(m+1):
        if key(cur) in table:
            k=i*m+table[key(cur)]
            if k<N and mul(k)==H:return k,{'m':m,'baby_entries':len(table),'giant_steps':i+1}
        cur=add(cur,step)
    return None,None
before_bsgs=dict(OPS);bsgs_k,bsgs_meta=bsgs_point(TARGET)
bsgs_ops={k:OPS[k]-before_bsgs[k] for k in OPS}
assert bsgs_k==derived_k

# Exact S4-restricted symbolic benchmark for the actual 3-factor relation equation.
semaev={}
try:
    import sympy as sp
    x1,x2,x3,x4,z=sp.symbols('x1 x2 x3 x4 z')
    def s3(X,Y,Z):return (X-Y)**2*Z**2-2*((X+Y)*(X*Y)+2*B)*Z+(X*Y)**2-4*B*(X+Y)
    S4=sp.Poly(sp.resultant(s3(x1,x2,z),s3(x3,x4,z),z),x1,x2,x3,x4,modulus=P)
    F=[]
    for X in (x1,x2,x3):
        q=sp.Poly(1,X,modulus=P)
        for a,_ in REPS:q*=sp.Poly(X-a,X,modulus=P)
        F.append(q)
    class Timeout(Exception):pass
    def handler(_s,_f):raise Timeout()
    signal.signal(signal.SIGALRM,handler)
    eq=S4.as_expr().subs(x4,TARGET[0])
    semaev={'S4_degrees':[S4.degree(v) for v in (x1,x2,x3,x4)],'S4_total_degree':S4.total_degree(),
            'S4_terms':len(S4.terms()),'factor_base_poly_degree':len(REPS),
            'boolean_box_monomial_bound':(len(REPS)+1)**3}
    try:
        signal.alarm(120);t=time.perf_counter()
        gb=sp.groebner([eq]+[q.as_expr() for q in F],x3,x2,x1,modulus=P,order='lex')
        signal.alarm(0)
        semaev.update({'groebner_status':'completed','groebner_seconds':time.perf_counter()-t,
                       'basis_count':len(gb.polys),'basis_degrees':[q.total_degree() for q in gb.polys],
                       'basis_terms':[len(q.terms()) for q in gb.polys]})
    except Timeout:
        signal.alarm(0);semaev['groebner_status']='timeout_120s'
    except Exception as e:
        signal.alarm(0);semaev.update({'groebner_status':'error','error':repr(e)})
except Exception as e:semaev={'error':repr(e)}

# Asymptotic audit for m-sum meet-in-middle and relation-linear-algebra costs.
# A random B-element base has m-sum coverage scale min(N,B^m); finding one m-sum
# by balanced MITM costs B^ceil(m/2).  To make probability constant, B~N^(1/m),
# hence target decomposition cost ~N^(ceil(m/2)/m), never below sqrt(N), and
# relation collection/linear algebra add at least B^2 for sparse iterative solving.
asymptotic=[]
for m in range(2,9):
    exp=math.ceil(m/2)/m
    asymptotic.append({'terms':m,'factor_base_exponent':1/m,'balanced_MITM_exponent':exp,
                       'beats_sqrt':exp<0.5})

result={
 'construction_non_circularity':{
   'known_target_scalar_read':False,'orbit_index_read':False,'phaseword_read':False,
   'factor_base_logs_read':False,'factor_base_rule':'min(x,p-x)<=16 and y canonical',
   'relations_source':'random known multiples [r]G'},
 'factor_base':{'height':16,'variables':len(REPS),'signed_points':len(SIGNED),'representatives':REPS},
 'pair_table':{'entries':len(pair),'operations':pair_build_ops},
 'relation_collection':{'trials':trials,'decomposed_trials':decomposed,'empirical_probability':decomposed/trials,
                        'independent_relations':len(rows),'rank_history':rank_history,'seconds':relation_seconds},
 'solved_factor_base_logs':logs,'factor_base_point_verification_all':all(log_verification),
 'target':{'point':TARGET,'coefficient_vector':target_coeff,'nonzero_terms':[(i,c) for i,c in enumerate(target_coeff) if c],
           'derived_k':derived_k,'point_verification':point_verification},
 'bsgs_baseline':{'derived_k':bsgs_k,'metadata':bsgs_meta,'operations':bsgs_ops},
 'total_operations_after_all_checks':OPS,'semaev_actual_relation_benchmark':semaev,
 'asymptotic_meet_in_middle_audit':asymptotic,
 'structural_conclusion':'This fixed-instance index calculus succeeds, but constant-probability m-sum decomposition has balanced MITM exponent ceil(m/2)/m >= 1/2; the coordinate factor base does not yield an asymptotic advantage over generic square-root methods.'
}
(OUT/'index_calculus_results.json').write_text(json.dumps(result,indent=2)+'\n')
report=f'''# Strict coordinate-factor-base index-calculus receipt

- Factor-base variables: {len(REPS)} (36 signed points), rule `min(x,p-x)<=16`.
- Relation trials: {trials}; decomposed: {decomposed}; probability {decomposed/trials:.6f}.
- Independent relations: {len(rows)}; all solved logs verified by point multiplication: {all(log_verification)}.
- Target decomposition nonzero terms: {[(i,c) for i,c in enumerate(target_coeff) if c]}.
- Derived scalar: `{derived_k}`; point verification: `{point_verification}`.
- Point-BSGS baseline scalar: `{bsgs_k}`; baby entries `{bsgs_meta['baby_entries']}`, giant steps `{bsgs_meta['giant_steps']}`.
- Actual S4/Groebner benchmark: `{json.dumps(semaev,sort_keys=True)}`.

No target scalar, orbit index, PhaseWord, atlas, or precomputed factor-base logarithm was used by the construction.
The success is fixed-instance and non-generic, but not asymptotically faster: balanced m-sum MITM has exponent `ceil(m/2)/m >= 1/2`.
'''
(OUT/'INDEX_CALCULUS_REPORT.md').write_text(report)
print(json.dumps(result,indent=2))
