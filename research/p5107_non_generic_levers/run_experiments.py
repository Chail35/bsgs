#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, itertools, json, math, os, random, signal, time, zipfile
from collections import Counter
from pathlib import Path

P=5107
N=4993
A=0
B=7
G=(2,530)
TARGET=(3711,748)
MPSI=2344
MU6=(1,2650,2651,4992,2343,2342)
OUT=Path(__file__).resolve().parent/'outputs'
OUT.mkdir(parents=True,exist_ok=True)

O=None

def inv(a,m=P): return pow(a%m,-1,m)
def neg(Q): return None if Q is None else (Q[0],(-Q[1])%P)
def add(Q,R):
    if Q is None:return R
    if R is None:return Q
    x1,y1=Q;x2,y2=R
    if x1==x2 and (y1+y2)%P==0:return None
    if Q!=R:s=((y2-y1)*inv(x2-x1))%P
    else:
        if y1==0:return None
        s=(3*x1*x1*inv(2*y1))%P
    x3=(s*s-x1-x2)%P
    y3=(s*(x1-x3)-y1)%P
    return x3,y3

def mul(k,Q=G):
    k%=N
    R=None
    T=Q
    while k:
        if k&1:R=add(R,T)
        T=add(T,T);k>>=1
    return R

def point_key(Q): return 'O' if Q is None else f'{Q[0]},{Q[1]}'
def center(a,m):
    a%=m
    return a-m if a>m//2 else a

def factor(n):
    d=2;f=[]
    while d*d<=n:
        e=0
        while n%d==0:n//=d;e+=1
        if e:f.append((d,e))
        d+=1
    if n>1:f.append((n,1))
    return f

def divisors(n):
    ds=[1]
    for q,e in factor(n):
        ds=[d*q**j for d in ds for j in range(e+1)]
    return sorted(ds)

def order_mod(a,m):
    if math.gcd(a,m)!=1:return 0
    o=m-1
    for q,e in factor(o):
        for _ in range(e):
            if o%q==0 and pow(a,o//q,m)==1:o//=q
            else:break
    return o

# Full group table; this is experimental ground truth, never used as an input to a claimed compact algorithm.
scalar_to_point=[None]*N
point_to_scalar={}
for k in range(N):
    Q=mul(k)
    scalar_to_point[k]=Q
    point_to_scalar[point_key(Q)]=k
assert scalar_to_point[0] is None and scalar_to_point[1]==G and mul(N-1)==neg(G)
assert mul(N) is None
assert all(Q is None or (Q[1]*Q[1]-Q[0]**3-B)%P==0 for Q in scalar_to_point)

# CM collapse on the prime-order rational subgroup.
zeta_roots=[z for z in range(P) if (z*z+z+1)%P==0]
cm=[]
for z in zeta_roots:
    WG=(z*G[0]%P,G[1])
    lam=point_to_scalar[point_key(WG)]
    cm.append({'zeta':z,'lambda_mod_N':lam,'lambda_poly':(lam*lam+lam+1)%N,'point':WG})
# choose the convention matching omega scalar 2342 when available
cm.sort(key=lambda r:(r['lambda_mod_N']!=2342,r['lambda_mod_N']))
ZETA=cm[0]['zeta']; LAMBDA=cm[0]['lambda_mod_N']
assert all(mul((a+b*LAMBDA)%N)==add(mul(a),mul(b,(ZETA*G[0]%P,G[1]))) for a,b in [(1,1),(7,11),(123,456),(N-2,77)])

# Factor-base constructors.
def subgroup(gen,ordr): return {pow(gen,j,N) for j in range(ordr)}
def orbit_reps(step,count): return {pow(MPSI,step*j,N) for j in range(count)}
def t_of_scalar(k):
    Q=scalar_to_point[k]
    return None if Q is None else pow(Q[0],3,P)
def hres(a,m=P):
    a%=m
    return min(a,m-a)

factor_bases={}
factor_bases['quotient_reps_832']=set(pow(MPSI,i,N) for i in range(832))
factor_bases['lattes_full_nonzero']=set(range(1,N))
factor_bases['suborbit13_reps']=orbit_reps(64,13)
factor_bases['suborbit64_reps']=orbit_reps(13,64)
# Saturated suborbits are actual multiplicative subgroups of the scalar field.
g78=pow(MPSI,64,N);g384=pow(MPSI,13,N)
factor_bases['suborbit13_mu6_saturated']=subgroup(g78,78)
factor_bases['suborbit64_mu6_saturated']=subgroup(g384,384)
for r in (4,8,13,16,32,64):
    factor_bases[f'quotient_prefix_{r}']=set(pow(MPSI,i,N) for i in range(r))
for H in (4,8,16,32,64,128,256):
    factor_bases[f't_height_le_{H}']={k for k in range(1,N) if hres(t_of_scalar(k))<=H}
for H in (4,8,16,32,64,128):
    factor_bases[f'x_height_le_{H}']={k for k in range(1,N) if hres(scalar_to_point[k][0])<=H}

rng=random.Random(20260724)
sample_targets=[rng.randrange(N) for _ in range(2000)]

def sumset_stats(F,max_m=4):
    F=set(x%N for x in F)
    counts=[0]*N;counts[0]=1
    rows=[]
    for m in range(1,max_m+1):
        new=[0]*N
        for f in F:
            for x,c in enumerate(counts):
                if c:new[(x+f)%N]+=c
        counts=new
        support=[i for i,c in enumerate(counts) if c]
        vals=[counts[i] for i in support]
        rows.append({'m':m,'coverage':len(support),'probability':len(support)/N,
                     'sample_hits':sum(counts[t]>0 for t in sample_targets),'sample_total':len(sample_targets),
                     'representation_min':min(vals) if vals else 0,'representation_max':max(vals) if vals else 0,
                     'representation_mean':sum(vals)/len(vals) if vals else 0.0})
        if len(support)==N:
            break
    return rows

fb_rows=[]
for name,F in factor_bases.items():
    pts={point_key(scalar_to_point[k]) for k in F}
    xs={scalar_to_point[k][0] for k in F if k}
    ts={t_of_scalar(k) for k in F if k}
    row={'name':name,'size':len(F),'point_size':len(pts),'unique_x':len(xs),'unique_t':len(ts),
         'negation_closed':all((-k)%N in F for k in F),'mu6_closed':all((u*k)%N in F for k in F for u in MU6),
         'known_scalar_parametrization':True}
    if len(F)<=900: row['sumsets']=sumset_stats(F)
    else: row['sumsets']='skipped_size_gt_900'
    fb_rows.append(row)

# Point-side decomposition search; target scalar is not consulted.
def find_decomp(F,max_terms=3):
    L=sorted(F)
    pair={}
    adds=0
    for i,a in enumerate(L):
        Qa=scalar_to_point[a]
        for b in L[i:]:
            S=add(Qa,scalar_to_point[b]);adds+=1
            pair.setdefault(point_key(S),(a,b))
    if point_key(TARGET) in pair:
        a,b=pair[point_key(TARGET)]
        return {'terms':2,'scalars':[a,b],'derived_k':(a+b)%N,'group_additions':adds}
    if max_terms>=3:
        for c in L:
            need=add(TARGET,neg(scalar_to_point[c]));adds+=1
            if point_key(need) in pair:
                a,b=pair[point_key(need)]
                return {'terms':3,'scalars':[a,b,c],'derived_k':(a+b+c)%N,'group_additions':adds}
    return {'terms':None,'group_additions':adds}

decomp=[]
for row in sorted(fb_rows,key=lambda x:x['size']):
    if row['size']<=400:
        d=find_decomp(factor_bases[row['name']])
        d['factor_base']=row['name'];d['factor_base_size']=row['size']
        if d['terms']:
            d['point_verification']=mul(d['derived_k'])==TARGET
        decomp.append(d)

# First actual decomposition, followed only then by comparison to the held-out reference.
first_solution=next((d for d in decomp if d.get('terms')),None)

def bsgs(g,h,order,mod):
    m=math.isqrt(order)+1
    tab={}
    e=1
    for j in range(m):tab.setdefault(e,j);e=e*g%mod
    fac=pow(pow(g,m,mod),-1,mod);cur=h
    for i in range(m+1):
        if cur in tab:
            x=i*m+tab[cur]
            if x<order and pow(g,x,mod)==h:return x
        cur=cur*fac%mod
    return None

phase_from_derived=None
if first_solution:
    kder=first_solution['derived_k']
    gq=pow(MPSI,6,N); y=pow(kder,6,N)
    idx=bsgs(gq,y,832,N)
    phase_from_derived={'derived_k':kder,'quotient_index':idx,'phase64':59*idx%64,'phase13':idx%13,
                        'predecessor_index':(idx-283)%832,'phase64_predecessor':59*(idx-283)%832%64,
                        'phase13_predecessor':(idx-283)%13,'held_out_k_match':kder==3955}

# Semaev polynomials and actual symbolic benchmarks.
semaev={}
try:
    import sympy as sp
    x1,x2,x3,x4,z=sp.symbols('x1 x2 x3 x4 z')
    def s3(X,Y,Z):
        return (X-Y)**2*Z**2 - 2*((X+Y)*(X*Y+A)+2*B)*Z + (X*Y-A)**2 - 4*B*(X+Y)
    S3=sp.Poly(sp.expand(s3(x1,x2,x3)),x1,x2,x3,modulus=P)
    t0=time.perf_counter()
    S4expr=sp.resultant(s3(x1,x2,z),s3(x3,x4,z),z)
    S4=sp.Poly(sp.expand(S4expr),x1,x2,x3,x4,modulus=P)
    semaev['S3']={'degrees':[S3.degree(v) for v in (x1,x2,x3)],'total_degree':S3.total_degree(),'terms':len(S3.terms())}
    semaev['S4']={'degrees':[S4.degree(v) for v in (x1,x2,x3,x4)],'total_degree':S4.total_degree(),'terms':len(S4.terms()),'construction_seconds':time.perf_counter()-t0}
    class Timeout(Exception):pass
    def alarm_handler(signum,frame):raise Timeout()
    signal.signal(signal.SIGALRM,alarm_handler)
    benches=[]
    candidates=[r for r in sorted(fb_rows,key=lambda r:r['unique_x']) if r['negation_closed'] and r['unique_x']<=45]
    # Include a spread of sizes, including the 39-x order-13 saturated base.
    picked=[]
    for r in candidates:
        if not picked or r['unique_x']>=picked[-1]['unique_x']+4 or r['name']=='suborbit13_mu6_saturated':picked.append(r)
        if len(picked)>=5:break
    if not any(r['name']=='suborbit13_mu6_saturated' for r in picked):
        picked.append(next(r for r in fb_rows if r['name']=='suborbit13_mu6_saturated'))
    for r in picked:
        roots=sorted({scalar_to_point[k][0] for k in factor_bases[r['name']]})
        F1=sp.Poly(1,x1,modulus=P);F2=sp.Poly(1,x2,modulus=P)
        for a in roots:
            F1*=sp.Poly(x1-a,x1,modulus=P);F2*=sp.Poly(x2-a,x2,modulus=P)
        rec={'factor_base':r['name'],'x_degree':len(roots),'S3_specialized_degrees':[2,2],
             'bezout_monomial_bound':(len(roots)+1)**2}
        try:
            signal.alarm(60);t=time.perf_counter()
            GB=sp.groebner([s3(x1,x2,TARGET[0]),F1.as_expr(),F2.as_expr()],x2,x1,modulus=P,order='lex')
            signal.alarm(0)
            rec.update({'groebner_status':'completed','groebner_seconds':time.perf_counter()-t,
                        'basis_count':len(GB.polys),'basis_degrees':[p.total_degree() for p in GB.polys],
                        'basis_terms':[len(p.terms()) for p in GB.polys]})
        except Timeout:
            signal.alarm(0);rec.update({'groebner_status':'timeout_60s'})
        except Exception as e:
            signal.alarm(0);rec.update({'groebner_status':'error','error':repr(e)})
        try:
            signal.alarm(60);t=time.perf_counter()
            Res=sp.Poly(sp.resultant(s3(x1,x2,TARGET[0]),F2.as_expr(),x2),x1,modulus=P)
            GCD=sp.gcd(Res,F1)
            signal.alarm(0)
            rec.update({'resultant_status':'completed','resultant_seconds':time.perf_counter()-t,
                        'resultant_degree':Res.degree(),'resultant_terms':len(Res.terms()),'gcd_degree':GCD.degree()})
        except Timeout:
            signal.alarm(0);rec.update({'resultant_status':'timeout_60s'})
        except Exception as e:
            signal.alarm(0);rec.update({'resultant_status':'error','resultant_error':repr(e)})
        benches.append(rec)
    semaev['benchmarks']=benches
except Exception as e:
    semaev={'error':repr(e)}

# Kani/torsion inventory.
def matmul(A0,B0,m):
    return [[sum(A0[i][k]*B0[k][j] for k in range(2))%m for j in range(2)] for i in range(2)]
def matpow(A0,e,m):
    R=[[1,0],[0,1]];Q=A0
    while e:
        if e&1:R=matmul(R,Q,m)
        Q=matmul(Q,Q,m);e>>=1
    return R
def matorder(A0,m,limit=1000000):
    R=[[1,0],[0,1]]
    for e in range(1,limit+1):
        R=matmul(R,A0,m)
        if R==[[1,0],[0,1]]:return e
    return None
trace=P+1-N
kani_torsion=[]
for m in (2,4,8,16,32,64,128,3,13):
    Frob=[[0,(-P)%m],[1,trace%m]]
    kani_torsion.append({'m':m,'gcd_m_group_order':math.gcd(m,N),'frobenius_matrix':Frob,
                         'full_torsion_extension_degree':matorder(Frob,m),
                         'rational_m_torsion_points':math.gcd(m,N)})
kani={'cm_roots':cm,'chosen_zeta':ZETA,'chosen_lambda':LAMBDA,
      'rank_test_symbolic_determinant':'det([[1,lambda],[k,k*lambda]]) = 0',
      'random_rank_tests':[{'k':k,'det':(1*(k*LAMBDA)-LAMBDA*k)%N} for k in (2,7,123,2026)],
      'available_unknown_map_images':1,
      'omega_image_is_redundant':mul(LAMBDA,TARGET)==(ZETA*TARGET[0]%P,TARGET[1]),
      'required_auxiliary_torsion_basis_rank':2,'torsion_inventory':kani_torsion,
      'structural_gap':'Kani graph kernels require the unknown map on a basis of smooth prime-power torsion; P=[k]G and omega(P) provide only one dependent N-torsion eigenline and no images on auxiliary torsion.'}

# Cheon prerequisite and affine-closure theorem.
cheon=[]
for d in divisors(N-1):
    if d>1:
        cheon.append({'d':d,'estimated_given_auxiliary':math.sqrt(N/d)+math.sqrt(d)})
cheon.sort(key=lambda r:r['estimated_given_auxiliary'])
known_scalars=sorted(set([1,P%N,MPSI,LAMBDA,(1-LAMBDA)%N]+list(MU6)))
nonlinear_matches=[]
for d in (2,3,4,6,8,13,16,32,64,128,384,416):
    best=(0,None)
    for c in known_scalars:
        cnt=sum(1 for k in range(1,N) if c*k%N==pow(k,d,N))
        if cnt>best[0]:best=(cnt,c)
    nonlinear_matches.append({'d':d,'best_fixed_endomorphism_scalar':best[1],'matching_nonzero_secrets':best[0],
                              'fraction':best[0]/(N-1)})
# Random circuit closure in pairs (a,b), representing [a+b*k]G.
rng2=random.Random(9);states=[(1,0),(0,1)]
for _ in range(1000):
    u=states[rng2.randrange(len(states))];v=states[rng2.randrange(len(states))]
    if rng2.randrange(2):w=((u[0]+v[0])%N,(u[1]+v[1])%N)
    else:
        c=known_scalars[rng2.randrange(len(known_scalars))];w=(c*u[0]%N,c*u[1]%N)
    states.append(w)
cheon_result={'N_minus_1_factorization':factor(N-1),'best_d_if_auxiliary_existed':cheon[:10],
              'known_endomorphism_scalars':known_scalars,'nonlinear_auxiliary_match_tests':nonlinear_matches,
              'affine_closure_states_tested':len(states),
              'theorem':'Starting from G and P=[k]G, group addition and fixed CM endomorphisms produce only [a+b*k]G. They cannot produce [k^d]G for d>1 without a k-dependent operation.'}

# Novel angle: hidden GLV decomposition lattice.
def gauss_reduce(v1,v2):
    v1=list(v1);v2=list(v2)
    while True:
        if v1[0]*v1[0]+v1[1]*v1[1] > v2[0]*v2[0]+v2[1]*v2[1]:v1,v2=v2,v1
        den=v1[0]*v1[0]+v1[1]*v1[1]
        mu=round((v1[0]*v2[0]+v1[1]*v2[1])/den)
        if mu==0:return tuple(v1),tuple(v2)
        v2=[v2[0]-mu*v1[0],v2[1]-mu*v1[1]]
lat_basis=gauss_reduce((N,0),(-LAMBDA,1))
# Exact minimal L-infinity representation for every scalar.
min_reps=[];maxB=0
for k in range(N):
    best=None
    for b in range(-N//2,N//2+1):
        a=center(k-b*LAMBDA,N)
        score=max(abs(a),abs(b))
        if best is None or score<best[0]:best=(score,a,b)
        if abs(b)>best[0] and abs(b)>math.isqrt(N)+5: # no later b can improve L-infinity
            pass
    min_reps.append(best);maxB=max(maxB,best[0])
# Solve target by 2D MITM without using reference scalar.
def glv_solve(Pt,Bnd):
    table={point_key(mul(a)):a for a in range(-Bnd,Bnd+1)}
    omegaG=(ZETA*G[0]%P,G[1]);ops=0
    for b in range(-Bnd,Bnd+1):
        need=add(Pt,neg(mul(b,omegaG)));ops+=1
        key=point_key(need)
        if key in table:
            a=table[key];return {'a':a,'b':b,'derived_k':(a+b*LAMBDA)%N,'lookups':ops,'table_size':len(table)}
    return None
glv_target=glv_solve(TARGET,maxB)
if glv_target:glv_target['verification']=mul(glv_target['derived_k'])==TARGET;glv_target['held_out_k_match']=glv_target['derived_k']==3955
glv={'lambda':LAMBDA,'kernel_lattice_basis_initial':[[N,0],[-LAMBDA,1]],'gauss_reduced_basis':lat_basis,
     'determinant':abs(lat_basis[0][0]*lat_basis[1][1]-lat_basis[0][1]*lat_basis[1][0]),
     'max_minimal_Linf_bound':maxB,'mean_minimal_Linf':sum(r[0] for r in min_reps)/N,
     'representation_count_in_box':(2*maxB+1)**2,'group_order':N,
     'target_solution':glv_target,
     'complexity':'A one-dimensional MITM table and one-dimensional scan each have 2B+1 = Theta(sqrt(N)) entries; CM gives a constant-factor GLV decomposition, not a sub-square-root algorithm.'}

# Branch-factor-base structural prerequisite from settled graph counts.
branch={'theta_graph_vertices':769,'theta_graph_edges':832,'degree_histogram':{'2':709,'4':57,'6':3},
        'branch_values':60,'branch_preimage_quotient_classes':57*2+3*3,
        'mu6_saturated_point_upper_count':6*(57*2+3*3),
        'empirical_decomposition_probability':'not_computed_without_nested branch-index CSV',
        'disqualification':'Membership in this factor base is defined by global collisions of Omega_1 over the complete orbit; the corpus has no point-local membership predicate. Constructing the set currently requires the very orbit scan an index-calculus factor base must avoid.'}

# Cross-check and report.
result={'instance':{'p':P,'N':N,'curve':'y^2=x^3+7','G':G,'target':TARGET,'mpsi':MPSI,
                    'mpsi_order_mod_N':order_mod(MPSI,N)},
        'cm_collapse':{'roots':cm,'chosen':{'zeta':ZETA,'lambda':LAMBDA},
                       'theorem':'Every Z[omega] endomorphism a+b*omega restricts on cyclic E(F_p) to scalar a+b*lambda mod N.'},
        'factor_bases':fb_rows,'target_decompositions':decomp,'first_target_solution':first_solution,
        'phase_from_derived_solution':phase_from_derived,'semaev':semaev,'branch_factor_base':branch,
        'kani':kani,'cheon':cheon_result,'glv':glv}
(OUT/'results.json').write_text(json.dumps(result,indent=2,default=str)+'\n')

with (OUT/'factor_base_summary.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['name','size','unique_x','unique_t','negation_closed','mu6_closed','m','coverage','probability','sample_hits'])
    for r in fb_rows:
        if isinstance(r['sumsets'],list):
            for s in r['sumsets']:w.writerow([r['name'],r['size'],r['unique_x'],r['unique_t'],r['negation_closed'],r['mu6_closed'],s['m'],s['coverage'],s['probability'],s['sample_hits']])
        else:w.writerow([r['name'],r['size'],r['unique_x'],r['unique_t'],r['negation_closed'],r['mu6_closed'],'','','',''])

report=[]
report.append('# p=5107 non-generic ECDLP levers — executable report')
report.append('')
report.append('## Executive result')
report.append(f'- CM scalar collapse: omega acts as lambda={LAMBDA} mod {N}; every a+b*omega action is one scalar.')
if first_solution:
    report.append(f"- First point-side factor-base decomposition found: {first_solution['factor_base']} with {first_solution['terms']} terms and {first_solution['group_additions']} group additions; derived k={first_solution['derived_k']}.")
if phase_from_derived:
    report.append(f"- Held-out verification only after construction: k match={phase_from_derived['held_out_k_match']}, quotient index={phase_from_derived['quotient_index']}, PhaseWords=({phase_from_derived['phase64']},{phase_from_derived['phase13']}).")
report.append(f"- Kani: omega(G) is dependent, and no nontrivial 2-,3-,13-torsion is rational because #E(F_p)={N} is prime.")
report.append(f"- Cheon: best hypothetical auxiliary-input d is {cheon[0]['d']} with cost proxy {cheon[0]['estimated_given_auxiliary']:.3f}, but fixed endomorphisms preserve affine degree in the secret.")
report.append(f"- GLV lattice: max minimal coefficient bound B={maxB}; MITM cost 2B+1={2*maxB+1}=Theta(sqrt(N)).")
report.append('')
report.append('## Factor-base measurements')
for r in fb_rows:
    if isinstance(r['sumsets'],list):
        txt=', '.join(f"{s['m']}-sum {s['coverage']}/{N}={s['probability']:.4f}" for s in r['sumsets'])
        report.append(f"- **{r['name']}**: |F|={r['size']}, x={r['unique_x']}, t={r['unique_t']}; {txt}.")
report.append('')
report.append('## Semaev/Groebner receipts')
report.append('```json\n'+json.dumps(semaev,indent=2,default=str)+'\n```')
report.append('')
report.append('## Kani torsion inventory')
report.append('```json\n'+json.dumps(kani,indent=2,default=str)+'\n```')
report.append('')
report.append('## Cheon prerequisite audit')
report.append('```json\n'+json.dumps(cheon_result,indent=2,default=str)+'\n```')
report.append('')
report.append('## Novel GLV hidden-decomposition audit')
report.append('```json\n'+json.dumps(glv,indent=2,default=str)+'\n```')
report.append('')
report.append('## Strict receiver status')
report.append('The experiments may derive the small-instance discrete logarithm by nonqualifying finite work, but they do not construct the normalized line/path tuple (mu1,w_R,PhaseWords) with the required uniform bound. The original strict receiver is therefore not claimed closed.')
(OUT/'REPORT.md').write_text('\n'.join(report)+'\n')

# Self-contained evidence bundle.
bundle=OUT/'p5107_non_generic_levers_20260724.zip'
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as zf:
    for pth in [Path(__file__),OUT/'results.json',OUT/'factor_base_summary.csv',OUT/'REPORT.md']:
        zf.write(pth,pth.relative_to(Path(__file__).resolve().parent))
sha=hashlib.sha256(bundle.read_bytes()).hexdigest()
(OUT/'p5107_non_generic_levers_20260724.zip.sha256').write_text(f'{sha}  {bundle.name}\n')
print(json.dumps({'bundle':str(bundle),'sha256':sha,'first_solution':first_solution,'phase':phase_from_derived,
                  'cm_lambda':LAMBDA,'glv_B':maxB,'semaev':semaev},indent=2,default=str))
