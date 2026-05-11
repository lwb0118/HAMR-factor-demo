#!/usr/bin/env python3
"""Overfit tests for HAMR factor."""
import sys,warnings,numpy as np,pandas as pd,tushare as ts
from scipy import stats
from pathlib import Path
warnings.filterwarnings('ignore')
ROOT=Path(__file__).parent.absolute()
sys.path.insert(0,str(ROOT))
from config import TUSHARE_TOKEN,GITHUB_TOKEN
ts.set_token(TUSHARE_TOKEN)
from src import ai_heat,template_cluster,quality,residual,funding_vacuum,hamr_factor
from src.github_timeseries import build_github_timeseries
from src.news_aiheat import build_news_timeseries
from src.ai_heat import load_search_aiheat
from src.guba_data import fetch_universe_posts,compute_community_aiheat,compute_stock_attention

panel=pd.read_pickle('data/_panel_200stocks.pkl')
codes=panel['code'].unique().tolist()

github_ts=build_github_timeseries(token=GITHUB_TOKEN)
news_ts=build_news_timeseries()
search_ts=load_search_aiheat()
guba_posts=fetch_universe_posts(codes,max_workers=3)
community=compute_community_aiheat(guba_posts) if guba_posts else None
stock_attn=compute_stock_attention(None,guba_posts) if guba_posts else None
ai_state=ai_heat.compute_ai_heat_from_panel(panel,github_ts=github_ts,news_ts=news_ts,community=community,search_ts=search_ts)
try:
    mismatch=template_cluster.compute_template_affinity(panel,recent_only=False,recent_days=20)
except:
    mismatch=template_cluster.compute_template_affinity(panel, n_clusters=3, recent_only=False, recent_days=20, lookback_days=20)
qual=quality.compute_quality_score(panel)
res_weak=residual.compute_residual_weakness(panel)
nonfund=residual.compute_non_fundamental_ok(panel)
mispricing=residual.compute_mispricing_pressure(mismatch,res_weak,nonfund)
vacuum=funding_vacuum.compute_funding_vacuum(panel,stock_attention_df=stock_attn)
liq=funding_vacuum.compute_liquidity_capacity(panel)
trap=funding_vacuum.compute_trap_guard(qual,liq,panel=panel)

df=panel[['date','code','close']].copy()
for src,col in [(mismatch,'MismatchScore'),(qual,'QualityScore'),(mispricing,'MispricingPressure'),(res_weak,'ResidualWeakness'),(nonfund,'NonFundamentalOK'),(vacuum,'FundingVacuum'),(liq,'LiquidityCapacity'),(trap,'TrapGuard')]:
    df=df.merge(src[['date','code',col]],on=['date','code'],how='left')
df=df.merge(ai_state[['date','AIStateScore']],on='date',how='left')
for c in ['MismatchScore','QualityScore','MispricingPressure','ResidualWeakness','NonFundamentalOK','FundingVacuum','LiquidityCapacity','TrapGuard']:
    df[c]=df[c].fillna(0.5).clip(0,1)
df['AIStateScore']=df['AIStateScore'].fillna(0.5).clip(0,1)
df=hamr_factor.compute_forward_returns(df)

def ic_summary(df,fcol,ret_col='fwd_20d'):
    ics=[]
    for dt,g in df.groupby('date'):
        v=g[[fcol,ret_col]].dropna()
        if len(v)<10: continue
        fv=v[fcol].values
        if np.std(fv)==0: continue
        ic,_=stats.spearmanr(fv,v[ret_col].values)
        if np.isfinite(ic): ics.append(ic)
    ics=np.array(ics)
    if len(ics)==0: return 0,0,0,0
    m=ics.mean();s=ics.std(ddof=1)
    nw_t=_nw_t(ics)
    return m,m/s if s>0 else 0,nw_t,(ics>0).mean()

def _nw_t(series):
    n=len(series)
    if n<3: return 0
    lag=int(4*(n/100)**(2/9))
    lag=min(lag,n-1)
    acov=[np.cov(series[:-k] if k>0 else series,series[k:])[0,1] for k in range(lag+1)]
    nw_var=acov[0]+2*sum((1-k/(lag+1))*acov[k] for k in range(1,lag+1))
    return series.mean()/np.sqrt(nw_var/n) if nw_var>0 else 0

dates=sorted(df['date'].unique())
train_end=dates[179]
train_dates=dates[:180]
test_dates=dates[180:]

print('Train:',len(train_dates),'days (',train_dates[0].date(),'->',train_dates[-1].date(),')')
print('Test:',len(test_dates),'days (',test_dates[0].date(),'->',test_dates[-1].date(),')')
print()

# Test 1: Document formula
print('='*60)
print('TEST 1: ORIGINAL DOCUMENT FORMULA')
print('  HAMR = AS * Q * MP * FV * L * T')
print('='*60)
df['hamr_doc']=(df['AIStateScore']*df['QualityScore']*df['MispricingPressure']*
                df['FundingVacuum']*df['LiquidityCapacity']*df['TrapGuard']).clip(0,1)
ic,icir,nwt,pos=ic_summary(df,'hamr_doc')
print('  FULL 263d: IC={:+.4f} ICIR={:+.3f} NWt={:+.3f} IC>0={:.0%}'.format(ic,icir,nwt,pos))
ic_t,icir_t,nwt_t,pos_t=ic_summary(df[df['date'].isin(train_dates)],'hamr_doc')
print('  TRAIN 180d: IC={:+.4f} ICIR={:+.3f} NWt={:+.3f} IC>0={:.0%}'.format(ic_t,icir_t,nwt_t,pos_t))
ic_os,icir_os,nwt_os,pos_os=ic_summary(df[df['date'].isin(test_dates)],'hamr_doc')
print('  TEST   83d: IC={:+.4f} ICIR={:+.3f} NWt={:+.3f} IC>0={:.0%}'.format(ic_os,icir_os,nwt_os,pos_os))

print()

# Test 2: Simplified
print('='*60)
print('TEST 2: SIMPLIFIED (AS * RW)')
print('='*60)
df['hamr_simple']=(df['AIStateScore']*df['ResidualWeakness']).clip(0,1)
ic,icir,nwt,pos=ic_summary(df,'hamr_simple')
print('  FULL 263d: IC={:+.4f} ICIR={:+.3f} NWt={:+.3f} IC>0={:.0%}'.format(ic,icir,nwt,pos))
ic_t,icir_t,nwt_t,pos_t=ic_summary(df[df['date'].isin(train_dates)],'hamr_simple')
print('  TRAIN 180d: IC={:+.4f} ICIR={:+.3f} NWt={:+.3f} IC>0={:.0%}'.format(ic_t,icir_t,nwt_t,pos_t))
ic_os,icir_os,nwt_os,pos_os=ic_summary(df[df['date'].isin(test_dates)],'hamr_simple')
print('  TEST   83d: IC={:+.4f} ICIR={:+.3f} NWt={:+.3f} IC>0={:.0%}'.format(ic_os,icir_os,nwt_os,pos_os))

print()

# Test 3: Placebo
print('='*60)
print('TEST 3: PLACEBO (1000 random shuffles)')
print('='*60)
real_ic,_,_,_=ic_summary(df,'hamr_simple')
null_ics=[]
np.random.seed(42)
for _ in range(1000):
    df['hamr_rand']=np.random.permutation(df['hamr_simple'].values)
    nic,_,_,_=ic_summary(df,'hamr_rand')
    null_ics.append(nic)
null_ics=np.array(null_ics)
null_ics=null_ics[np.isfinite(null_ics)]
p_val=np.mean(null_ics>=real_ic)
print('  Real IC: {:.4f}'.format(real_ic))
print('  Null 95% CI: [{:.4f}, {:.4f}]'.format(np.percentile(null_ics,2.5),np.percentile(null_ics,97.5)))
print('  P-value: {:.4f}'.format(p_val))
print('  {}'.format('PASS (p<0.05)' if p_val<0.05 else 'FAIL — not significant vs noise'))

print()

# Test 4: Walk-forward
print('='*60)
print('TEST 4: WALK-FORWARD (60d train, 20d test)')
print('='*60)
window=60;test_w=20
rolling=[]
for i in range(window,len(dates)-test_w,test_w):
    train_p=dates[i-window:i]
    test_p=dates[i:i+test_w]
    ic_t,_,_,_=ic_summary(df[df['date'].isin(train_p)],'hamr_simple')
    ic_s,_,_,_=ic_summary(df[df['date'].isin(test_p)],'hamr_simple')
    rolling.append({'end':test_p[-1],'train_ic':ic_t,'test_ic':ic_s})
rd=pd.DataFrame(rolling)
print('  Periods:',len(rd))
print('  Train IC: mean={:.4f} std={:.4f}'.format(rd['train_ic'].mean(),rd['train_ic'].std()))
print('  Test  IC: mean={:.4f} std={:.4f}'.format(rd['test_ic'].mean(),rd['test_ic'].std()))
print('  Test IC>0: {:.0%}'.format((rd['test_ic']>0).mean()))
print('  Correlation: {:.3f}'.format(rd['train_ic'].corr(rd['test_ic'])))
for _,r in rd.iterrows():
    print('    {:>12s}: train={:+.4f} test={:+.4f}'.format(str(r['end'].date()),r['train_ic'],r['test_ic']))
